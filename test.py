import os
import boto3
import httpx  # Asegúrate de instalarlo: pip install httpx
import json
import asyncio
from datetime import datetime
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from google.adk.core import Model

# --- 1. CONFIGURACIÓN DE ENDPOINTS (LEÍDOS DESDE EL ENTORNO) ---

# La URL del VPCe real para la FIRMA SigV4
# ej: https://vpce-0d9383...amazonaws.com
VPC_ENDPOINT_URL = os.environ.get("VPC_ENDPOINT_URL")

# La URL del Gateway de Aplicación para la EJECUCIÓN
# ej: https://aws-ia-dev.banregio.com
PROXY_EXECUTION_URL = os.environ.get("PROXY_EXECUTION_URL")

# La región de AWS donde está el servicio
# ej: us-east-1
AWS_REGION = os.environ.get("AWS_REGION_NAME")

# --- VERIFICACIÓN DE VARIABLES ---
if not all([VPC_ENDPOINT_URL, PROXY_EXECUTION_URL, AWS_REGION]):
    raise ValueError(
        "Error: Faltan variables de entorno requeridas: "
        "VPC_ENDPOINT_URL, PROXY_EXECUTION_URL, o AWS_REGION_NAME"
    )

# Extraer el host (ej. vpce-....amazonaws.com) de la URL para la firma
VPC_HOST = VPC_ENDPOINT_URL.split("://")[1]


class ManualSigV4Model(Model):
    """
    Un modelo de ADK robusto que:
    1. Firma solicitudes SigV4 para un VPC_HOST (Host A).
    2. Envía la solicitud como un POST HTTP a un PROXY_EXECUTION_URL (Host B).
    3. Maneja reintentos de conexión y errores 429/5xx.
    """

    def __init__(self, model_id: str):
        super().__init__()
        # model_id debe ser el ID de Bedrock SIN el prefijo 'bedrock/'
        # ej: "anthropic.claude-3-haiku-20240307-v1:0"
        self.model_id = model_id.split('/')[-1] if '/' in model_id else model_id

        # 1. Obtener credenciales del rol IAM (asumido por el pod de OpenShift)
        session = boto3.Session()
        self._creds = session.get_credentials()
        if not self._creds:
            raise Exception("No se pudieron obtener credenciales de AWS.")

        # 2. Configurar cliente httpx con reintentos de conexión y timeouts
        transport = httpx.AsyncHTTPTransport(retries=3)  # Reintentos de conexión
        self._client = httpx.AsyncClient(transport=transport, timeout=60.0)

        print(f"INFO: Modelo ManualSigV4 (Robusto) inicializado.")
        print(f"INFO:   Firmará para el Host: {VPC_HOST}")
        print(f"INFO:   Enviará a la URL: {PROXY_EXECUTION_URL}")
        print(f"INFO:   Modelo Bedrock: {self.model_id}")

    def _get_signed_headers(self, method, url_path, data_body):
        """
        Usa botocore para crear una firma SigV4 para el VPC_ENDPOINT.
        Esta es la parte de "firmar para Host A".
        """
        # 1. Crear la solicitud base para el VPCe
        request = AWSRequest(
            method=method,
            url=f"{VPC_ENDPOINT_URL}{url_path}",  # URL de firma
            data=data_body
        )

        # 2. Establecer encabezados requeridos para la firma
        request.headers['Host'] = VPC_HOST  # Host de firma
        request.headers['Content-Type'] = 'application/json'

        # 3. Firmar la solicitud
        SigV4Auth(self._creds, "bedrock", AWS_REGION).add_auth(request)

        # 4. Retornar solo los encabezados firmados
        return request.headers

    async def generate(self, *args, **kwargs):
        """
        Llamado por el ADK para generar una respuesta.
        Esta es la parte de "enviar a Host B".
        """
        # 1. Obtener los mensajes del ADK
        #    El 'Generator' del ADK los pasa en el kwarg 'prompt'
        messages = kwargs.get("prompt", [])
        if not messages:
            raise ValueError("No se recibieron 'messages' en el prompt.")
        
        # 2. Preparar el body para la API Bedrock Converse (Claude)
        body = {
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31", # Requerido por Claude
            "max_tokens": 2048
            # ... puedes añadir 'system' o 'inferenceConfig' aquí si los necesitas
        }
        body_json = json.dumps(body)

        # 3. Definir el path de la API de Bedrock
        url_path = f"/model/{self.model_id}/converse"

        # 4. Obtener los encabezados firmados para el VPCe
        signed_headers = self._get_signed_headers(
            method="POST",
            url_path=url_path,
            data_body=body_json
        )

        # 5. Construir la URL de EJECUCIÓN (el Gateway)
        execution_url = f"{PROXY_EXECUTION_URL}{url_path}"

        # 6. Lógica de reintentos de API (429, 5xx)
        max_retries = 3
        base_delay = 1.0  # 1 segundo

        for attempt in range(max_retries):
            try:
                print(f"INFO: Enviando POST manual a: {execution_url} (Intento {attempt+1})")

                # ESTA ES LA LLAMADA CLAVE:
                # Un POST HTTP normal, que SÍ evita el error 400.
                response = await self._client.post(
                    execution_url,
                    content=body_json,
                    headers=signed_headers
                )

                # Errores 4xx (excepto 429) NO son reintentables.
                # Son errores de configuración (400, 401, 403) y deben fallar.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    print(f"ERROR HTTP {response.status_code} (No reintentable): {response.text}")
                    response.raise_for_status()  # Lanza la excepción y detiene

                # Lanza excepción para 429 y 5xx para que sean capturados abajo
                response.raise_for_status()

                # --- ÉXITO ---
                response_data = response.json()
                
                # Extraer contenido de la API Converse
                content = response_data.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")

                if not content:
                     # Fallback por si la respuesta es diferente
                     content = str(response_data)

                return content  # Devuelve el éxito y sale del bucle

            except (httpx.HTTPStatusError, httpx.NetworkError) as e:
                # Revisa si es un error reintentable
                is_retriable = False
                status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None

                if isinstance(e, httpx.NetworkError):
                    is_retriable = True  # Error de conexión
                elif status == 429:  # Too Many Requests
                    is_retriable = True
                elif status in [500, 502, 503, 504]:  # Errores de Servidor
                    is_retriable = True

                if is_retriable and (attempt < max_retries - 1):
                    # Pausa con "exponential backoff" antes de reintentar
                    delay = (base_delay * (2 ** attempt)) + (os.urandom(1)[0] / 255.0)
                    print(f"ADVERTENCIA: Intento {attempt+1} falló ({e}). Reintentando en {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    # Si no es reintentable o se acabaron los intentos
                    print(f"ERROR: Fallo final después de {attempt+1} intentos.")
                    raise e  # Lanza la excepción final

            except Exception as e:
                # Otro error (ej. JSON parse)
                print(f"Error en generate (no HTTP): {e}")
                raise
