import os
import boto3
import httpx  # Asegúrate de instalarlo: pip install httpx
import json
import asyncio
from datetime import datetime
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from google.adk.core import Model
# ❗ Importamos el decodificador de eventos oficial de botocore
from botocore.eventstream import EventStreamParser

# --- 1. CONFIGURACIÓN DE ENDPOINTS (LEÍDOS DESDE EL ENTORNO) ---
VPC_ENDPOINT_URL = os.environ.get("VPC_ENDPOINT_URL")
PROXY_EXECUTION_URL = os.environ.get("PROXY_EXECUTION_URL")
AWS_REGION = os.environ.get("AWS_REGION_NAME")

if not all([VPC_ENDPOINT_URL, PROXY_EXECUTION_URL, AWS_REGION]):
    raise ValueError(
        "Error: Faltan variables de entorno requeridas: "
        "VPC_ENDPOINT_URL, PROXY_EXECUTION_URL, o AWS_REGION_NAME"
    )

VPC_HOST = VPC_ENDPOINT_URL.split("://")[1]


class ManualSigV4Model(Model):
    """
    Versión ROBUSTA y CON STREAMING del modelo manual.
    """

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id.split('/')[-1] if '/' in model_id else model_id

        session = boto3.Session()
        self._creds = session.get_credentials()
        if not self._creds:
            raise Exception("No se pudieron obtener credenciales de AWS.")

        transport = httpx.AsyncHTTPTransport(retries=3)
        self._client = httpx.AsyncClient(transport=transport, timeout=60.0)

        print(f"INFO: Modelo ManualSigV4 (Streaming) inicializado.")
        print(f"INFO:   Firmará para el Host: {VPC_HOST}")
        print(f"INFO:   Enviará a la URL: {PROXY_EXECUTION_URL}")
        print(f"INFO:   Modelo Bedrock: {self.model_id}")

    def _get_signed_headers(self, method, url_path, data_body):
        # (Este método no cambia)
        request = AWSRequest(
            method=method,
            url=f"{VPC_ENDPOINT_URL}{url_path}",
            data=data_body
        )
        request.headers['Host'] = VPC_HOST
        request.headers['Content-Type'] = 'application/json'
        SigV4Auth(self._creds, "bedrock", AWS_REGION).add_auth(request)
        return request.headers

    async def generate(self, *args, **kwargs):
        """
        Llamado por el ADK para generar una respuesta.
        Esta versión usa 'yield' para soportar streaming.
        """
        messages = kwargs.get("prompt", [])
        if not messages:
            raise ValueError("No se recibieron 'messages' en el prompt.")
        
        body = {
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            # ... puedes añadir 'system' o 'inferenceConfig' aquí si los necesitas
        }
        body_json = json.dumps(body)

        # --- CAMBIOS PARA STREAMING ---
        
        # 1. Usamos el endpoint de streaming
        url_path = f"/model/{self.model_id}/converse-stream"

        signed_headers = self._get_signed_headers(
            method="POST",
            url_path=url_path,
            data_body=body_json
        )
        
        execution_url = f"{PROXY_EXECUTION_URL}{url_path}"

        # 2. Creamos el parser de eventos
        parser = EventStreamParser()

        try:
            print(f"INFO: Iniciando POST de streaming a: {execution_url}")

            # 3. Usamos 'self._client.stream' en lugar de 'post'
            async with self._client.stream(
                "POST",
                execution_url,
                content=body_json,
                headers=signed_headers
            ) as response:
                
                # Lanzar error si la conexión inicial falla
                response.raise_for_status()

                # 4. Iteramos sobre los bytes crudos a medida que llegan
                async for chunk in response.aiter_bytes():
                    # 5. Alimentamos los bytes al parser de botocore
                    parser.feed(chunk)
                    
                    # 6. Iteramos sobre los eventos completos que el parser encuentra
                    for event in parser.events():
                        # 'event' es un diccionario con 'headers' y 'payload'
                        payload_bytes = event.get('payload')
                        if payload_bytes:
                            try:
                                # 7. Decodificamos el payload (que es JSON)
                                payload = json.loads(payload_bytes)
                                
                                # 8. Buscamos el trozo de texto (delta)
                                if 'contentBlockDelta' in payload:
                                    text_chunk = payload['contentBlockDelta'].get('delta', {}).get('text', '')
                                    if text_chunk:
                                        # 9. 'yield' (producimos) el trozo de texto al ADK
                                        yield text_chunk
                                
                                # Opcional: manejar otros eventos como 'messageStop'

                            except json.JSONDecodeError:
                                print(f"ADVERTENCIA: No se pudo decodificar el payload JSON del evento: {payload_bytes}")

        except httpx.HTTPStatusError as e:
            # Imprimir el cuerpo del error si es posible
            error_body = await e.response.aread()
            print(f"ERROR HTTP: {e.response.status_code}. Respuesta: {error_body.decode()}")
            raise e
        except Exception as e:
            print(f"Error en generate (streaming): {e}")
            raise
