import os
import json
import base64
import hashlib
from typing import Any, Dict, Optional, Iterator

import httpx
from pydantic import BaseModel, PrivateAttr

# Optional dependency: botocore for SigV4 signing
try:  # pragma: no cover - runtime check
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials
        _HAS_BOTOCORE = True
except Exception:  # pragma: no cover - will be handled at runtime
        _HAS_BOTOCORE = False

# Optional dependency: h2 for HTTP/2 support in httpx
try:  # pragma: no cover - runtime check
    import h2  # type: ignore
    _HAS_H2 = True
except Exception:  # pragma: no cover
    _HAS_H2 = False

class ManualSigV4Model:
    """Minimal model wrapper to call AWS Bedrock via VPC endpoint with SigV4,
    sending the signed request to a corporate Application Gateway without using
    HTTPS proxy CONNECT.

    Contract:
    - complete(prompt: str) -> str
      Sends a non-streaming request and returns the text output.
        - stream_complete(prompt: str) -> Iterator[str]
            Streaming variant using InvokeModelWithResponseStream; yields textual
            deltas as they arrive.

    Environment variables (or constructor params):
    - AWS_REGION: AWS region (e.g., us-east-1)
    - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional)
    - AWS_BEDROCK_VPC_ENDPOINT: full https URL for Host A (vpce-*.vpce.amazonaws.com)
    - AWS_BEDROCK_MODEL_ID: e.g., anthropic.claude-3-5-sonnet-20240620-v1
    - CORPORATE_GATEWAY_URL: full https URL for Host B (e.g., https://aws-ia-dev.../claude)
      If not set, falls back to CORPORATE_PROXY for backward compatibility.
    - PROXY_SSL_VERIFY: "false" to disable TLS verification when calling Host B (testing only)
    - DEBUG_AWS_REQUESTS: enable debug prints
    """

    def __init__(
        self,
        *,
        region: str,
        access_key: str,
        secret_key: str,
        session_token: Optional[str],
        vpc_endpoint_url: str,
        model_id: str,
        gateway_url: str,
        verify_tls: bool = True,
        timeout: float = 30.0,
        debug: bool = False,
    ) -> None:
        if not _HAS_BOTOCORE:
            raise RuntimeError(
                "ManualSigV4Model requires botocore. Please add 'botocore' to requirements and install."
            )

        self.region = region
        self.creds = Credentials(access_key, secret_key, session_token)
        self.vpc_endpoint_url = vpc_endpoint_url.rstrip("/")
        self.model_id = model_id
        self.gateway_url = gateway_url.rstrip("/")
        self.verify_tls = verify_tls
        self.timeout = timeout
        self.debug = debug

        # Resolve VPC host (Host A)
        from urllib.parse import urlparse
        parsed = urlparse(self.vpc_endpoint_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("AWS_BEDROCK_VPC_ENDPOINT must be a full https URL")
        self.vpc_host = parsed.netloc

        # Pre-create HTTP client to Host B (Application Gateway)
        # Use HTTP/2 only if h2 is available; fallback to HTTP/1.1 otherwise.
        http2_enabled = _HAS_H2
        try:
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.timeout),
                verify=self.verify_tls,
                trust_env=False,  # Avoid picking HTTPS_PROXY; we are NOT using CONNECT
                http2=http2_enabled,
            )
        except ImportError:
            # Safety net: if http2 import error occurs, retry without http2
            self.client = httpx.Client(
                timeout=httpx.Timeout(self.timeout),
                verify=self.verify_tls,
                trust_env=False,
                http2=False,
            )

    @classmethod
    def from_env(cls) -> "ManualSigV4Model":
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        vpc_endpoint_url = os.getenv("AWS_BEDROCK_VPC_ENDPOINT", "").strip()
        model_id = os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1")
        gateway_url = os.getenv("CORPORATE_GATEWAY_URL") or os.getenv("CORPORATE_PROXY", "")
        verify_tls = str(os.getenv("PROXY_SSL_VERIFY", "true")).lower() not in {"0", "false", "no"}
        debug = str(os.getenv("DEBUG_AWS_REQUESTS", "")).lower() in {"1", "true", "yes"}

        missing = []
        if not access_key:
            missing.append("AWS_ACCESS_KEY_ID")
        if not secret_key:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if not vpc_endpoint_url:
            missing.append("AWS_BEDROCK_VPC_ENDPOINT")
        if not gateway_url:
            missing.append("CORPORATE_GATEWAY_URL (or CORPORATE_PROXY)")
        if missing:
            raise RuntimeError(f"Missing required env vars for ManualSigV4Model: {', '.join(missing)}")

        return cls(
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            vpc_endpoint_url=vpc_endpoint_url,
            model_id=model_id,
            gateway_url=gateway_url,
            verify_tls=verify_tls,
            timeout=float(os.getenv("AWS_HTTP_TIMEOUT", "30")),
            debug=debug,
        )

    # --- Public API used by Google ADK agents ---
    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Simple completion wrapper using Bedrock Claude messages API via InvokeModel.

        Returns the first text segment found in the response.
        """
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
        }

        resp = self._invoke_bedrock(payload)
        # Try multiple shapes commonly returned by Bedrock providers
        # 1) Anthropic-style messages
        try:
            content = resp.get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                    return str(part["text"]).strip()
        except Exception:
            pass

        # 2) Converse API shape
        try:
            msg = resp.get("output", {}).get("message", {})
            for part in msg.get("content", []):
                if "text" in part:
                    return str(part["text"]).strip()
        except Exception:
            pass

        # 3) Titan-style
        if "outputText" in resp:
            return str(resp["outputText"]).strip()

        # Fallback: raw JSON
        return json.dumps(resp)

    def stream_complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> Iterator[str]:
        """Stream text deltas from Bedrock using InvokeModelWithResponseStream.

        Yields small text chunks as they arrive. Currently optimized for
        Anthropic Claude on Bedrock, but will attempt to surface text from other
        providers (e.g., Titan) when possible.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": max(1, int(max_tokens)),
            "temperature": float(temperature),
            # You can add additional provider params here if needed
        }

        path = f"/model/{self.model_id}/invoke-with-response-stream"

        # For streaming, include Accept header so Bedrock returns eventstream
        body_bytes = json.dumps(body).encode("utf-8")
        signed = self._build_signed_request(
            method="POST",
            path=path,
            query_string="",
            body=body_bytes,
        )

        # Add Accept header (not required for signature validation to pass),
        # OK to add even if not originally signed because SigV4 validates the
        # canonical signed headers subset.
        headers = dict(signed["headers"])  # copy
        headers.setdefault("accept", "application/vnd.amazon.eventstream")

        url = self.gateway_url
        if self.debug:
            print("[ManualSigV4] STREAM POST ->", url)
            print("[ManualSigV4] Using Host header for VPC:", headers.get("host"))
            print("[ManualSigV4] Authorization prefix:", headers.get("authorization", "")[:48], "...")

        with self.client.stream("POST", url, headers=headers, content=body_bytes) as r:
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            if "application/vnd.amazon.eventstream" in ctype:
                # Parse AWS eventstream frames and surface provider text
                yield from self._iter_eventstream_text(r)
            else:
                # Fallback: not a stream, try to parse once and yield
                try:
                    data = r.json()
                except Exception:
                    data = {"raw": r.text}
                text = self._first_text_from_response(data)
                if text:
                    yield text

    # --- Internal helpers for streaming ---
    def _first_text_from_response(self, resp: Dict[str, Any]) -> str:
        # Anthropic-style messages
        try:
            content = resp.get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                    return str(part["text"]).strip()
        except Exception:
            pass

        # Converse API shape
        try:
            msg = resp.get("output", {}).get("message", {})
            for part in msg.get("content", []):
                if "text" in part:
                    return str(part["text"]).strip()
        except Exception:
            pass

        if "outputText" in resp:
            return str(resp["outputText"]).strip()
        return ""

    def _iter_eventstream_text(self, response: httpx.Response) -> Iterator[str]:
        """Decode AWS eventstream frames and yield text deltas.

        Format reference (simplified):
        - Message prelude: 4B total_len (big-endian), 4B headers_len (big-endian), 4B prelude_crc
        - Headers: headers_len bytes (ignored here)
        - Payload: payload_len bytes where payload_len = total_len - headers_len - 16
        - Message CRC: 4B (ignored)

        Inside the payload, Bedrock wraps provider bytes in JSON events like:
          {"chunk": {"bytes": "<base64>"}}  -> base64-decoded bytes contain NDJSON
        For Anthropic, NDJSON lines include objects like:
          {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
        """
        buf = bytearray()
        for raw in response.iter_bytes():
            if not raw:
                continue
            buf.extend(raw)
            # Attempt to extract as many frames as available
            while True:
                if len(buf) < 12:
                    break
                total_len = int.from_bytes(buf[0:4], "big")
                headers_len = int.from_bytes(buf[4:8], "big")
                # prelude_crc = buf[8:12]  # ignored
                if total_len <= 0 or len(buf) < total_len:
                    break

                pos = 12
                # Skip headers
                pos += headers_len
                payload_len = total_len - headers_len - 16  # minus prelude(12) and message_crc(4)
                if payload_len < 0 or pos + payload_len + 4 > len(buf):
                    # Incomplete; wait for more
                    break

                payload = bytes(buf[pos:pos + payload_len])
                # message_crc = buf[pos + payload_len: pos + payload_len + 4]  # ignored

                # Consume this frame
                del buf[:total_len]

                # Try to parse payload JSON (Bedrock event)
                try:
                    ev = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue

                # Common Bedrock stream keys: chunk, internalServerException, modelStreamErrorException, etc.
                if isinstance(ev, dict) and "chunk" in ev and isinstance(ev["chunk"], dict):
                    b64 = ev["chunk"].get("bytes")
                    if isinstance(b64, str) and b64:
                        try:
                            provider_bytes = base64.b64decode(b64)
                            for piece in self._iter_provider_stream(provider_bytes):
                                if piece:
                                    yield piece
                        except Exception:
                            # If decode fails, ignore this chunk
                            continue
                elif isinstance(ev, dict) and "message" in ev:
                    # Some providers may send message-like JSON directly
                    text = self._first_text_from_response(ev)
                    if text:
                        yield text
                else:
                    # Unknown event type: ignore or surface minimal info
                    continue

    def _iter_provider_stream(self, data: bytes) -> Iterator[str]:
        """Iterate over provider NDJSON events and yield text deltas.

        Anthropic over Bedrock typically returns NDJSON events where each line
        is a JSON object representing a stream event.
        """
        try:
            s = data.decode("utf-8", errors="ignore")
        except Exception:
            return

        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Anthropic streaming shapes
            if obj.get("type") == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    yield delta["text"]
            elif obj.get("type") == "message_delta":
                # Some models send trailing updates; no direct text chunk
                continue
            elif obj.get("type") in {"message_start", "content_block_start", "message_stop"}:
                # Control events; ignore
                continue
            else:
                # Try generic text holders if present
                if isinstance(obj, dict):
                    if "outputText" in obj and isinstance(obj["outputText"], str):
                        yield obj["outputText"]
                    elif "text" in obj and isinstance(obj["text"], str):
                        yield obj["text"]

    # --- Internal helpers ---
    def _invoke_bedrock(self, body: Dict[str, Any]) -> Dict[str, Any]:
        path = f"/model/{self.model_id}/invoke"  # non-streaming endpoint
        signed = self._build_signed_request(
            method="POST",
            path=path,
            query_string="",
            body=json.dumps(body).encode("utf-8"),
        )

        # Send to Host B (Application Gateway) as a regular HTTPS POST
        url = self.gateway_url  # Full URL like https://aws-ia-dev.../claude
        if self.debug:
            print("[ManualSigV4] POST ->", url)
            print("[ManualSigV4] Using Host header for VPC:", signed["headers"].get("host"))
            print("[ManualSigV4] Authorization prefix:", signed["headers"].get("authorization", "")[:48], "...")

        # Important: we forward the exact signed headers so AWS validates SigV4.
        # The gateway must transparently forward method, path, headers and body to Host A.
        r = self.client.post(
            url,
            headers=signed["headers"],
            content=signed["body"],
        )
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            # Some gateways wrap payload; try to unwrap or return raw
            return {"raw": r.text}

    def _build_signed_request(self, *, method: str, path: str, query_string: str, body: bytes) -> Dict[str, Any]:
        # Construct the URL for Host A (used only for signing values)
        url = f"https://{self.vpc_host}{path}{('?' + query_string) if query_string else ''}"

        # Compute SHA256 payload hash
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {
            "host": self.vpc_host,
            "content-type": "application/json",
            "x-amz-content-sha256": payload_hash,
        }

        if self.creds.token:
            headers["x-amz-security-token"] = self.creds.token

        # Sign the request for service 'bedrock' in the given region
        aws_req = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(self.creds, "bedrock", self.region).add_auth(aws_req)

        # botocore converts headers to a HTTPHeaders object; normalize to str dict
        signed_headers = {k.lower(): v for k, v in aws_req.headers.items()}

        if self.debug:
            print("[ManualSigV4] Signed URL:", url)
            print("[ManualSigV4] Headers:")
            for k in ["host", "authorization", "x-amz-date", "x-amz-content-sha256", "x-amz-security-token"]:
                if k in signed_headers:
                    val = signed_headers[k]
                    if k == "authorization":
                        val = val[:72] + "..."
                    print(f"  - {k}: {val}")

        return {"url": url, "headers": signed_headers, "body": body}


def manual_bedrock_model() -> ManualSigV4Model:
    """Explicit factory for ManualSigV4Model using env vars."""
    return ManualSigV4Model.from_env()


class ManualSigV4Adapter(BaseModel):
    """Pydantic-friendly adapter that wraps ManualSigV4Model instance.

    This makes the runtime client an instance of a BaseModel so frameworks
    that validate model inputs with pydantic (like google.adk) accept it.

    The actual HTTP/Signing client is stored in a private attribute and
    delegated to for `complete` and `stream_complete` calls.
    """
    model_type: str = "manual_sigv4"
    # runtime-only client
    _client: ManualSigV4Model = PrivateAttr()

    @classmethod
    def from_env(cls) -> "ManualSigV4Adapter":
        inst = cls()
        inst._client = ManualSigV4Model.from_env()
        return inst

    def complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        return self._client.complete(prompt, max_tokens=max_tokens, temperature=temperature)

    def stream_complete(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> Iterator[str]:
        return self._client.stream_complete(prompt, max_tokens=max_tokens, temperature=temperature)
