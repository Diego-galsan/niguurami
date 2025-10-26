import os
import json
import base64
import hashlib
from typing import Any, Dict, Optional, Iterator

import httpx
from pydantic import BaseModel, PrivateAttr
import asyncio
from typing import AsyncGenerator, List
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

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
        # Preserve the gateway URL exactly as provided. Some gateways rely on
        # a trailing slash for routing and will respond with 301 if it's
        # missing. We no longer strip trailing slashes here.
        self.gateway_url = gateway_url
        self.verify_tls = verify_tls
        self.timeout = timeout
        self.debug = debug

        # Resolve VPC host (Host A)
        from urllib.parse import urlparse
        parsed = urlparse(self.vpc_endpoint_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("AWS_BEDROCK_VPC_ENDPOINT must be a full https URL")
        self.vpc_host = parsed.netloc

        # Internal flag to stop consuming a stream when provider signals completion
        self._stream_should_stop = False

        # Pre-create HTTP client to Host B (Application Gateway)
        # Use HTTP/2 only if h2 is available; fallback to HTTP/1.1 otherwise.
        http2_enabled = _HAS_H2
        # Granular timeouts to avoid hangs on slow proxies/streams
        timeout_cfg = httpx.Timeout(
            connect=self.timeout,
            read=self.timeout,
            write=self.timeout,
            pool=self.timeout,
        )
        try:
            self.client = httpx.Client(
                timeout=timeout_cfg,
                verify=self.verify_tls,
                trust_env=False,  # Avoid picking HTTPS_PROXY; we are NOT using CONNECT
                http2=http2_enabled,
            )
        except ImportError:
            # Safety net: if http2 import error occurs, retry without http2
            self.client = httpx.Client(
                timeout=timeout_cfg,
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

        # Build the forward URL by joining the gateway base path with the Bedrock path
        url = self._gateway_join(path)
        if self.debug:
            print("[ManualSigV4] STREAM POST ->", url)
            print("[ManualSigV4] Using Host header for VPC:", headers.get("host"))
            print("[ManualSigV4] Authorization prefix:", headers.get("authorization", "")[:48], "...")

        # reset stop flag for this stream
        self._stream_should_stop = False
        with self.client.stream("POST", url, headers=headers, content=body_bytes) as r:
            # Handle trailing-slash redirects for streaming as well
            if r.status_code in (301, 302, 307, 308):
                loc = r.headers.get("location") or r.headers.get("Location")
                if loc:
                    redirect_url = self._build_redirect_url(url, loc, path)
                    if self.debug:
                        print(f"[ManualSigV4] STREAM redirect {r.status_code} -> {redirect_url}")
                    r.close()
                    # Re-open the stream at the redirect target
                    with self.client.stream("POST", redirect_url, headers=headers, content=body_bytes) as r2:
                        if r2.status_code >= 400:
                            self._debug_http_error(r2, "stream redirect target")
                        r2.raise_for_status()
                        ctype = (r2.headers.get("content-type") or "").lower()
                        if "application/vnd.amazon.eventstream" in ctype:
                            yield from self._iter_eventstream_text(r2)
                        else:
                            try:
                                data = r2.json()
                            except Exception:
                                data = {"raw": r2.text}
                            text = self._first_text_from_response(data)
                            if text:
                                yield text
                    return
            if r.status_code >= 400:
                self._debug_http_error(r, "stream initial")
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
                            if self._stream_should_stop:
                                return
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
            elif obj.get("type") in {"message_start", "content_block_start"}:
                # Control events; ignore
                continue
            elif obj.get("type") == "message_stop":
                # Signal outer loop to stop consuming
                self._stream_should_stop = True
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

        # Send to Host B (Application Gateway) joining base gateway path with Bedrock path
        url = self._gateway_join(path)
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
        # Handle common gateway redirects caused by missing trailing slash or host change
        if r.status_code in (301, 302, 307, 308):
            loc = r.headers.get("location") or r.headers.get("Location")
            if loc:
                redirect_url = self._build_redirect_url(url, loc, path)
                if self.debug:
                    print(f"[ManualSigV4] Redirect {r.status_code} -> {redirect_url}")
                # Re-issue POST to the redirect target preserving body/headers
                r.close()
                r = self.client.post(
                    redirect_url,
                    headers=signed["headers"],
                    content=signed["body"],
                )
        if r.status_code >= 400:
            self._debug_http_error(r, "invoke")
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            # Some gateways wrap payload; try to unwrap or return raw
            return {"raw": r.text}

    # --- URL helper ---
    def _gateway_join(self, extra_path: str) -> str:
        """Join the corporate gateway base URL with the Bedrock path.

        Preserves gateway base path segments (e.g., '/claude') and appends the
        Bedrock path (e.g., '/model/...'). Avoids losing base path which can
        happen with urljoin when the second path starts with '/'.
        """
        from urllib.parse import urlparse, urlunparse

        base = self.gateway_url
        bp = urlparse(base)
        base_path = (bp.path or "/").strip("/")
        add_path = (extra_path or "").strip("/")
        pieces = [p for p in [base_path, add_path] if p]
        joined_path = "/" + "/".join(pieces)
        return urlunparse((bp.scheme, bp.netloc, joined_path, "", "", ""))

    def _build_redirect_url(self, current_url: str, location: str, intended_path: str) -> str:
        """Build a safe redirect target.

        - If Location is absolute and points to a different host (e.g., directly to the VPC endpoint),
          reconstruct the target using the intended Bedrock path to avoid wrong prefixes like '/claude/'.
        - Otherwise, resolve relative redirects against the current URL.
        """
        from urllib.parse import urlparse, urljoin, urlunparse

        cur = urlparse(current_url)
        loc = urlparse(location)

        # Absolute redirect to another host (often the VPC endpoint)
        if loc.scheme and loc.netloc and (loc.netloc != cur.netloc):
            # Keep scheme+host from Location, but enforce correct Bedrock path
            new_path = intended_path if intended_path.startswith("/") else "/" + intended_path
            return urlunparse((loc.scheme, loc.netloc, new_path, "", "", ""))

        # Otherwise, relative or same-host redirect; resolve normally and keep path
        base = current_url + ("/" if not current_url.endswith("/") else "")
        return urljoin(base, location)

    def _debug_http_error(self, r: httpx.Response, where: str) -> None:
        """Print helpful diagnostics for HTTP errors when debug is enabled."""
        if not self.debug:
            return
        try:
            body = r.json()
        except Exception:
            body = r.text
        # Avoid dumping sensitive auth header
        headers_sample = {k: (v if k.lower() != "authorization" else "<redacted>") for k, v in r.request.headers.items()}
        print(f"[ManualSigV4] HTTP error at {where}: {r.status_code} {r.reason_phrase}")
        try:
            print("[ManualSigV4] URL:", str(r.request.url))
        except Exception:
            pass
        print("[ManualSigV4] Request headers:", headers_sample)
        print("[ManualSigV4] Response body:", body)

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


class ManualSigV4Adapter(BaseLlm):
    """Pydantic-friendly adapter that wraps ManualSigV4Model instance.

    This makes the runtime client an instance of a BaseModel so frameworks
    that validate model inputs with pydantic (like google.adk) accept it.

    The actual HTTP/Signing client is stored in a private attribute and
    delegated to for `complete` and `stream_complete` calls.
    """
    model_type: str = "manual_sigv4"
    model: str = "aws/bedrock.manual-sigv4"
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

    # ---- BaseLlm required methods ----
    @classmethod
    def supported_models(cls) -> List[str]:
        # Not used by registry in our integration; return a permissive list
        return [".*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Bridge ADK LlmRequest to our client and yield LlmResponse.

        - Non-streaming: one LlmResponse with full text and turn_complete=True.
        - Streaming: yield partial chunks as LlmResponse(partial=True), then a
          final LlmResponse(turn_complete=True).
        """
        prompt = _contents_to_prompt(llm_request.contents)
        # Pull common params if user set them in config
        max_tokens = getattr(getattr(llm_request, 'config', None), 'max_output_tokens', 256) or 256
        temperature = getattr(getattr(llm_request, 'config', None), 'temperature', 0.2) or 0.2

        if not stream:
            text = await asyncio.to_thread(self.complete, prompt, max_tokens, float(temperature))
            yield _make_response(text, partial=False, turn_complete=True)
            return

        # Streaming path
        accumulated = []
        async def _iter_stream():
            for delta in self.stream_complete(prompt, max_tokens=max_tokens, temperature=float(temperature)):
                yield delta

        async for delta in _iter_stream():
            if not delta:
                continue
            accumulated.append(delta)
            yield _make_response(delta, partial=True, turn_complete=False)

        # Final message (optional, to indicate completion)
        final_text = "".join(accumulated)
        yield _make_response(final_text, partial=False, turn_complete=True)

    def connect(self, llm_request: LlmRequest):  # noqa: D401 - documented in base
        # Live connections are not supported for this manual client.
        # ADK will fall back to generate_content_async when connect is not used.
        raise NotImplementedError('Live connection is not supported for ManualSigV4Adapter')


# ---- Small helpers to bridge ADK types to our client ----
def _contents_to_prompt(contents: List[genai_types.Content]) -> str:
    """Extract a user-friendly prompt by concatenating text parts."""
    if not contents:
        return ""
    parts: List[str] = []
    for c in contents:
        for p in getattr(c, 'parts', []) or []:
            txt = getattr(p, 'text', None)
            if isinstance(txt, str) and txt:
                parts.append(txt)
    return "\n\n".join(parts).strip()


def _make_response(text: str, *, partial: bool, turn_complete: bool) -> LlmResponse:
    content = genai_types.Content(
        role='model',
        parts=[genai_types.Part(text=text or '')],
    )
    return LlmResponse(
        content=content,
        partial=partial,
        turn_complete=turn_complete,
    )
