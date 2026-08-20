from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from inferdoctor import __version__
from inferdoctor.core.endpoint_safety import redact_endpoint
from inferdoctor.core.http import join_url


MAX_RESPONSE_BYTES = 1024 * 1024


class OpenAICompatibleTransportError(ConnectionError):
    pass


@dataclass(frozen=True)
class OpenAICompatibleResponse:
    url: str
    status: int
    json_data: Optional[Any]
    json_valid: bool
    elapsed_ms: int
    body_bytes: int


def models_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if urlsplit(normalized).path.rstrip("/").endswith("/v1"):
        return join_url(normalized, "models")
    return join_url(normalized, "v1/models")


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if urlsplit(normalized).path.rstrip("/").endswith("/v1"):
        return join_url(normalized, "chat/completions")
    return join_url(normalized, "v1/chat/completions")


def extract_chat_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
    return first.get("text") if isinstance(first.get("text"), str) else ""


def _headers(api_key: Optional[str], *, has_body: bool) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "InferDoctor/{0}".format(__version__),
    }
    if has_body:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = "Bearer {0}".format(api_key)
    return headers


def _read_bounded(response: Any) -> bytes:
    data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise OpenAICompatibleTransportError(
            "OpenAI-compatible response exceeded 1 MiB"
        )
    return data


def _request_json(
    url: str,
    *,
    method: str,
    timeout: float,
    api_key: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
) -> OpenAICompatibleResponse:
    encoded = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        method=method,
        headers=_headers(api_key, has_body=encoded is not None),
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            getcode = getattr(response, "getcode", None)
            code = getcode() if callable(getcode) else 200
            status = int(code if code is not None else 200)
            data = _read_bounded(response)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        data = _read_bounded(exc)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        message = " ".join(str(exc).split())[:240] or "request failed"
        if api_key:
            message = message.replace(api_key, "REDACTED")
        raise OpenAICompatibleTransportError(message) from exc

    parsed: Optional[Any] = None
    json_valid = False
    if data:
        try:
            parsed = json.loads(data.decode("utf-8"))
            json_valid = True
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

    return OpenAICompatibleResponse(
        url=redact_endpoint(url),
        status=status,
        json_data=parsed,
        json_valid=json_valid,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        body_bytes=len(data),
    )


def list_models(
    base_url: str,
    *,
    timeout: float,
    api_key: Optional[str] = None,
) -> OpenAICompatibleResponse:
    return _request_json(
        models_url(base_url),
        method="GET",
        timeout=timeout,
        api_key=api_key,
    )


def create_chat_completion(
    base_url: str,
    *,
    payload: Dict[str, Any],
    timeout: float,
    api_key: Optional[str] = None,
) -> OpenAICompatibleResponse:
    return _request_json(
        chat_completions_url(base_url),
        method="POST",
        timeout=timeout,
        api_key=api_key,
        payload=payload,
    )
