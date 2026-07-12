from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SENSITIVE_QUERY_MARKERS = ("key", "token", "secret", "password", "auth", "credential")


@dataclass(frozen=True)
class EndpointSafety:
    endpoint: str
    sanitized_endpoint: str
    category: str
    host: str
    warnings: List[str]
    requires_explicit_allow: bool


def redact_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.port:
        host = "{0}:{1}".format(host, parts.port)
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if any(marker in key.lower() for marker in SENSITIVE_QUERY_MARKERS):
            query_items.append((key, "REDACTED"))
        else:
            query_items.append((key, value))
    return urlunsplit((parts.scheme, host, parts.path, urlencode(query_items), ""))


def _host_category(host: str) -> str:
    normalized = host.strip().lower().strip("[]")
    if normalized in {"localhost", "ip6-localhost"}:
        return "localhost"
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        if normalized.endswith(".local") or normalized.endswith(".lan"):
            return "private"
        return "public"
    if ip.is_loopback or normalized == "0.0.0.0":
        return "localhost"
    if ip.is_private or ip.is_link_local:
        return "private"
    return "public"


def classify_endpoint(endpoint: str) -> EndpointSafety:
    parts = urlsplit(endpoint)
    warnings: List[str] = []
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return EndpointSafety(endpoint, redact_endpoint(endpoint), "invalid", parts.hostname or "", ["Endpoint must be an http:// or https:// URL with a host."], False)
    category = _host_category(parts.hostname)
    if category == "localhost":
        warnings.append("Endpoint is local to this machine.")
    elif category == "private":
        warnings.append("Endpoint appears to be on a private or LAN address; send only harmless smoke-test prompts.")
    else:
        warnings.append("Endpoint appears public or external; InferDoctor will not send a live smoke-test prompt without explicit confirmation.")
    return EndpointSafety(
        endpoint=endpoint,
        sanitized_endpoint=redact_endpoint(endpoint),
        category=category,
        host=parts.hostname or "",
        warnings=warnings,
        requires_explicit_allow=category in {"private", "public"},
    )


def render_endpoint_safety_error(safety: EndpointSafety) -> str:
    if safety.category == "invalid":
        return "Invalid endpoint: {0}. {1}".format(safety.sanitized_endpoint, " ".join(safety.warnings))
    return (
        "Refusing to send a live smoke-test prompt to non-local endpoint {0}. "
        "If this is your private endpoint and the prompt is harmless, rerun with --allow-non-local."
    ).format(safety.sanitized_endpoint)
