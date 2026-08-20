from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from inferdoctor import __version__
from inferdoctor.core.endpoint_safety import classify_endpoint, redact_endpoint
from inferdoctor.core.openai_compatible import (
    OpenAICompatibleResponse,
    OpenAICompatibleTransportError,
    create_chat_completion,
    extract_chat_text,
    list_models,
)


PROVIDER_CHECK_SCHEMA_VERSION = "inferdoctor.provider.check.v1"
OPENAI_COMPATIBLE_PROTOCOL = "openai-compatible"
UNSUPPORTED_MODELS_STATUSES = {404, 405, 501}


class ProviderError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    display_name: str
    protocol: str
    base_url: str
    docs_url: str
    signup_url: str
    partner_url: Optional[str]
    api_key_env: str
    default_model: Optional[str]
    capabilities: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        return data


_PROVIDERS = {
    "orcarouter": ProviderPreset(
        id="orcarouter",
        display_name="OrcaRouter",
        protocol=OPENAI_COMPATIBLE_PROTOCOL,
        base_url="https://api.orcarouter.ai/v1",
        docs_url="https://docs.orcarouter.ai/",
        signup_url="https://www.orcarouter.ai/",
        partner_url="https://www.orcarouter.ai/ref/ref_a81451091cc4d54480f8",
        api_key_env="ORCAROUTER_API_KEY",
        default_model="orcarouter/auto",
        capabilities=("models", "chat_completions"),
    )
}


def provider_ids() -> Tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def list_provider_presets() -> List[ProviderPreset]:
    return [_PROVIDERS[item] for item in provider_ids()]


def get_provider_preset(provider_id: str) -> ProviderPreset:
    try:
        return _PROVIDERS[provider_id]
    except KeyError as exc:
        raise ProviderError("unknown provider preset: {0}".format(provider_id)) from exc


def _add_check(
    result: Dict[str, Any],
    name: str,
    status: str,
    summary: str,
    **evidence: Any,
) -> None:
    item = {"name": name, "status": status, "summary": summary}
    if evidence:
        item["evidence"] = evidence
    result["checks"].append(item)


def _new_result(
    provider: ProviderPreset,
    *,
    key_present: bool,
    smoke: bool,
    model: Optional[str],
    endpoint_category: str,
) -> Dict[str, Any]:
    return {
        "schema_version": PROVIDER_CHECK_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inferdoctor_version": __version__,
        "status": "UNKNOWN",
        "provider": provider.to_dict(),
        "endpoint": redact_endpoint(provider.base_url),
        "endpoint_category": endpoint_category,
        "api_key_env": provider.api_key_env,
        "api_key_present": key_present,
        "request_sent": False,
        "smoke_requested": smoke,
        "model": model,
        "checks": [],
        "models_probe": {
            "status": "UNKNOWN",
            "http_status": None,
            "model_count": None,
            "selected_model_listed": None,
        },
        "chat_smoke": {
            "status": "UNKNOWN" if smoke else "SKIP",
            "http_status": None,
            "selected_model_invoked": False,
            "response_content_retained": False,
        },
        "metrics": {"ttft_ms": None, "total_latency_ms": None},
        "pricing": {
            "status": "UNKNOWN",
            "input_cost_per_million_tokens": None,
            "output_cost_per_million_tokens": None,
            "currency": None,
        },
        "cost": {
            "api_cost": None,
            "total_compute_cost": None,
            "currency": None,
        },
        "limitations": [
            "Provider Check is a bounded connectivity/auth/model smoke check, not a benchmark.",
            "TTFT is UNKNOWN unless a streaming measurement is performed elsewhere.",
            "Pricing and total compute cost remain UNKNOWN without direct evidence.",
            "Partner metadata never influences checks, scores, or diagnostic status.",
        ],
    }


def _endpoint_category(
    provider: ProviderPreset,
    *,
    allow_non_local: bool,
    allow_public: bool,
) -> str:
    safety = classify_endpoint(provider.base_url)
    parts = urlsplit(provider.base_url)
    if safety.category == "invalid":
        raise ProviderError("invalid provider endpoint URL")
    if parts.username or parts.password:
        raise ProviderError("provider endpoint URL credentials are not allowed")
    if safety.category == "private" and not allow_non_local:
        raise ProviderError("LAN/private provider endpoint requires --allow-non-local")
    if safety.category == "public" and not allow_public:
        raise ProviderError("public provider endpoint requires --allow-public")
    return safety.category


def _model_ids(response: OpenAICompatibleResponse) -> Optional[List[str]]:
    data = response.json_data
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        return None
    return [
        item["id"]
        for item in data["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _evaluate_models(
    result: Dict[str, Any],
    response: OpenAICompatibleResponse,
    model: Optional[str],
) -> Tuple[str, bool]:
    status = response.status
    result["request_sent"] = True
    result["models_probe"]["http_status"] = status
    _add_check(
        result,
        "connectivity",
        "PASS",
        "The provider endpoint returned HTTP {0}.".format(status),
        http_status=status,
    )

    if status in (401, 403):
        _add_check(result, "authentication", "FAIL", "The provider rejected the configured API key.", http_status=status)
        _add_check(result, "model_availability", "UNKNOWN", "Authentication failed before model availability could be checked.")
        return "FAIL", False

    if status in UNSUPPORTED_MODELS_STATUSES:
        _add_check(result, "authentication", "UNKNOWN", "Authentication could not be inferred because /models is unsupported.", http_status=status)
        _add_check(result, "model_availability", "UNKNOWN", "Unsupported /models is not evidence that the model is unavailable.", http_status=status)
        return "UNKNOWN", True

    if not 200 <= status < 300:
        _add_check(result, "authentication", "UNKNOWN", "Authentication could not be inferred from HTTP {0}.".format(status), http_status=status)
        _add_check(result, "model_availability", "UNKNOWN", "The models response did not establish availability.", http_status=status)
        return ("FAIL" if status >= 500 else "UNKNOWN"), False

    ids = _model_ids(response)
    if ids is None:
        _add_check(result, "authentication", "UNKNOWN", "HTTP success without a valid models shape does not prove authentication.", http_status=status)
        _add_check(result, "model_availability", "UNKNOWN", "The response did not contain an OpenAI-compatible data list.")
        return "UNKNOWN", True

    listed = model in ids if model else None
    result["models_probe"].update(
        {"status": "PASS", "model_count": len(ids), "selected_model_listed": listed}
    )
    _add_check(result, "authentication", "PASS", "The authenticated request returned an OpenAI-compatible model list.", http_status=status)
    if listed is True:
        _add_check(
            result,
            "model_catalog",
            "PASS",
            "The selected model was listed in the provider model catalog.",
            model=model,
            model_count=len(ids),
        )
        _add_check(
            result,
            "model_availability",
            "UNKNOWN",
            "Catalog listing does not prove that this API key can invoke the selected model.",
            model=model,
            model_count=len(ids),
        )
        return "UNKNOWN", True
    if model:
        _add_check(result, "model_availability", "UNKNOWN", "The selected model was not listed; invocation was not attempted.", model=model, model_count=len(ids))
        return "UNKNOWN", True
    _add_check(result, "model_availability", "PASS", "The provider returned {0} model(s).".format(len(ids)), model_count=len(ids))
    return "PASS", True


def _run_chat(
    result: Dict[str, Any],
    provider: ProviderPreset,
    *,
    api_key: str,
    model: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    try:
        response = create_chat_completion(
            provider.base_url,
            payload=payload,
            timeout=timeout,
            api_key=api_key,
        )
    except OpenAICompatibleTransportError as exc:
        result["request_sent"] = True
        result["chat_smoke"]["status"] = "FAIL"
        _add_check(result, "chat_smoke", "FAIL", "The optional chat request failed in transport.", error=str(exc)[:240])
        return "FAIL"

    status = response.status
    result["request_sent"] = True
    result["chat_smoke"]["http_status"] = status
    result["metrics"]["total_latency_ms"] = response.elapsed_ms
    if not 200 <= status < 300:
        result["chat_smoke"]["status"] = "FAIL"

        if status == 401:
            _add_check(
                result,
                "authentication",
                "FAIL",
                "The optional chat request rejected the configured API key.",
                http_status=status,
            )

        elif status == 403:
            _add_check(
                result,
                "model_access",
                "FAIL",
                "The API key authenticated, but the provider denied permission to invoke the selected model.",
                http_status=status,
                model=model,
            )

        elif status == 402:
            _add_check(
                result,
                "billing_or_credit",
                "FAIL",
                "The provider rejected the request because billing or credit is required.",
                http_status=status,
            )

        elif status == 429:
            _add_check(
                result,
                "quota_or_rate_limit",
                "FAIL",
                "The provider rejected the request because a quota or rate limit was reached.",
                http_status=status,
            )

        else:
            _add_check(
                result,
                "request_rejection",
                "UNKNOWN",
                "The provider rejected the optional chat request for an unclassified reason.",
                http_status=status,
            )

        _add_check(
            result,
            "chat_smoke",
            "FAIL",
            "The optional chat request returned HTTP {0}.".format(
                status
            ),
            http_status=status,
        )

        return "FAIL"
    if not response.json_valid or not extract_chat_text(response.json_data).strip():
        result["chat_smoke"]["status"] = "FAIL"
        _add_check(result, "chat_smoke", "FAIL", "The optional chat response contained no usable content.", http_status=status)
        return "FAIL"

    result["chat_smoke"].update({"status": "PASS", "selected_model_invoked": True})
    _add_check(result, "authentication", "PASS", "The optional authenticated chat request was accepted.", http_status=status)
    _add_check(
        result,
        "model_access",
        "PASS",
        "The selected model was successfully invoked with this API key.",
        model=model,
    )
    _add_check(result, "model_availability", "PASS", "The selected model produced usable chat content.", model=model)
    _add_check(result, "chat_smoke", "PASS", "Usable content was returned and not retained.", http_status=status, model=model, total_latency_ms=response.elapsed_ms)
    return "PASS"


def run_provider_check(
    provider: ProviderPreset,
    *,
    api_key: Optional[str],
    timeout: float = 10.0,
    allow_non_local: bool = False,
    allow_public: bool = False,
    smoke: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    category = _endpoint_category(
        provider,
        allow_non_local=allow_non_local,
        allow_public=allow_public,
    )
    selected_model = model or provider.default_model
    result = _new_result(
        provider,
        key_present=bool(api_key),
        smoke=smoke,
        model=selected_model,
        endpoint_category=category,
    )
    _add_check(result, "endpoint_safety", "PASS", "The endpoint category was permitted for this check.", category=category)

    if not api_key:
        _add_check(result, "credential_configuration", "FAIL", "The provider API key environment variable is not set.", api_key_env=provider.api_key_env)
        _add_check(result, "model_availability", "UNKNOWN", "No request was sent, so model availability remains unknown.")
        result["status"] = "FAIL"
        return result

    _add_check(result, "credential_configuration", "PASS", "The API key environment variable is set; its value is not recorded.", api_key_env=provider.api_key_env)
    try:
        response = list_models(provider.base_url, timeout=timeout, api_key=api_key)
    except OpenAICompatibleTransportError as exc:
        result["request_sent"] = True
        _add_check(result, "connectivity", "FAIL", "The provider models request failed in transport.", error=str(exc)[:240])
        _add_check(result, "model_availability", "UNKNOWN", "Model availability was not established.")
        result["status"] = "FAIL"
        return result

    status, can_smoke = _evaluate_models(result, response, selected_model)
    if smoke and selected_model and can_smoke and status != "FAIL":
        status = _run_chat(result, provider, api_key=api_key, model=selected_model, timeout=timeout)
    elif smoke:
        result["chat_smoke"]["status"] = "SKIP"
        summary = "No model was configured." if not selected_model else "The models check failed."
        _add_check(result, "chat_smoke", "SKIP", summary + " The optional chat request was skipped.")
    result["status"] = status
    return result


def render_provider_list(providers: Sequence[ProviderPreset]) -> str:
    lines = ["InferDoctor Provider Presets", "============================="]
    for provider in providers:
        lines.extend(
            [
                "{0} ({1})".format(provider.id, provider.display_name),
                "  protocol: {0}".format(provider.protocol),
                "  base_url: {0}".format(redact_endpoint(provider.base_url)),
                "  default_model: {0}".format(provider.default_model or "none"),
            ]
        )
    return "\n".join(lines)


def render_provider_show(provider: ProviderPreset) -> str:
    data = provider.to_dict()
    fields = (
        "id", "display_name", "protocol", "base_url", "docs_url", "signup_url",
        "partner_url", "api_key_env", "default_model", "capabilities",
    )
    lines = ["InferDoctor Provider Preset", "==========================="]
    for field in fields:
        value = data[field]
        if field == "base_url":
            value = redact_endpoint(value)
        elif field == "capabilities":
            value = ", ".join(value)
        lines.append("{0}: {1}".format(field, value if value is not None else "none"))
    return "\n".join(lines)


def render_provider_check(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    provider = result.get("provider") or {}
    metrics = result.get("metrics") or {}
    lines = [
        "InferDoctor Provider Check",
        "==========================",
        "Provider: {0} ({1})".format(provider.get("display_name", "unknown"), provider.get("id", "unknown")),
        "Status: {0}".format(result.get("status") or "UNKNOWN"),
        "Endpoint: {0}".format(result.get("endpoint") or "unknown"),
        "Model: {0}".format(result.get("model") or "not configured"),
        "API key env: {0} ({1})".format(result.get("api_key_env") or "none", "present" if result.get("api_key_present") else "missing"),
        "",
        "Checks:",
    ]
    for item in result.get("checks") or []:
        lines.append("- [{0}] {1}: {2}".format(item.get("status", "UNKNOWN"), item.get("name", "check"), item.get("summary", "")))
    ttft = metrics.get("ttft_ms")
    latency = metrics.get("total_latency_ms")
    lines.extend(
        [
            "",
            "Metrics:",
            "- TTFT: {0}".format("UNKNOWN" if ttft is None else "{0} ms".format(ttft)),
            "- Total latency: {0}".format("UNKNOWN" if latency is None else "{0} ms".format(latency)),
            "- Pricing: {0}".format((result.get("pricing") or {}).get("status") or "UNKNOWN"),
            "- Response content retained: no",
        ]
    )
    return "\n".join(lines)
