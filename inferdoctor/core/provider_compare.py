from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from inferdoctor import __version__
from inferdoctor.core.openai_compatible import (
    OpenAICompatibleTransportError,
    create_chat_completion,
    extract_chat_text,
    list_models,
)
from inferdoctor.core.providers import (
    OPENAI_COMPATIBLE_PROTOCOL,
    ProviderTarget,
    analyze_provider_models_response,
    classify_provider_invocation_failure,
    validate_provider_endpoint,
)
from inferdoctor.core.rag import (
    evaluate_deterministic_answer,
    sha256_text,
    unavailable_deterministic_answer,
    validate_case_object,
)


PROVIDER_COMPARE_SCHEMA_VERSION = "inferdoctor.provider.compare.v1"
OBSERVABLE_LAYER_ORDER = (
    "endpoint",
    "connectivity",
    "authentication",
    "billing_or_credit",
    "quota_or_rate_limit",
    "model_access",
    "generation",
    "verification",
)


class ProviderCompareError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(status: str, summary: str, **evidence: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": status,
        "summary": summary,
    }
    if evidence:
        result["evidence"] = evidence
    return result


def _workload(case: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": str(case.get("question") or ""),
        }
    ]
    request_options = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }
    encoded = json.dumps(
        request_options,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    required = case.get("required_facts") or []
    forbidden = case.get("forbidden_claims") or []
    return {
        "case_id": case.get("case_id"),
        "case_schema_version": case.get("schema_version"),
        "workload_sha256": sha256_text(encoded),
        "prompt_sha256": sha256_text(str(case.get("question") or "")),
        "prompt_retained": False,
        "message_count": len(messages),
        "required_facts_total": len(required) if isinstance(required, list) else 0,
        "forbidden_claims_total": len(forbidden) if isinstance(forbidden, list) else 0,
        "request_options": {
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
        },
        "_messages": messages,
    }


def _target_metadata(
    target: ProviderTarget,
    *,
    endpoint_category: str,
    api_key_present: bool,
) -> Dict[str, Any]:
    return target.to_public_dict(
        endpoint_category=endpoint_category,
        api_key_present=api_key_present,
    )


def _new_target_result(
    target: ProviderTarget,
    case: Dict[str, Any],
    *,
    endpoint_category: str,
    api_key_present: bool,
    workload_sha256: str,
) -> Dict[str, Any]:
    return {
        "target": _target_metadata(
            target,
            endpoint_category=endpoint_category,
            api_key_present=api_key_present,
        ),
        "status": "UNKNOWN",
        "workload_sha256": workload_sha256,
        "request_attempted": False,
        "request_sent": False,
        "checks": {
            "endpoint": _check(
                "PASS",
                "The endpoint category was explicitly permitted.",
                category=endpoint_category,
            ),
            "connectivity": _check(
                "UNKNOWN",
                "No HTTP response has been observed yet.",
            ),
            "authentication": _check(
                "UNKNOWN" if target.api_key_env else "SKIP",
                (
                    "Authentication has not been established."
                    if target.api_key_env
                    else "No API key is configured for this target."
                ),
            ),
            "model_catalog": _check(
                "UNKNOWN",
                "Model catalog evidence has not been collected.",
            ),
            "billing_or_credit": _check(
                "UNKNOWN",
                "No billing or credit evidence was observed.",
            ),
            "quota_or_rate_limit": _check(
                "UNKNOWN",
                "No quota or rate-limit evidence was observed.",
            ),
            "model_access": _check(
                "UNKNOWN",
                "Model invocation access has not been established.",
            ),
            "generation": _check(
                "UNKNOWN",
                "No usable generated response has been observed.",
            ),
            "verification": _check(
                "UNKNOWN",
                "Deterministic answer verification has not run.",
            ),
        },
        "models_probe": {
            "request_attempted": False,
            "request_sent": False,
            "status": "UNKNOWN",
            "http_status": None,
            "model_count": None,
            "selected_model_listed": None,
        },
        "invocation": {
            "request_attempted": False,
            "request_sent": False,
            "status": "UNKNOWN",
            "http_status": None,
            "usable_response": False,
            "response_sha256": None,
            "response_length_chars": None,
            "response_content_retained": False,
            "response_evaluated_in_memory": False,
        },
        "metrics": {
            "ttft_ms": None,
            "observed_total_latency_ms": None,
            "sample_count": 0,
        },
        "quality": unavailable_deterministic_answer(
            case,
            evidence_state="response_unavailable",
        ),
        "first_broken_layer": None,
        "first_broken_layer_status": "not_established",
        "pricing": {
            "status": "UNKNOWN",
            "input_cost_per_million_tokens": None,
            "output_cost_per_million_tokens": None,
            "currency": None,
        },
        "cost": {
            "status": "UNKNOWN",
            "api_cost": None,
            "total_compute_cost": None,
            "currency": None,
        },
        "limitations": [
            "This is one bounded request in one comparison run, not a benchmark.",
            "TTFT is UNKNOWN because the request is non-streaming.",
            "Pricing, API cost, and total compute cost remain UNKNOWN.",
            "Catalog presence does not prove invocation access.",
            "Raw response content is evaluated in memory and is not retained.",
        ],
    }


def _safe_transport_summary(exc: OpenAICompatibleTransportError) -> str:
    text = str(exc).lower()
    if "invalid whitespace" in text or "control characters" in text:
        return "The configured API key contains invalid whitespace or control characters."
    if "api key must be a string" in text:
        return "The configured API key is not a string."
    if "exceeded 1 mib" in text:
        return "The OpenAI-compatible response exceeded the bounded response limit."
    return "The OpenAI-compatible request failed before a usable HTTP response was recorded."


def _is_local_credential_error(exc: OpenAICompatibleTransportError) -> bool:
    text = str(exc).lower()
    return (
        "invalid whitespace" in text
        or "control characters" in text
        or "api key must be a string" in text
    )


def _probe_models(
    result: Dict[str, Any],
    target: ProviderTarget,
    *,
    api_key: Optional[str],
    timeout: float,
) -> None:
    probe = result["models_probe"]
    result["request_attempted"] = True
    probe["request_attempted"] = True
    try:
        response = list_models(
            target.base_url,
            timeout=timeout,
            api_key=api_key,
        )
    except OpenAICompatibleTransportError as exc:
        if result["request_sent"] is False:
            result["request_sent"] = None
        probe["request_sent"] = None
        summary = _safe_transport_summary(exc)
        if _is_local_credential_error(exc):
            probe["status"] = "FAIL"
            result["checks"]["authentication"] = _check(
                "FAIL",
                summary,
            )
        result["checks"]["model_catalog"] = _check(
            "UNKNOWN",
            summary,
        )
        return

    result["request_sent"] = True
    probe["request_sent"] = True
    analysis = analyze_provider_models_response(response, target.model)
    probe["http_status"] = response.status
    result["checks"]["connectivity"] = _check(
        "PASS",
        "The models route returned an HTTP response.",
        http_status=response.status,
    )

    if analysis["kind"] == "authentication_failure":
        result["checks"]["authentication"] = _check(
            "FAIL",
            "The models request rejected the configured API key.",
            http_status=response.status,
        )
        result["checks"]["model_catalog"] = _check(
            "UNKNOWN",
            "Authentication failed before catalog evidence was available.",
        )
        return
    if analysis["kind"] == "unsupported":
        result["checks"]["model_catalog"] = _check(
            "UNKNOWN",
            "The models route is unsupported; this is not evidence that the model is unavailable.",
            http_status=response.status,
        )
        return
    if analysis["kind"] == "http_error":
        result["checks"]["model_catalog"] = _check(
            "UNKNOWN",
            "The models response did not establish catalog evidence.",
            http_status=response.status,
        )
        return
    if analysis["kind"] == "invalid_catalog":
        result["checks"]["model_catalog"] = _check(
            "UNKNOWN",
            "HTTP success did not contain an OpenAI-compatible model list.",
            http_status=response.status,
        )
        return

    listed = analysis["selected_model_listed"]
    model_count = int(analysis["model_count"] or 0)
    probe.update(
        {
            "status": "PASS",
            "model_count": model_count,
            "selected_model_listed": listed,
        }
    )
    result["checks"]["model_catalog"] = _check(
        "PASS" if listed else "UNKNOWN",
        (
            "The selected model was present in the returned catalog."
            if listed
            else (
                "The selected model was not present in the returned catalog; "
                "this does not prove invocation is unavailable."
            )
        ),
        model_count=model_count,
        selected_model_listed=listed,
    )
    if analysis["authentication_status"] == "PASS" and api_key:
        result["checks"]["authentication"] = _check(
            "PASS",
            "The authenticated models request returned a compatible catalog.",
            http_status=response.status,
        )


def _set_invocation_failure(
    result: Dict[str, Any],
    *,
    layer: str,
    summary: str,
    http_status: Optional[int] = None,
) -> None:
    result["invocation"]["status"] = "FAIL"
    result["checks"][layer] = _check(
        "FAIL",
        summary,
        **({"http_status": http_status} if http_status is not None else {}),
    )


def _invoke_workload(
    result: Dict[str, Any],
    target: ProviderTarget,
    case: Dict[str, Any],
    *,
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    timeout: float,
) -> None:
    payload = {
        "model": target.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }
    invocation = result["invocation"]
    invocation["request_attempted"] = True
    result["request_attempted"] = True
    try:
        response = create_chat_completion(
            target.base_url,
            payload=payload,
            timeout=timeout,
            api_key=api_key,
        )
    except OpenAICompatibleTransportError as exc:
        if result["request_sent"] is False:
            result["request_sent"] = None
        invocation["request_sent"] = None
        layer = (
            "authentication"
            if _is_local_credential_error(exc)
            else "connectivity"
        )
        result["checks"][layer] = _check(
            "FAIL",
            _safe_transport_summary(exc),
        )
        invocation["status"] = "FAIL"
        return

    invocation["request_sent"] = True
    result["request_sent"] = True
    invocation["http_status"] = response.status
    result["metrics"]["observed_total_latency_ms"] = response.elapsed_ms
    result["metrics"]["sample_count"] = 1
    result["checks"]["connectivity"] = _check(
        "PASS",
        "The same-workload chat request returned an HTTP response.",
        http_status=response.status,
    )

    failure = classify_provider_invocation_failure(response.status)
    if failure is not None:
        _set_invocation_failure(
            result,
            layer=failure["compare_layer"],
            summary=failure["summary"],
            http_status=response.status,
        )
        return

    if api_key:
        result["checks"]["authentication"] = _check(
            "PASS",
            "The authenticated same-workload request was accepted.",
            http_status=response.status,
        )
    else:
        result["checks"]["authentication"] = _check(
            "SKIP",
            "The endpoint accepted the request without a configured API key.",
            http_status=response.status,
        )
    result["checks"]["model_access"] = _check(
        "PASS",
        "The selected model accepted the same-workload invocation.",
        http_status=response.status,
    )

    if not response.json_valid:
        _set_invocation_failure(
            result,
            layer="generation",
            summary="The invocation succeeded at HTTP level but returned invalid JSON.",
            http_status=response.status,
        )
        return

    answer = extract_chat_text(response.json_data)
    if not answer.strip():
        _set_invocation_failure(
            result,
            layer="generation",
            summary="The invocation returned no usable answer content.",
            http_status=response.status,
        )
        return

    invocation.update(
        {
            "status": "PASS",
            "usable_response": True,
            "response_sha256": sha256_text(answer),
            "response_length_chars": len(answer),
            "response_evaluated_in_memory": True,
        }
    )
    result["checks"]["generation"] = _check(
        "PASS",
        "A usable answer was returned and evaluated in memory without retaining its content.",
        http_status=response.status,
        response_length_chars=len(answer),
    )
    evaluation = evaluate_deterministic_answer(answer, case)
    normalized_status = evaluation.get("status")
    if normalized_status == "INCONCLUSIVE":
        normalized_status = "UNKNOWN"
    result["quality"] = {
        **evaluation,
        "status": normalized_status,
    }
    if normalized_status == "PASS":
        result["checks"]["verification"] = _check(
            "PASS",
            "All available deterministic answer checks passed.",
        )
    elif normalized_status == "FAIL":
        result["checks"]["verification"] = _check(
            "FAIL",
            evaluation.get("diagnostic_interpretation")
            or "At least one deterministic answer check failed.",
        )
    else:
        result["checks"]["verification"] = _check(
            "UNKNOWN",
            evaluation.get("diagnostic_interpretation")
            or "Deterministic answer verification was inconclusive.",
        )


def first_failed_observable_layer(checks: Mapping[str, Dict[str, Any]]) -> Optional[str]:
    for layer in OBSERVABLE_LAYER_ORDER:
        item = checks.get(layer)
        if isinstance(item, dict) and item.get("status") == "FAIL":
            return layer
    return None


def _finalize_target(result: Dict[str, Any]) -> None:
    first_broken = first_failed_observable_layer(result["checks"])
    result["first_broken_layer"] = first_broken
    result["first_broken_layer_status"] = (
        "established" if first_broken else "not_established"
    )
    if first_broken:
        result["status"] = "FAIL"
    elif result["checks"]["generation"]["status"] == "PASS" and result[
        "checks"
    ]["verification"]["status"] == "PASS":
        result["status"] = "PASS"
    else:
        result["status"] = "UNKNOWN"


def _run_target(
    target: ProviderTarget,
    case: Dict[str, Any],
    *,
    workload: Dict[str, Any],
    endpoint_category: str,
    api_key: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    result = _new_target_result(
        target,
        case,
        endpoint_category=endpoint_category,
        api_key_present=bool(api_key),
        workload_sha256=workload["workload_sha256"],
    )
    if target.api_key_required and not api_key:
        result["checks"]["authentication"] = _check(
            "FAIL",
            "The required API key environment variable is not set.",
            api_key_env=target.api_key_env,
        )
        _finalize_target(result)
        return result

    _probe_models(
        result,
        target,
        api_key=api_key,
        timeout=timeout,
    )
    _invoke_workload(
        result,
        target,
        case,
        messages=workload["_messages"],
        api_key=api_key,
        timeout=timeout,
    )
    _finalize_target(result)
    return result


def _comparison_observations(targets: Sequence[Dict[str, Any]]) -> List[str]:
    observations: List[str] = []
    for item in targets:
        metadata = item.get("target") or {}
        label = metadata.get("display_name") or metadata.get("id") or "Target"
        quality = item.get("quality") or {}
        status = quality.get("status") or "UNKNOWN"
        if status == "PASS":
            observations.append(
                "{0} satisfied all available deterministic case requirements in this run.".format(
                    label
                )
            )
        elif status == "FAIL":
            observations.append(
                "{0} failed at least one deterministic case requirement in this run.".format(
                    label
                )
            )
        else:
            observations.append(
                "{0} deterministic quality remains UNKNOWN from the available evidence.".format(
                    label
                )
            )
        if item.get("first_broken_layer"):
            observations.append(
                "{0} first failed observable layer: {1}.".format(
                    label,
                    item["first_broken_layer"],
                )
            )
    latency_values = [
        (
            (item.get("target") or {}).get("display_name")
            or (item.get("target") or {}).get("id"),
            (item.get("metrics") or {}).get("observed_total_latency_ms"),
        )
        for item in targets
    ]
    if latency_values and all(value is not None for _, value in latency_values):
        observations.append(
            "Observed total latency in this bounded run: {0}. This single observation is not a benchmark.".format(
                ", ".join(
                    "{0}={1} ms".format(label, value)
                    for label, value in latency_values
                )
            )
        )
    return observations


def _differences(targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fields = (
        ("deterministic_quality", lambda item: (item.get("quality") or {}).get("status")),
        ("observed_total_latency_ms", lambda item: (item.get("metrics") or {}).get("observed_total_latency_ms")),
        ("first_broken_layer", lambda item: item.get("first_broken_layer")),
    )
    differences: List[Dict[str, Any]] = []
    for field, getter in fields:
        values = {
            str((item.get("target") or {}).get("id")): getter(item)
            for item in targets
        }
        differences.append(
            {
                "field": field,
                "values": values,
                "evidence_only": True,
            }
        )
    return differences


def run_provider_compare(
    targets: Sequence[ProviderTarget],
    case: Dict[str, Any],
    *,
    api_keys: Optional[Mapping[str, Optional[str]]] = None,
    timeout: float = 30.0,
    allow_non_local: bool = False,
    allow_public: bool = False,
) -> Dict[str, Any]:
    if len(targets) < 2:
        raise ProviderCompareError("provider comparison requires at least two targets")
    ids = [target.id for target in targets]
    if len(set(ids)) != len(ids):
        raise ProviderCompareError("provider comparison target ids must be unique")
    if any(target.protocol != OPENAI_COMPATIBLE_PROTOCOL for target in targets):
        raise ProviderCompareError("provider comparison currently supports OpenAI-compatible targets only")
    if any(not target.model for target in targets):
        raise ProviderCompareError("every provider comparison target requires a model")
    findings = validate_case_object(case)
    failures = [item for item in findings if item.get("status") == "FAIL"]
    if failures:
        raise ProviderCompareError(
            "invalid RAG Case: {0}".format(failures[0].get("message") or "validation failed")
        )

    endpoint_categories = {
        target.id: validate_provider_endpoint(
            target.base_url,
            allow_non_local=allow_non_local,
            allow_public=allow_public,
        )
        for target in targets
    }
    workload = _workload(case)
    key_values = api_keys or {}
    target_results = [
        _run_target(
            target,
            case,
            workload=workload,
            endpoint_category=endpoint_categories[target.id],
            api_key=key_values.get(target.id),
            timeout=timeout,
        )
        for target in targets
    ]
    statuses = [item["status"] for item in target_results]
    if "FAIL" in statuses:
        status = "FAIL"
    elif "UNKNOWN" in statuses:
        status = "UNKNOWN"
    else:
        status = "PASS"
    public_workload = {
        key: value for key, value in workload.items() if not key.startswith("_")
    }
    return {
        "schema_version": PROVIDER_COMPARE_SCHEMA_VERSION,
        "timestamp": _now(),
        "inferdoctor_version": __version__,
        "status": status,
        "workload": public_workload,
        "target_order": ids,
        "targets": target_results,
        "differences": _differences(target_results),
        "observations": _comparison_observations(target_results),
        "recommendations": [],
        "limitations": [
            "Provider Compare executes one bounded same-workload request per target by default; it is not a benchmark.",
            "Deterministic term checks are not an LLM-as-a-judge evaluation.",
            "No provider ranking, composite score, routing, or Doctor Recommendation is produced.",
            "TTFT, pricing, API cost, and total compute cost remain UNKNOWN.",
            "Partner metadata is informational and cannot influence comparison evidence.",
        ],
    }


def _fmt_status(value: Any) -> str:
    return "UNKNOWN" if value is None else str(value)


def render_provider_compare(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)

    targets = result.get("targets") or []
    labels = [
        str((item.get("target") or {}).get("display_name") or "Target")
        for item in targets
    ]
    width = 24
    lines = [
        "InferDoctor Provider Compare",
        "=" * 76,
        "Case: {0}".format((result.get("workload") or {}).get("case_id") or "unknown"),
        "Status: {0}".format(result.get("status") or "UNKNOWN"),
        "Same workload: {0}".format((result.get("workload") or {}).get("workload_sha256") or "unknown"),
        "",
        "{0:<24}{1}".format(
            "Evidence",
            "".join("{0:<{1}}".format(label[: width - 1], width) for label in labels),
        ),
        "-" * (24 + width * len(labels)),
    ]

    def values(getter: Any) -> str:
        return "".join(
            "{0:<{1}}".format(_fmt_status(getter(item))[: width - 1], width)
            for item in targets
        )

    for label, getter in (
        ("Endpoint", lambda item: (item.get("target") or {}).get("endpoint")),
        ("Endpoint category", lambda item: (item.get("target") or {}).get("endpoint_category")),
        ("Model", lambda item: (item.get("target") or {}).get("model")),
        ("Connectivity", lambda item: (item.get("checks") or {}).get("connectivity", {}).get("status")),
        ("Authentication", lambda item: (item.get("checks") or {}).get("authentication", {}).get("status")),
        ("Model catalog", lambda item: (item.get("checks") or {}).get("model_catalog", {}).get("status")),
        ("Model access", lambda item: (item.get("checks") or {}).get("model_access", {}).get("status")),
        ("Generation", lambda item: (item.get("checks") or {}).get("generation", {}).get("status")),
        ("Required facts", lambda item: ("UNKNOWN" if not (item.get("quality") or {}).get("required_fact_checks", {}).get("evaluable") else "{0}/{1}".format((item.get("quality") or {}).get("required_facts_matched", 0), (item.get("quality") or {}).get("required_facts_total", 0)))),
        ("Forbidden claims", lambda item: ("UNKNOWN" if not (item.get("quality") or {}).get("forbidden_claim_checks", {}).get("evaluable") else (item.get("quality") or {}).get("forbidden_claims_matched"))),
        ("Quality", lambda item: (item.get("quality") or {}).get("status")),
        ("Observed latency", lambda item: ("UNKNOWN" if (item.get("metrics") or {}).get("observed_total_latency_ms") is None else "{0} ms".format((item.get("metrics") or {}).get("observed_total_latency_ms")))),
        ("TTFT", lambda _item: "UNKNOWN"),
        ("First broken layer", lambda item: item.get("first_broken_layer") or "none established"),
    ):
        lines.append("{0:<24}{1}".format(label, values(getter)))

    lines.extend(["", "Doctor's evidence:"])
    lines.extend("- {0}".format(item) for item in result.get("observations") or [])
    lines.extend(
        [
            "",
            "Note: Same-workload deterministic evidence from one bounded run; no winner, score, pricing, or benchmark claim is produced.",
        ]
    )
    return "\n".join(lines)
