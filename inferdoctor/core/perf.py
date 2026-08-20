from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from inferdoctor import __version__
from inferdoctor.core.http import HTTPCheckError, describe_http_error
from inferdoctor.core.openai_compatible import (
    chat_completions_url as _chat_url,
    models_url as _models_url,
)

SMOKE_PROMPT = "Reply with one short sentence: local AI endpoint smoke test."
MAX_BODY_BYTES = 1024 * 1024
MAX_SSE_LINES = 2048
MAX_SSE_BYTES = 1024 * 1024
MAX_RUNS = 3
MAX_WARMUP = 1
REPORT_SCHEMA_VERSION = "inferdoctor.perf.v1"
SENSITIVE_QUERY_MARKERS = ("key", "token", "secret", "password", "auth", "credential")


@dataclass(frozen=True)
class PerfCheck:
    name: str
    status: str
    summary: str
    details: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PerfRun:
    index: int
    warmup: bool
    success: bool
    streaming_requested: bool
    streaming_observed: str
    status_code: Optional[int] = None
    ttft_seconds: Optional[float] = None
    total_latency_seconds: Optional[float] = None
    generation_duration_seconds: Optional[float] = None
    output_tokens: Optional[int] = None
    output_token_source: str = "unknown"
    tokens_per_second: Optional[float] = None
    tps_quality: str = "unknown"
    content_chunks: int = 0
    error: Optional[str] = None
    warning: Optional[str] = None


@dataclass(frozen=True)
class ExperienceReadiness:
    category: str
    explanation: str
    confidence: str
    actions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PerfResult:
    mode: str
    endpoint: str
    model: Optional[str]
    reachable: bool
    openai_compatible: str
    streaming_supported: str = "unknown"
    ttft_seconds: Optional[float] = None
    total_latency_seconds: Optional[float] = None
    rough_tokens_per_second: Optional[float] = None
    output_tokens_estimate: Optional[int] = None
    user_experience: str = "Inconclusive"
    checks: List[PerfCheck] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    generation_duration_seconds: Optional[float] = None
    output_tokens_exact: Optional[int] = None
    output_token_source: str = "unknown"
    tps_quality: str = "unknown"
    successful_runs: int = 0
    failed_runs: int = 0
    runs: List[PerfRun] = field(default_factory=list)
    aggregate_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    metric_quality: Dict[str, str] = field(default_factory=dict)
    experience_explanation: str = ""
    confidence: str = "low"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = REPORT_SCHEMA_VERSION


def sanitize_endpoint(url: str) -> str:
    parts = urlsplit(url)
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


def _headers() -> Dict[str, str]:
    return {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "InferDoctor/{0}".format(__version__),
    }


def _sanitize_error(message: str) -> str:
    text = " ".join(str(message).split())
    return text[:240] if text else "Unknown error."


def _read_body(response: Any) -> str:
    return response.read(MAX_BODY_BYTES).decode("utf-8", errors="replace")


def _http_error_check(url: str, status: int) -> PerfCheck:
    safe_url = sanitize_endpoint(url)
    if status in (401, 403):
        return PerfCheck("authentication", "WARN", "Endpoint is reachable but requires authentication.", ["HTTP {0} from {1}".format(status, safe_url)])
    if status == 404:
        return PerfCheck("route", "WARN", "Route returned HTTP 404. The base URL may be missing or duplicating /v1.", ["Probe URL: {0}".format(safe_url)])
    return PerfCheck("http", "WARN", "Endpoint returned HTTP {0}.".format(status), ["Probe URL: {0}".format(safe_url), "Response body was not recorded to avoid leaking secrets."])


def _get_json(url: str, timeout: float) -> tuple[int, Optional[Any], str]:
    request = Request(url, headers=_headers(), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.getcode())
            body = _read_body(response)
    except HTTPError as exc:
        return int(exc.code), None, ""
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPCheckError(_sanitize_error(str(exc))) from exc
    parsed: Optional[Any] = None
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
    return status, parsed, body


def _usage_completion_tokens(data: Any) -> Optional[int]:
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _extract_message_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("error"), dict):
        return ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


def _extract_stream_content(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("error"), dict):
        return ""
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        return _content_to_text(delta.get("content"))
    return _content_to_text(choice.get("text"))


def _rough_token_count(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, round(len(text.split()) * 1.3))


def _token_metric(text: str, data: Optional[Any]) -> tuple[Optional[int], Optional[int], str]:
    exact = _usage_completion_tokens(data)
    if exact is not None:
        return exact, exact, "api_usage"
    estimated = _rough_token_count(text)
    if estimated:
        return None, estimated, "rough_estimate"
    return None, None, "unknown"


def _tps(tokens: Optional[int], duration: Optional[float], quality: str) -> tuple[Optional[float], str]:
    if tokens is None or not duration or duration <= 0:
        return None, "unknown"
    if quality == "api_usage":
        return tokens / duration, "exact"
    if quality == "rough_estimate":
        return tokens / duration, "estimated"
    return None, "unknown"


def _response_content_type(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        return str(headers.get("Content-Type", ""))
    getheader = getattr(response, "getheader", None)
    if callable(getheader):
        return str(getheader("Content-Type", ""))
    return ""


def _error_from_json(data: Any) -> Optional[str]:
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        message = error.get("message") or error.get("type") or "Endpoint returned an error object."
        return _sanitize_error(str(message))
    return None


def _parse_json_body(body: str) -> Optional[Any]:
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _post_chat_once(base_url: str, model: str, timeout: float, stream: bool, index: int, warmup: bool) -> PerfRun:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": stream,
    }
    request = Request(_chat_url(base_url), data=json.dumps(payload).encode("utf-8"), headers=_headers(), method="POST")
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.getcode())
            content_type = _response_content_type(response).lower()
            if status < 200 or status >= 300:
                return PerfRun(index, warmup, False, stream, "failed", status_code=status, total_latency_seconds=time.monotonic() - started, error="HTTP {0}".format(status))
            if stream:
                if "text/event-stream" in content_type:
                    return _parse_sse_response(response, started, index, warmup, status)
                body = _read_body(response)
                data = _parse_json_body(body)
                error = _error_from_json(data)
                total = time.monotonic() - started
                if error:
                    return PerfRun(index, warmup, False, True, "failed", status_code=status, total_latency_seconds=total, error=error)
                text = _extract_message_text(data)
                if text:
                    exact, tokens, source = _token_metric(text, data)
                    return PerfRun(
                        index,
                        warmup,
                        True,
                        True,
                        "accepted_full_response",
                        status_code=status,
                        total_latency_seconds=total,
                        output_tokens=tokens,
                        output_token_source=source,
                        tps_quality="unknown",
                        warning="Endpoint accepted stream=true but returned a full non-streaming JSON response.",
                    )
                return PerfRun(index, warmup, False, True, "inconclusive", status_code=status, total_latency_seconds=total, error="HTTP 200 response did not include streamed content or chat message content.")
            body = _read_body(response)
            data = _parse_json_body(body)
            error = _error_from_json(data)
            total = time.monotonic() - started
            if error:
                return PerfRun(index, warmup, False, False, "not_requested", status_code=status, total_latency_seconds=total, error=error)
            text = _extract_message_text(data)
            exact, tokens, source = _token_metric(text, data)
            return PerfRun(
                index,
                warmup,
                bool(text),
                False,
                "not_requested",
                status_code=status,
                total_latency_seconds=total,
                output_tokens=tokens,
                output_token_source=source,
                error=None if text else "Response did not include chat message content.",
            )
    except HTTPError as exc:
        return PerfRun(index, warmup, False, stream, "failed", status_code=int(exc.code), total_latency_seconds=time.monotonic() - started, error="HTTP {0}".format(exc.code))
    except (URLError, TimeoutError, OSError) as exc:
        return PerfRun(index, warmup, False, stream, "failed", total_latency_seconds=time.monotonic() - started, error=describe_http_error(HTTPCheckError(_sanitize_error(str(exc)))))


def _parse_sse_response(response: Any, started: float, index: int, warmup: bool, status: int) -> PerfRun:
    first_content_at: Optional[float] = None
    text_parts: List[str] = []
    content_chunks = 0
    malformed_events = 0
    bytes_read = 0
    latest_usage_tokens: Optional[int] = None
    saw_done = False
    try:
        for _ in range(MAX_SSE_LINES):
            line = response.readline()
            if not line:
                break
            bytes_read += len(line)
            if bytes_read > MAX_SSE_BYTES:
                return PerfRun(index, warmup, False, True, "failed", status_code=status, total_latency_seconds=time.monotonic() - started, error="SSE response exceeded safe read limit.")
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded or decoded.startswith(":"):
                continue
            if not decoded.startswith("data:"):
                continue
            payload_text = decoded[5:].strip()
            if not payload_text:
                continue
            if payload_text == "[DONE]":
                saw_done = True
                break
            try:
                data = json.loads(payload_text)
            except json.JSONDecodeError:
                malformed_events += 1
                continue
            error = _error_from_json(data)
            if error:
                return PerfRun(index, warmup, False, True, "failed", status_code=status, total_latency_seconds=time.monotonic() - started, error=error)
            usage_tokens = _usage_completion_tokens(data)
            if usage_tokens is not None:
                latest_usage_tokens = usage_tokens
            content = _extract_stream_content(data)
            if not content:
                continue
            if first_content_at is None:
                first_content_at = time.monotonic()
            content_chunks += 1
            text_parts.append(content)
    except (TimeoutError, OSError) as exc:
        total = time.monotonic() - started
        generation = total - (first_content_at - started) if first_content_at is not None else None
        return PerfRun(
            index,
            warmup,
            False,
            True,
            "confirmed" if first_content_at is not None else "failed",
            status_code=status,
            ttft_seconds=(first_content_at - started) if first_content_at is not None else None,
            total_latency_seconds=total,
            generation_duration_seconds=generation,
            content_chunks=content_chunks,
            error="Stream ended with timeout or connection error: {0}".format(_sanitize_error(str(exc))),
        )
    total = time.monotonic() - started
    if first_content_at is None:
        warning = "No non-empty generated content was observed."
        if malformed_events:
            warning += " {0} malformed SSE event(s) were ignored.".format(malformed_events)
        return PerfRun(index, warmup, False, True, "no_content", status_code=status, total_latency_seconds=total, content_chunks=content_chunks, error=warning)
    text = "".join(text_parts)
    exact = latest_usage_tokens
    if exact is not None:
        tokens = exact
        source = "api_usage"
    else:
        tokens = _rough_token_count(text) or None
        source = "rough_estimate" if tokens else "unknown"
    ttft = first_content_at - started
    generation_duration = max(total - ttft, 0.0)
    rate, quality = _tps(tokens, generation_duration, source)
    warning_parts: List[str] = []
    if malformed_events:
        warning_parts.append("{0} malformed SSE event(s) were ignored.".format(malformed_events))
    if not saw_done:
        warning_parts.append("Stream ended without a [DONE] event; the response may have been truncated.")
    warning = " ".join(warning_parts) or None
    return PerfRun(
        index,
        warmup,
        True,
        True,
        "confirmed",
        status_code=status,
        ttft_seconds=ttft,
        total_latency_seconds=total,
        generation_duration_seconds=generation_duration,
        output_tokens=tokens,
        output_token_source=source,
        tokens_per_second=rate,
        tps_quality=quality,
        content_chunks=content_chunks,
        warning=warning,
    )


def _validate_repeat_options(runs: int, warmup: int) -> None:
    if runs < 1 or runs > MAX_RUNS:
        raise ValueError("--runs must be between 1 and {0}".format(MAX_RUNS))
    if warmup < 0 or warmup > MAX_WARMUP:
        raise ValueError("--warmup must be between 0 and {0}".format(MAX_WARMUP))


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _aggregate_numbers(runs: List[PerfRun]) -> Dict[str, Optional[float]]:
    measured = [run for run in runs if not run.warmup and run.success]
    ttfts = [run.ttft_seconds for run in measured if run.ttft_seconds is not None]
    totals = [run.total_latency_seconds for run in measured if run.total_latency_seconds is not None]
    generations = [run.generation_duration_seconds for run in measured if run.generation_duration_seconds is not None]
    rates = [run.tokens_per_second for run in measured if run.tokens_per_second is not None]
    return {
        "ttft_min": min(ttfts) if ttfts else None,
        "ttft_median": _median(ttfts),
        "ttft_max": max(ttfts) if ttfts else None,
        "total_latency_median": _median(totals),
        "generation_duration_median": _median(generations),
        "generation_tps_median": _median(rates),
    }


def _streaming_state(runs: List[PerfRun], mode: str) -> str:
    if mode != "streaming":
        return "not_requested"
    measured = [run for run in runs if not run.warmup]
    states = [run.streaming_observed for run in measured]
    if "confirmed" in states:
        return "confirmed"
    if "accepted_full_response" in states:
        return "accepted_full_response"
    if "no_content" in states:
        return "no_content"
    if "failed" in states:
        return "failed"
    return "inconclusive"


def _metric_quality(runs: List[PerfRun]) -> Dict[str, str]:
    measured = [run for run in runs if not run.warmup and run.success]
    if not measured:
        return {"tokens": "unknown", "tps": "unknown"}
    token_sources = {run.output_token_source for run in measured if run.output_token_source != "unknown"}
    tps_qualities = {run.tps_quality for run in measured if run.tps_quality != "unknown"}
    token_quality = "exact" if token_sources == {"api_usage"} else "estimated" if token_sources else "unknown"
    tps_quality = "exact" if tps_qualities == {"exact"} else "estimated" if tps_qualities else "unknown"
    return {"tokens": token_quality, "tps": tps_quality}


def _evaluate_experience(result: PerfResult) -> ExperienceReadiness:
    if not result.reachable or result.failed_runs and not result.successful_runs:
        return ExperienceReadiness(
            "Endpoint/configuration failure",
            "The smoke test could not complete a successful measured request.",
            "high",
            [
                "Verify the endpoint URL, port, container networking, and runtime process.",
                "Check whether the base URL should end with /v1.",
                "Run inferdoctor check vllm or inferdoctor check sglang for endpoint diagnostics.",
            ],
        )
    ttft = result.ttft_seconds
    total = result.total_latency_seconds
    tps = result.rough_tokens_per_second
    streaming = result.streaming_supported
    confidence = "medium" if result.successful_runs >= 2 and result.failed_runs == 0 else "low"
    single_run_note = " A single measured run is not enough for a stability claim." if result.successful_runs == 1 else ""
    if streaming == "confirmed" and ttft is not None and ttft <= 1.5:
        return ExperienceReadiness(
            "Responsive for interactive use",
            "The first non-empty generated content arrived quickly and streaming was confirmed." + single_run_note,
            confidence,
            ["Keep streaming enabled.", "Use a warmup prompt before demos.", "Keep context size controlled."],
        )
    if streaming == "confirmed" and ttft is not None and ttft <= 3.0:
        return ExperienceReadiness(
            "Usable with streaming",
            "Streaming is working, but TTFT may be noticeable for impatient users." + single_run_note,
            confidence,
            ["Warm up the endpoint.", "Reduce prompt/context size.", "Measure cold and warm runs with --runs 2 --warmup 1."],
        )
    if total is not None and total <= 5.0 and streaming in {"not_requested", "accepted_full_response"}:
        return ExperienceReadiness(
            "Acceptable for an internal prototype",
            "The response completed quickly enough for basic demos, but streaming was not confirmed." + single_run_note,
            confidence,
            ["Validate streaming before customer-facing demos.", "Show progress while waiting.", "Use inferdoctor perf streaming."],
        )
    if ttft is not None and ttft > 3.0:
        return ExperienceReadiness(
            "Likely frustrating without progress feedback",
            "TTFT is high; users may see a blank wait before any answer appears. Fast TPS does not compensate for a long blank wait." + single_run_note,
            confidence,
            ["Show progress immediately.", "Warm up the model.", "Use a smaller model or shorter context."],
        )
    if total is not None and total > 10.0:
        return ExperienceReadiness(
            "Too slow for an interactive demo",
            "Total latency is high for an interactive local AI app. Streaming may improve perception, but the full turn still feels slow." + single_run_note,
            confidence,
            ["Reduce context length.", "Try a smaller or quantized model.", "Check GPU/runtime fit."],
        )
    if tps is not None and tps < 12:
        return ExperienceReadiness(
            "Likely frustrating without progress feedback",
            "Output speed is low, so long answers will feel slow even after generation starts." + single_run_note,
            confidence,
            ["Keep responses short.", "Use streaming.", "Try a smaller model for chat-heavy workflows."],
        )
    return ExperienceReadiness(
        "Inconclusive",
        "The smoke test did not collect enough reliable metrics for a stronger UX read.",
        "low",
        ["Run with --runs 2 or --runs 3.", "Check streaming explicitly.", "Do not treat a single smoke test as a benchmark."],
    )


def _variation_warning(metrics: Dict[str, Optional[float]]) -> Optional[str]:
    minimum = metrics.get("ttft_min")
    maximum = metrics.get("ttft_max")
    if minimum is not None and maximum is not None and minimum > 0 and maximum / minimum >= 2.0:
        return "TTFT varied by 2x or more across measured runs; treat stability as uncertain."
    return None


def _aggregate_result(
    mode: str,
    endpoint: str,
    model: Optional[str],
    reachable: bool,
    openai_compatible: str,
    checks: List[PerfCheck],
    runs: List[PerfRun],
    raw: Dict[str, Any],
) -> PerfResult:
    measured = [run for run in runs if not run.warmup]
    successes = [run for run in measured if run.success]
    failures = [run for run in measured if not run.success]
    metrics = _aggregate_numbers(runs)
    quality = _metric_quality(runs)
    selected = successes[0] if successes else None
    if selected:
        output_exact = selected.output_tokens if selected.output_token_source == "api_usage" else None
        output_estimate = selected.output_tokens if selected.output_token_source == "rough_estimate" else None
    else:
        output_exact = None
        output_estimate = None
    warnings = [run.warning for run in runs if run.warning]
    errors = [run.error for run in runs if run.error]
    variation = _variation_warning(metrics)
    if variation:
        warnings.append(variation)
    result = PerfResult(
        mode=mode,
        endpoint=sanitize_endpoint(endpoint),
        model=model,
        reachable=reachable,
        openai_compatible=openai_compatible,
        streaming_supported=_streaming_state(runs, mode),
        ttft_seconds=metrics.get("ttft_median"),
        total_latency_seconds=metrics.get("total_latency_median"),
        rough_tokens_per_second=metrics.get("generation_tps_median"),
        output_tokens_estimate=output_estimate,
        checks=checks,
        raw_data=raw,
        generation_duration_seconds=metrics.get("generation_duration_median"),
        output_tokens_exact=output_exact,
        output_token_source=(selected.output_token_source if selected else "unknown"),
        tps_quality=quality.get("tps", "unknown"),
        successful_runs=len(successes),
        failed_runs=len(failures),
        runs=runs,
        aggregate_metrics=metrics,
        metric_quality=quality,
        warnings=warnings,
        errors=errors,
    )
    readiness = _evaluate_experience(result)
    return PerfResult(
        **{
            **result.__dict__,
            "user_experience": readiness.category,
            "experience_explanation": readiness.explanation,
            "confidence": readiness.confidence,
            "suggestions": _base_suggestions(result, readiness.actions),
        }
    )


def _base_suggestions(result: PerfResult, readiness_actions: List[str]) -> List[str]:
    suggestions: List[str] = []
    suggestions.extend(readiness_actions[:3])
    if result.openai_compatible != "yes":
        suggestions.append("Check the OpenAI-compatible base URL. Many local runtimes expect the base URL to end with /v1.")
    if result.streaming_supported in {"accepted_full_response", "no_content", "failed"}:
        suggestions.append("Validate stream=true behavior before a customer-facing demo.")
    if result.ttft_seconds is not None and result.ttft_seconds > 2.0:
        suggestions.append("Warm up the endpoint and reduce prompt/context size to improve TTFT.")
    if result.total_latency_seconds is not None and result.total_latency_seconds > 8.0:
        suggestions.append("Use streaming, shorter context, or a smaller/quantized model for interactive UX.")
    if not suggestions:
        suggestions.append("Use this as a smoke signal only. Run real load tests separately before production claims.")
    if result.ttft_seconds is not None or result.total_latency_seconds is not None:
        suggestions.append(
            "Next: inferdoctor optimize endpoint --ttft {0} --latency {1}".format(
                "{0:.2f}".format(result.ttft_seconds) if result.ttft_seconds is not None else "0",
                "{0:.2f}".format(result.total_latency_seconds) if result.total_latency_seconds is not None else "0",
            )
        )
    else:
        suggestions.append("Next: inferdoctor optimize endpoint")
    return list(dict.fromkeys(suggestions))[:5]


def run_endpoint_smoke(endpoint: str, model: Optional[str] = None, timeout: float = 30.0, runs: int = 1, warmup: int = 0) -> PerfResult:
    _validate_repeat_options(runs, warmup)
    endpoint = endpoint.rstrip("/")
    checks: List[PerfCheck] = []
    raw: Dict[str, Any] = {"endpoint": sanitize_endpoint(endpoint), "timeout": timeout, "smoke_test": True, "runs": runs, "warmup": warmup}
    models_url = _models_url(endpoint)
    try:
        status, models_json, _models_body = _get_json(models_url, timeout)
    except HTTPCheckError as exc:
        check = PerfCheck("reachability", "FAIL", describe_http_error(exc), [sanitize_endpoint(models_url)])
        return _aggregate_result("endpoint", endpoint, model, False, "no", [check], [], raw)
    raw["models_status"] = status
    if status < 200 or status >= 300:
        return _aggregate_result("endpoint", endpoint, model, True, "no", [_http_error_check(models_url, status)], [], raw)
    if not isinstance(models_json, dict) or not isinstance(models_json.get("data"), list):
        check = PerfCheck("openai-compatible", "WARN", "/models did not return an OpenAI-compatible data list.", [sanitize_endpoint(models_url)])
        return _aggregate_result("endpoint", endpoint, model, True, "no", [check], [], raw)
    checks.append(PerfCheck("models", "PASS", "/models returned {0} model(s).".format(len(models_json["data"]))))
    if not model:
        checks.append(PerfCheck("chat-completions", "SKIP", "No model was provided, so no chat request was sent."))
        return _aggregate_result("endpoint", endpoint, None, True, "yes", checks, [], raw)
    run_results = []
    for index in range(1, warmup + runs + 1):
        run_results.append(_post_chat_once(endpoint, model, timeout, stream=False, index=index, warmup=index <= warmup))
    passed = sum(1 for run in run_results if not run.warmup and run.success)
    failed = sum(1 for run in run_results if not run.warmup and not run.success)
    checks.append(PerfCheck("chat-completions", "PASS" if passed else "FAIL", "{0} measured run(s) succeeded, {1} failed.".format(passed, failed)))
    return _aggregate_result("endpoint", endpoint, model, True, "yes", checks, run_results, raw)


def run_streaming_smoke(endpoint: str, model: str, timeout: float = 30.0, runs: int = 1, warmup: int = 0) -> PerfResult:
    _validate_repeat_options(runs, warmup)
    endpoint = endpoint.rstrip("/")
    checks: List[PerfCheck] = []
    raw: Dict[str, Any] = {"endpoint": sanitize_endpoint(endpoint), "timeout": timeout, "smoke_test": True, "stream": True, "runs": runs, "warmup": warmup}
    models_url = _models_url(endpoint)
    try:
        status, models_json, _models_body = _get_json(models_url, timeout)
    except HTTPCheckError as exc:
        check = PerfCheck("reachability", "FAIL", describe_http_error(exc), [sanitize_endpoint(models_url)])
        return _aggregate_result("streaming", endpoint, model, False, "no", [check], [], raw)
    raw["models_status"] = status
    if status < 200 or status >= 300:
        return _aggregate_result("streaming", endpoint, model, True, "no", [_http_error_check(models_url, status)], [], raw)
    if not isinstance(models_json, dict) or not isinstance(models_json.get("data"), list):
        return _aggregate_result("streaming", endpoint, model, True, "no", [PerfCheck("openai-compatible", "WARN", "/models did not return an OpenAI-compatible data list.")], [], raw)
    checks.append(PerfCheck("models", "PASS", "/models returned {0} model(s).".format(len(models_json["data"]))))
    run_results = []
    for index in range(1, warmup + runs + 1):
        run_results.append(_post_chat_once(endpoint, model, timeout, stream=True, index=index, warmup=index <= warmup))
    passed = sum(1 for run in run_results if not run.warmup and run.success)
    failed = sum(1 for run in run_results if not run.warmup and not run.success)
    status_label = "PASS" if passed else "FAIL"
    checks.append(PerfCheck("streaming", status_label, "{0} measured run(s) succeeded, {1} failed.".format(passed, failed)))
    return _aggregate_result("streaming", endpoint, model, True, "yes", checks, run_results, raw)


def _fmt_seconds(value: Optional[float]) -> str:
    return "unknown" if value is None else "{0:.2f}s".format(value)


def _fmt_rate(value: Optional[float], quality: str = "unknown") -> str:
    if value is None:
        return "unknown"
    label = "estimated " if quality == "estimated" else ""
    return "~{0:.1f} {1}tokens/sec".format(value, label)


def _fmt_optional(value: Optional[float]) -> str:
    return "unknown" if value is None else "{0:.2f}".format(value)


def render_perf_result(result: PerfResult) -> str:
    lines = [
        "InferDoctor Performance UX Smoke Test",
        "=" * 57,
        "Mode: {0}".format(result.mode),
        "Endpoint: {0}".format(sanitize_endpoint(result.endpoint)),
        "Model: {0}".format(result.model or "not provided"),
        "Reachable: {0}".format("yes" if result.reachable else "no"),
        "OpenAI-compatible check: {0}".format(result.openai_compatible),
        "Streaming observed: {0}".format(result.streaming_supported),
        "Successful measured runs: {0}".format(result.successful_runs),
        "Failed measured runs: {0}".format(result.failed_runs),
        "TTFT median: {0}".format(_fmt_seconds(result.ttft_seconds)),
        "Total latency median: {0}".format(_fmt_seconds(result.total_latency_seconds)),
        "Generation duration median: {0}".format(_fmt_seconds(result.generation_duration_seconds)),
        "Generation speed: {0}".format(_fmt_rate(result.rough_tokens_per_second, result.tps_quality)),
        "Token metric quality: {0}".format(result.metric_quality.get("tokens", result.output_token_source)),
        "User experience read: {0}".format(result.user_experience),
        "Confidence: {0}".format(result.confidence),
    ]
    if result.experience_explanation:
        lines.append("Why: {0}".format(result.experience_explanation))
    if result.aggregate_metrics:
        lines.extend(
            [
                "",
                "Aggregate metrics:",
                "- TTFT min/median/max: {0}s / {1}s / {2}s".format(
                    _fmt_optional(result.aggregate_metrics.get("ttft_min")),
                    _fmt_optional(result.aggregate_metrics.get("ttft_median")),
                    _fmt_optional(result.aggregate_metrics.get("ttft_max")),
                ),
                "- Total latency median: {0}s".format(_fmt_optional(result.aggregate_metrics.get("total_latency_median"))),
                "- Generation TPS median: {0}".format(_fmt_rate(result.aggregate_metrics.get("generation_tps_median"), result.tps_quality)),
            ]
        )
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend("- {0}".format(item) for item in result.warnings)
    if result.errors:
        lines.extend(["", "Errors:"])
        lines.extend("- {0}".format(item) for item in result.errors[:3])
    lines.extend(["", "Checks:"])
    for check in result.checks:
        lines.append("- [{0}] {1}: {2}".format(check.status, check.name, check.summary))
        for detail in check.details:
            lines.append("  - {0}".format(detail))
    if result.runs:
        lines.extend(["", "Measured runs:"])
        for run in [run for run in result.runs if not run.warmup]:
            lines.append(
                "- Run {0}: {1}, streaming={2}, TTFT={3}, total={4}, TPS={5}".format(
                    run.index,
                    "PASS" if run.success else "FAIL",
                    run.streaming_observed,
                    _fmt_seconds(run.ttft_seconds),
                    _fmt_seconds(run.total_latency_seconds),
                    _fmt_rate(run.tokens_per_second, run.tps_quality),
                )
            )
    lines.extend(["", "Top optimization suggestions:"])
    for index, suggestion in enumerate(result.suggestions[:5], start=1):
        lines.append("{0}. {1}".format(index, suggestion))
    lines.extend(["", "Note: This is a timeout-bounded smoke test, not a benchmark. Results vary by prompt, model, runtime, cache state, and hardware."])
    return "\n".join(lines)


def perf_result_to_dict(result: PerfResult) -> Dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "timestamp": result.timestamp,
        "endpoint": sanitize_endpoint(result.endpoint),
        "model": result.model,
        "test_type": result.mode,
        "streaming_requested": result.mode == "streaming",
        "streaming_observed": result.streaming_supported,
        "reachable": result.reachable,
        "openai_compatible": result.openai_compatible,
        "successful_runs": result.successful_runs,
        "failed_runs": result.failed_runs,
        "metrics": {
            "ttft_seconds": result.ttft_seconds,
            "total_latency_seconds": result.total_latency_seconds,
            "generation_duration_seconds": result.generation_duration_seconds,
            "generation_tokens_per_second": result.rough_tokens_per_second,
            "output_tokens_exact": result.output_tokens_exact,
            "output_tokens_estimate": result.output_tokens_estimate,
            "aggregate": result.aggregate_metrics,
        },
        "metric_quality": result.metric_quality,
        "experience_read": {
            "category": result.user_experience,
            "confidence": result.confidence,
            "explanation": result.experience_explanation,
        },
        "suggestions": result.suggestions,
        "warnings": result.warnings,
        "errors": result.errors,
        "runs": [
            {
                "index": run.index,
                "warmup": run.warmup,
                "success": run.success,
                "streaming_requested": run.streaming_requested,
                "streaming_observed": run.streaming_observed,
                "status_code": run.status_code,
                "ttft_seconds": run.ttft_seconds,
                "total_latency_seconds": run.total_latency_seconds,
                "generation_duration_seconds": run.generation_duration_seconds,
                "output_tokens": run.output_tokens,
                "output_token_source": run.output_token_source,
                "tokens_per_second": run.tokens_per_second,
                "tps_quality": run.tps_quality,
                "content_chunks": run.content_chunks,
                "error": run.error,
                "warning": run.warning,
            }
            for run in result.runs
        ],
    }


def render_perf_json(result: PerfResult) -> str:
    return json.dumps(perf_result_to_dict(result), indent=2, sort_keys=True)


def render_perf_markdown(result: PerfResult) -> str:
    lines = [
        "# InferDoctor Performance Smoke Test",
        "",
        "- Endpoint: `{0}`".format(sanitize_endpoint(result.endpoint)),
        "- Model: `{0}`".format(result.model or "not provided"),
        "- Test type: `{0}`".format(result.mode),
        "- Streaming observed: `{0}`".format(result.streaming_supported),
        "- Successful runs: `{0}`".format(result.successful_runs),
        "- Failed runs: `{0}`".format(result.failed_runs),
        "- TTFT median: `{0}`".format(_fmt_seconds(result.ttft_seconds)),
        "- Total latency median: `{0}`".format(_fmt_seconds(result.total_latency_seconds)),
        "- Generation speed: `{0}`".format(_fmt_rate(result.rough_tokens_per_second, result.tps_quality)),
        "- Experience: **{0}**".format(result.user_experience),
        "",
        "## Notes",
        "",
        "This is a timeout-bounded smoke test, not a benchmark. Do not use it as a throughput claim.",
        "",
        "## Suggestions",
        "",
    ]
    lines.extend("{0}. {1}".format(index, item) for index, item in enumerate(result.suggestions[:5], start=1))
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend("- {0}".format(item) for item in result.warnings)
    if result.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend("- {0}".format(item) for item in result.errors[:5])
    return "\n".join(lines)
