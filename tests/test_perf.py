
from __future__ import annotations

import json

import pytest
from urllib.error import HTTPError, URLError

from inferdoctor.core import perf
from inferdoctor.core.perf import PerfResult, perf_result_to_dict, render_perf_json, render_perf_markdown, render_perf_result, run_endpoint_smoke, run_streaming_smoke


class FakeResponse:
    def __init__(self, status=200, body="", lines=None, content_type="application/json"):
        self.status = status
        self.body = body.encode("utf-8")
        self.lines = [line.encode("utf-8") if isinstance(line, str) else line for line in (lines or [])]
        self.headers = {"Content-Type": content_type}
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self.status

    def read(self, _size=-1):
        return self.body

    def readline(self):
        if self.index >= len(self.lines):
            return b""
        line = self.lines[self.index]
        self.index += 1
        if isinstance(line, BaseException):
            raise line
        return line


def test_perf_endpoint_success_mock(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "local-model"
        assert payload["stream"] is False
        return FakeResponse(body=json.dumps({"choices": [{"message": {"content": "Smoke test ok."}}]}))

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_endpoint_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)
    rendered = render_perf_result(result)

    assert result.reachable is True
    assert result.openai_compatible == "yes"
    assert result.total_latency_seconds is not None
    assert result.output_tokens_estimate is not None
    assert "Performance UX Smoke Test" in rendered
    assert len(calls) == 2


def test_perf_endpoint_timeout_mock(monkeypatch):
    def fake_urlopen(request, timeout):
        raise URLError("timed out")

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_endpoint_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=1)

    assert result.reachable is False
    assert result.openai_compatible == "no"
    assert result.user_experience == "Endpoint/configuration failure"
    assert "time" in result.checks[0].summary.lower()


def test_perf_streaming_first_chunk_mock(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["stream"] is True
        return FakeResponse(
            lines=[
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "confirmed"
    assert result.ttft_seconds is not None
    assert result.total_latency_seconds is not None
    assert result.output_tokens_estimate is not None


def test_perf_streaming_unsupported_mock(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(body=json.dumps({"choices": [{"message": {"content": "not streamed"}}]}))

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "accepted_full_response"
    assert result.successful_runs == 1



def test_streaming_ignores_role_only_and_empty_chunks(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                ": keepalive\n",
                "\n",
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"Visible\"}}]}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "confirmed"
    assert result.ttft_seconds is not None
    assert result.runs[0].content_chunks == 1


def test_streaming_no_content_is_not_success(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "no_content"
    assert result.failed_runs == 1
    assert result.ttft_seconds is None


def test_streaming_uses_usage_tokens_for_exact_tps(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n",
                "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"completion_tokens\":3}}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.metric_quality["tokens"] == "exact"
    assert result.tps_quality == "exact"
    assert result.output_tokens_exact == 3


def test_http_200_error_object_fails_without_body_leak(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(body=json.dumps({"error": {"message": "bad api key secret-value-123456"}}))

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_endpoint_smoke("http://user:pass@127.0.0.1:8000/v1?api_key=secret", "local-model", timeout=3)

    assert result.failed_runs == 1
    assert "user:pass" not in result.endpoint
    assert "secret" not in result.endpoint



def test_streaming_handles_content_array_delta(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                "data: {\"choices\":[{\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"Array text\"}]}}]}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "confirmed"
    assert result.runs[0].content_chunks == 1


def test_streaming_malformed_json_is_ignored_when_content_arrives(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                "data: {not-json}\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"Visible\"}}]}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "confirmed"
    assert "malformed" in result.warnings[0]


def test_streaming_http_200_error_object_fails(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                "data: {\"error\":{\"message\":\"model not found\"}}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "failed"
    assert result.failed_runs == 1
    assert "model not found" in result.errors[0]


def test_streaming_timeout_before_first_content(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(lines=[TimeoutError("timed out")], content_type="text/event-stream")

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "failed"
    assert result.ttft_seconds is None
    assert result.failed_runs == 1


def test_streaming_timeout_after_partial_content(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=[
                "data: {\"choices\":[{\"delta\":{\"content\":\"Partial\"}}]}\n",
                TimeoutError("connection reset"),
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "confirmed"
    assert result.runs[0].ttft_seconds is not None
    assert result.failed_runs == 1


def test_streaming_truncated_stream_warns(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(
            lines=["data: {\"choices\":[{\"delta\":{\"content\":\"Partial\"}}]}\n"],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)

    assert result.streaming_supported == "confirmed"
    assert "without a [DONE]" in result.warnings[0]


def test_models_auth_failure_is_sanitized(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(status=401, body="{\"error\":\"secret token\"}")
        raise AssertionError("chat should not be called")

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://user:pass@127.0.0.1:8000/v1?api_key=secret", "local-model", timeout=3)

    assert result.openai_compatible == "no"
    assert result.endpoint == "http://127.0.0.1:8000/v1?api_key=REDACTED"
    assert "secret token" not in "\n".join(result.checks[0].details)



def test_perf_runs_and_warmup_are_bounded():
    with pytest.raises(ValueError):
        run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", runs=4)
    with pytest.raises(ValueError):
        run_endpoint_smoke("http://127.0.0.1:8000/v1", "local-model", warmup=2)


def test_streaming_repeatability_summary_uses_measured_runs(monkeypatch):
    chat_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal chat_calls
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        chat_calls += 1
        return FakeResponse(
            lines=[
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n",
                "data: [DONE]\n",
            ],
            content_type="text/event-stream",
        )

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_streaming_smoke("http://127.0.0.1:8000/v1", "local-model", runs=3, warmup=1)

    assert chat_calls == 4
    assert len(result.runs) == 4
    assert result.successful_runs == 3
    assert result.failed_runs == 0
    assert result.aggregate_metrics["ttft_median"] is not None



def test_perf_json_report_has_stable_redacted_fields(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(body=json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 2}}))

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_endpoint_smoke("http://user:pass@127.0.0.1:8000/v1?api_key=secret&debug=true", "local-model", timeout=3)
    report = perf_result_to_dict(result)
    rendered = render_perf_json(result)

    assert report["schema_version"] == "inferdoctor.perf.v1"
    assert report["endpoint"] == "http://127.0.0.1:8000/v1?api_key=REDACTED&debug=true"
    assert "user:pass" not in rendered
    assert "secret" not in rendered
    assert "raw" not in report
    assert "metrics" in report
    assert "metric_quality" in report


def test_perf_markdown_report_is_issue_friendly(monkeypatch):
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "local-model"}]}))
        return FakeResponse(body=json.dumps({"choices": [{"message": {"content": "ok"}}]}))

    monkeypatch.setattr(perf, "urlopen", fake_urlopen)

    result = run_endpoint_smoke("http://127.0.0.1:8000/v1", "local-model", timeout=3)
    rendered = render_perf_markdown(result)

    assert "# InferDoctor Performance UX Smoke Test" in rendered
    assert "not a benchmark" in rendered
    assert "## Suggestions" in rendered



def test_experience_readiness_thresholds_are_transparent():
    responsive = perf._evaluate_experience(PerfResult(
        mode="streaming",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        streaming_supported="confirmed",
        ttft_seconds=1.0,
        total_latency_seconds=6.0,
        successful_runs=1,
    ))
    assert responsive.category == "Responsive for interactive use"
    assert responsive.confidence == "low"
    assert "single measured run" in responsive.explanation

    high_ttft = perf._evaluate_experience(PerfResult(
        mode="streaming",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        streaming_supported="confirmed",
        ttft_seconds=3.5,
        rough_tokens_per_second=60,
        successful_runs=2,
    ))
    assert high_ttft.category == "Likely frustrating without progress feedback"
    assert "Fast TPS does not compensate" in high_ttft.explanation

    failure = perf._evaluate_experience(PerfResult(
        mode="endpoint",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=False,
        openai_compatible="no",
        failed_runs=1,
    ))
    assert failure.category == "Endpoint/configuration failure"
    assert failure.confidence == "high"


def test_perf_json_schema_uses_stable_english_fields():
    result = PerfResult(
        mode="streaming",
        endpoint="http://user:secret@127.0.0.1:8000/v1?api_key=hidden",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        streaming_supported="confirmed",
        successful_runs=1,
        failed_runs=0,
        user_experience="Responsive for interactive use",
    )

    document = json.loads(render_perf_json(result))

    assert document["schema_version"] == "inferdoctor.perf.v1"
    assert document["test_type"] == "streaming"
    assert document["streaming_observed"] == "confirmed"
    assert "secret" not in document["endpoint"]
    assert "hidden" not in document["endpoint"]
    assert "api_key=REDACTED" in document["endpoint"]
    markdown = render_perf_markdown(result)
    console = render_perf_result(result)
    assert "secret" not in markdown
    assert "hidden" not in markdown
    assert "secret" not in console
    assert "hidden" not in console
