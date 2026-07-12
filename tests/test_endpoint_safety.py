from __future__ import annotations

from unittest.mock import patch

from inferdoctor.cli import main
from inferdoctor.core.endpoint_safety import classify_endpoint, redact_endpoint
from inferdoctor.core.perf import PerfResult


def test_redact_endpoint_removes_credentials_and_sensitive_query():
    redacted = redact_endpoint("http://user:secret@192.168.1.20:8000/v1?api_key=hidden&foo=bar")

    assert redacted == "http://192.168.1.20:8000/v1?api_key=REDACTED&foo=bar"
    assert "secret" not in redacted
    assert "hidden" not in redacted


def test_classify_local_private_public_and_invalid_endpoints():
    assert classify_endpoint("http://127.0.0.1:8000/v1").category == "localhost"
    assert classify_endpoint("http://localhost:11434/v1").category == "localhost"
    private = classify_endpoint("http://192.168.1.20:8000/v1")
    assert private.category == "private"
    assert private.requires_explicit_allow is True
    public = classify_endpoint("https://example.com/v1")
    assert public.category == "public"
    assert public.requires_explicit_allow is True
    assert classify_endpoint("not-a-url").category == "invalid"


def test_perf_endpoint_rejects_non_local_endpoint_without_explicit_allow(capsys):
    exit_code = main([
        "perf",
        "endpoint",
        "--endpoint",
        "http://192.168.1.20:8000/v1",
        "--model",
        "local-model",
    ])

    assert exit_code == 2
    assert "--allow-non-local" in capsys.readouterr().err


def test_perf_endpoint_allows_non_local_when_explicitly_allowed(capsys):
    result = PerfResult(
        mode="endpoint",
        endpoint="http://192.168.1.20:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        total_latency_seconds=1.0,
        successful_runs=1,
        failed_runs=0,
    )
    with patch("inferdoctor.cli.run_endpoint_smoke", return_value=result):
        exit_code = main([
            "perf",
            "endpoint",
            "--endpoint",
            "http://192.168.1.20:8000/v1",
            "--model",
            "local-model",
            "--allow-non-local",
        ])

    assert exit_code == 0
    assert "Performance UX Smoke Test" in capsys.readouterr().out
