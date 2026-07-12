from __future__ import annotations

import json
from unittest.mock import patch

from inferdoctor.cli import main
from inferdoctor.core.experience import get_profile, profile_names, render_profile
from inferdoctor.core.perf import PerfResult


def test_experience_profiles_are_available():
    names = profile_names()

    assert "customer-service" in names
    assert "rag" in names
    rendered = render_profile(get_profile("customer-service"))
    assert "Customer Service Assistant" in rendered
    assert "TTFT" in rendered
    assert "heuristics" in rendered.lower()


def test_experience_profile_cli(capsys):
    exit_code = main(["experience", "profile", "rag"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "RAG Application" in output
    assert "Relevant InferDoctor commands" in output


def test_optimize_endpoint_accepts_profile(capsys):
    exit_code = main(["optimize", "endpoint", "--runtime", "vllm", "--profile", "customer-service"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Experience profile is Customer Service Assistant" in output
    assert "Profile customer-service" in output


def test_optimize_plan_accepts_profile(capsys):
    exit_code = main(["optimize", "plan", "--profile", "restaurant-ordering", "--streaming"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Restaurant Ordering Assistant" in output
    assert "Application profile" in output


def test_perf_endpoint_profile_adds_context_without_real_network(capsys):
    result = PerfResult(
        mode="endpoint",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        total_latency_seconds=1.2,
        successful_runs=1,
        failed_runs=0,
    )
    with patch("inferdoctor.cli.run_endpoint_smoke", return_value=result):
        exit_code = main([
            "perf",
            "endpoint",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--model",
            "local-model",
            "--profile",
            "customer-service",
            "--format",
            "json",
        ])

    assert exit_code == 0
    data = json.loads(capsys.readouterr().out)
    assert any("Profile customer-service" in item for item in data["suggestions"])
    assert any("Experience profile customer-service" in item for item in data["warnings"])
