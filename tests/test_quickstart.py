from __future__ import annotations

from inferdoctor.cli import main
from inferdoctor.core.quickstart import build_quickstart_plan, render_quickstart_plan


def test_build_quickstart_customer_service_plan():
    plan = build_quickstart_plan(goal="customer-service", preference="easiest", hardware="gpu")

    assert plan.goal == "customer-service"
    assert plan.template == "customer-service"
    assert any("template create customer-service" in command for command in plan.validation_commands)
    assert any("perf baseline create" in command for command in plan.performance_commands)


def test_build_quickstart_rag_with_endpoint():
    plan = build_quickstart_plan(
        goal="rag",
        preference="performance",
        endpoint="http://192.168.1.20:8000/v1",
        location="lan",
        hardware="gpu",
        runtime="vllm",
    )

    rendered = render_quickstart_plan(plan)
    assert plan.goal == "rag"
    assert plan.template == "local-doc-qa"
    assert plan.recommended_runtime == "vllm"
    assert "http://192.168.1.20:8000/v1" in rendered
    assert "optimize plan" in rendered


def test_quickstart_cli_customer_service(capsys):
    exit_code = main(["quickstart", "customer-service", "--preference", "easiest"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "InferDoctor Quickstart" in output
    assert "Template: customer-service" in output
    assert "inferdoctor template smoke-test" in output
    assert "perf baseline create" in output


def test_quickstart_cli_rag_with_endpoint(capsys):
    exit_code = main([
        "quickstart",
        "rag",
        "--preference",
        "performance",
        "--endpoint",
        "http://192.168.1.20:8000/v1",
        "--hardware",
        "gpu",
        "--runtime",
        "vllm",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Goal: rag" in output
    assert "Template: local-doc-qa" in output
    assert "Configured endpoint hint: http://192.168.1.20:8000/v1" in output
