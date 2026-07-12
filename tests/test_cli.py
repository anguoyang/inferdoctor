from unittest.mock import patch

import pytest

from inferdoctor.cli import _results_for_target, main
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.i18n import TRANSLATIONS, t
from inferdoctor.core.perf import PerfResult


def _sample_result():
    return CheckResult(
        name="system",
        status=Status.PASS,
        summary="System information collected",
    )


def _sample_run():
    return [_sample_result()], Config()


def _sample_run_for_language(target=None, config_path=None, timeout=None, endpoint=None, language=None):
    config = Config()
    if language is not None:
        config.language = language
    return [_sample_result()], config


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_default_command_renders_health_dashboard(results, capsys):
    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "InferDoctor - Local AI Stack Health Check" in output
    assert "Overall Health: 100 / 100" in output
    results.assert_called_once_with(None, None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target", side_effect=_sample_run_for_language)
def test_global_language_flag_uses_check_command(results, capsys):
    exit_code = main(["--language", "zh"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "InferDoctor - 本地 AI 堆栈健康检查" in output
    assert "整体健康度" in output
    results.assert_called_once_with(None, None, None, None, "zh")


@patch("inferdoctor.cli._results_for_target", side_effect=_sample_run_for_language)
def test_global_language_flag_supports_japanese(results, capsys):
    exit_code = main(["--language", "ja"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "InferDoctor - ローカルAIスタックヘルスチェック" in output
    assert "全体の健全性" in output
    results.assert_called_once_with(None, None, None, None, "ja")


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_default_command_does_not_force_language_override(results, capsys):
    exit_code = main([])

    assert exit_code == 0
    assert "InferDoctor - Local AI Stack Health Check" in capsys.readouterr().out
    results.assert_called_once_with(None, None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_check_command_renders_dashboard(results, capsys):
    exit_code = main(["check", "system"])

    assert exit_code == 0
    assert "System      PASS" in capsys.readouterr().out
    results.assert_called_once_with("system", None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target", side_effect=_sample_run_for_language)
def test_check_command_accepts_language(results, capsys):
    exit_code = main(["check", "--language", "zh"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "InferDoctor - 本地 AI 堆栈健康检查" in output
    results.assert_called_once_with(None, None, None, None, "zh")


def test_report_command_rejects_language_option(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["report", "--language", "zh", "--format", "markdown"])

    assert exc.value.code == 2
    assert "unrecognized arguments: --language zh" in capsys.readouterr().err


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_report_command_writes_json(results, tmp_path):
    output = tmp_path / "report.json"

    exit_code = main(["report", "--format", "json", "--output", str(output)])

    assert exit_code == 0
    assert '"name": "system"' in output.read_text(encoding="utf-8")
    results.assert_called_once_with(None, None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_check_timeout_verbose_and_endpoint_options(results, capsys):
    endpoint = "http://127.0.0.1:30000/v1"
    exit_code = main(
        ["check", "sglang", "--timeout", "4.5", "--endpoint", endpoint, "--verbose"]
    )

    assert exit_code == 0
    assert "Detailed diagnostics:" in capsys.readouterr().out
    results.assert_called_once_with("sglang", None, 4.5, endpoint, "auto")


def test_endpoint_requires_http_checker():
    with pytest.raises(SystemExit, match="does not use an HTTP endpoint"):
        _results_for_target("system", None, endpoint="http://127.0.0.1:1")


def test_timeout_must_be_positive(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["check", "--timeout", "0"])

    assert exc.value.code == 2
    assert "must be greater than zero" in capsys.readouterr().err


def test_invalid_language_flag_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--language", "fr"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_invalid_config_language_is_rejected_by_cli(tmp_path):
    config_path = tmp_path / "inferdoctor.yaml"
    config_path.write_text("language: fr\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["check", "--config", str(config_path)])

    assert "configuration error" in str(exc.value)
    assert "language" in str(exc.value)


def _nested_keys(d, prefix=""):
    """Yield all dot-path keys from a nested dict."""
    for k, v in d.items():
        path = "{0}.{1}".format(prefix, k) if prefix else k
        if isinstance(v, dict):
            yield from _nested_keys(v, path)
        else:
            yield path


def test_translation_dictionaries_have_identical_key_sets():
    en_keys = set(_nested_keys(TRANSLATIONS["en"]))
    zh_keys = set(_nested_keys(TRANSLATIONS["zh"]))
    ja_keys = set(_nested_keys(TRANSLATIONS["ja"]))
    assert zh_keys == en_keys, "Missing in zh: {0}".format(en_keys - zh_keys)
    assert ja_keys == en_keys, "Missing in ja: {0}".format(en_keys - ja_keys)


def test_translation_falls_back_to_english_for_unknown_language():
    assert t("dashboard.title", "fr") == "InferDoctor - Local AI Stack Health Check"


def test_endpoint_override_rejects_invalid_url():
    with pytest.raises(SystemExit, match="http:// or https://"):
        _results_for_target("sglang", None, endpoint="127.0.0.1:30000/v1")


@patch("inferdoctor.cli._results_for_target")
def test_explain_command_does_not_run_checks(results, capsys):
    exit_code = main(["explain", "cuda-toolkit-missing"])

    assert exit_code == 0
    assert "InferDoctor Explain:" in capsys.readouterr().out
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_capacity_command_does_not_run_checks(results, capsys):
    exit_code = main(["capacity", "--vram", "24"])

    assert exit_code == 0
    assert "InferDoctor Capacity Preview" in capsys.readouterr().out
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_scenario_command_renders_readiness(results, capsys):
    exit_code = main(["scenario"])

    assert exit_code == 0
    assert "InferDoctor Scenario Readiness" in capsys.readouterr().out
    results.assert_called_once_with(None, None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_scenarios_alias_renders_readiness(results, capsys):
    exit_code = main(["scenarios", "cpu-only-fallback"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "CPU-only fallback:" in output
    assert "Local chatbot:" not in output


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_profile_command_renders_markdown(results, capsys):
    exit_code = main(["profile", "--format", "markdown"])

    assert exit_code == 0
    assert "InferDoctor Safe Diagnostic Profile" in capsys.readouterr().out
    results.assert_called_once_with(None, None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target", return_value=_sample_run())
def test_profile_command_writes_json(results, tmp_path):
    output = tmp_path / "profile.json"

    exit_code = main(["profile", "--format", "json", "--output", str(output)])

    assert exit_code == 0
    assert '"safe_to_share": true' in output.read_text(encoding="utf-8")
    results.assert_called_once_with(None, None, None, None, "auto")


@patch("inferdoctor.cli._results_for_target")
def test_capacity_command_accepts_request_options(results, capsys):
    exit_code = main(["capacity", "--gpu", "RTX 3090", "--model-size", "14b", "--quant", "q4", "--runtime", "ollama"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Requested estimate:" in output
    assert "RTX 3090" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_template_list_does_not_run_checks(results, capsys):
    exit_code = main(["template", "list"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Local AI App Templates" in output
    assert "customer-service" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_template_show_does_not_run_checks(results, capsys):
    exit_code = main(["template", "show", "restaurant-ordering"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Restaurant Ordering Assistant" in output
    assert "Required:" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_template_create_writes_starter_project(results, tmp_path, capsys):
    output_dir = tmp_path / "customer-service-demo"

    exit_code = main(["template", "create", "customer-service", "--output", str(output_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Starter Project Created" in output
    assert (output_dir / "README.md").exists()
    assert (output_dir / "app.py").exists()
    assert (output_dir / "data" / "faq.md").exists()
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_template_smoke_test_runs_safe_commands(results, tmp_path, capsys):
    output_dir = tmp_path / "customer-service-demo"
    main(["template", "create", "customer-service", "--output", str(output_dir)])
    capsys.readouterr()

    exit_code = main(["template", "smoke-test", str(output_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Template Smoke Test" in output
    assert "python app.py --dry-run" in output
    assert "Overall status: PASS" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_init_command_recommends_customer_service(results, capsys):
    exit_code = main(["init", "--goal", "customer-service", "--preference", "easiest"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Guided Setup" in output
    assert "Use the customer-service template." in output
    assert "Start with Ollama." in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_init_command_recommends_document_qa_gpu(results, capsys):
    exit_code = main(["init", "--goal", "document-qa", "--preference", "gpu"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Use the local-doc-qa template." in output
    assert "GPU-capable" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_recommend_command_uses_vram_override(results, capsys):
    exit_code = main(["recommend", "--goal", "customer-service", "--vram", "24"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Stack Recommendation" in output
    assert "Hardware: provided VRAM override" in output
    assert "inferdoctor capacity --vram 24" in output
    assert "Template: customer-service" in output
    assert "7B or 14B" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_recommend_command_supports_easiest_document_qa(results, capsys):
    exit_code = main(["recommend", "--goal", "document-qa", "--preference", "easiest", "--vram", "8"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Template: local-doc-qa" in output
    assert "Runtime: Ollama" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_model_fit_command_uses_vram_override(results, capsys):
    exit_code = main(["model", "fit", "--size", "14b", "--quant", "q4", "--vram", "24"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Model Fit Advisor" in output
    assert "Model size: 14B" in output
    assert "Fit:" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_template_validate_does_not_run_checks(results, tmp_path, capsys):
    output_dir = tmp_path / "customer-service-demo"
    main(["template", "create", "customer-service", "--output", str(output_dir)])
    capsys.readouterr()

    exit_code = main(["template", "validate", str(output_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Template Validation" in output
    assert "Overall status: PASS" in output
    assert "python app.py" in output
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_stack_plan_command_does_not_run_checks(results, capsys):
    exit_code = main(["stack", "plan", "--goal", "customer-service", "--vram", "24"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Local AI Stack Plan" in output
    assert "Starter template: customer-service" in output
    assert "inferdoctor template validate ./customer-service-demo" in output
    results.assert_not_called()


def test_help_includes_beginner_workflow(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "inferdoctor recommend --goal customer-service" in capsys.readouterr().out


def test_template_help_mentions_validate(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["template", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "template validate" in output


@patch("inferdoctor.cli._results_for_target")
def test_stack_bootstrap_requires_dry_run(results, capsys):
    exit_code = main(["stack", "bootstrap", "--goal", "customer-service"])

    assert exit_code == 2
    assert "requires --dry-run" in capsys.readouterr().err
    results.assert_not_called()


@patch("inferdoctor.cli._results_for_target")
def test_stack_bootstrap_dry_run_outputs_plan(results, capsys):
    exit_code = main(["stack", "bootstrap", "--goal", "customer-service", "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "InferDoctor Stack Bootstrap Plan (Dry Run)" in output
    assert "inferdoctor template create customer-service" in output
    assert "will not do automatically" in output
    results.assert_not_called()



def test_template_compose_command(tmp_path, capsys):
    output = tmp_path / "compose"

    exit_code = main(["template", "compose", "customer-service", "--output", str(output)])

    assert exit_code == 0
    assert "Docker Compose Files Created" in capsys.readouterr().out
    assert (output / "docker-compose.yml").exists()
    assert (output / ".env.example").exists()



def test_stack_bootstrap_output_generates_files(tmp_path, capsys):
    output = tmp_path / "bootstrap"

    exit_code = main(["stack", "bootstrap", "--goal", "customer-service", "--output", str(output)])

    assert exit_code == 0
    assert "Stack Bootstrap Files Created" in capsys.readouterr().out
    assert (output / "README.md").exists()
    assert (output / "bootstrap_plan.md").exists()



def test_template_registry_command(capsys):
    exit_code = main(["template", "registry"])

    assert exit_code == 0
    assert "InferDoctor Template Registry" in capsys.readouterr().out


@patch("inferdoctor.cli.run_endpoint_smoke")
def test_perf_endpoint_command_uses_smoke_runner(smoke, capsys):
    smoke.return_value = PerfResult(
        mode="endpoint",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        user_experience="Good for interactive demo",
    )

    exit_code = main([
        "perf",
        "endpoint",
        "--endpoint",
        "http://127.0.0.1:8000/v1",
        "--model",
        "local-model",
        "--timeout",
        "3",
    ])

    assert exit_code == 0
    assert "Performance UX Smoke Test" in capsys.readouterr().out
    smoke.assert_called_once_with("http://127.0.0.1:8000/v1", "local-model", 3.0, runs=1, warmup=0)


@patch("inferdoctor.cli.run_streaming_smoke")
def test_perf_streaming_command_uses_smoke_runner(smoke, capsys):
    smoke.return_value = PerfResult(
        mode="streaming",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        streaming_supported="confirmed",
        ttft_seconds=0.8,
        user_experience="Good for interactive demo",
    )

    exit_code = main([
        "perf",
        "streaming",
        "--endpoint",
        "http://127.0.0.1:8000/v1",
        "--model",
        "local-model",
    ])

    assert exit_code == 0
    assert "Streaming observed: confirmed" in capsys.readouterr().out
    smoke.assert_called_once_with("http://127.0.0.1:8000/v1", "local-model", 30.0, runs=1, warmup=0)


def test_optimize_endpoint_command_renders_advice(capsys):
    exit_code = main([
        "optimize",
        "endpoint",
        "--runtime",
        "vllm",
        "--vram",
        "24",
        "--model-size",
        "14b",
        "--quant",
        "q4",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Endpoint UX Optimization Advice" in output
    assert "inferdoctor perf streaming" in output


def test_optimize_rag_command_renders_advice(capsys):
    exit_code = main([
        "optimize",
        "rag",
        "--top-k",
        "8",
        "--ttft",
        "2.5",
        "--streaming",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "RAG UX Optimization Advice" in output
    assert "top_k" in output


def test_perf_rejects_excessive_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["perf", "streaming", "--endpoint", "http://127.0.0.1:8000/v1", "--model", "local-model", "--runs", "4"])

    assert exc.value.code == 2
    assert "--runs must be between 1 and 3" in capsys.readouterr().err


def test_perf_rejects_excessive_warmup(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["perf", "endpoint", "--endpoint", "http://127.0.0.1:8000/v1", "--model", "local-model", "--warmup", "2"])

    assert exc.value.code == 2
    assert "--warmup must be between 0 and 1" in capsys.readouterr().err


@patch("inferdoctor.cli.run_endpoint_smoke")
def test_perf_endpoint_writes_json_report(smoke, tmp_path, capsys):
    smoke.return_value = PerfResult(
        mode="endpoint",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        successful_runs=1,
    )
    output = tmp_path / "perf.json"

    exit_code = main([
        "perf",
        "endpoint",
        "--endpoint",
        "http://127.0.0.1:8000/v1",
        "--model",
        "local-model",
        "--format",
        "json",
        "--output",
        str(output),
    ])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert "\"schema_version\": \"inferdoctor.perf.v1\"" in output.read_text(encoding="utf-8")


@patch("inferdoctor.cli.run_streaming_smoke")
def test_perf_streaming_writes_markdown_report(smoke, tmp_path):
    smoke.return_value = PerfResult(
        mode="streaming",
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        reachable=True,
        openai_compatible="yes",
        streaming_supported="confirmed",
        successful_runs=1,
    )
    output = tmp_path / "perf.md"

    exit_code = main([
        "perf",
        "streaming",
        "--endpoint",
        "http://127.0.0.1:8000/v1",
        "--model",
        "local-model",
        "--format",
        "markdown",
        "--output",
        str(output),
    ])

    assert exit_code == 0
    assert "# InferDoctor Performance UX Smoke Test" in output.read_text(encoding="utf-8")
