from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from inferdoctor import __version__
from inferdoctor.checkers import default_registry
from inferdoctor.core.capacity import render_capacity
from inferdoctor.core.dify import (
    DifyError,
    available_dify_template_names,
    export_dify_template,
    load_dify_config,
    optimize_dify,
    render_dify_check,
    render_dify_knowledge,
    render_dify_optimize,
    render_dify_perf,
    render_dify_smoke,
    render_dify_template_export,
    render_dify_template_list,
    render_dify_template_show,
    render_dify_validation,
    run_dify_check,
    run_dify_knowledge_check,
    run_dify_perf,
    run_dify_smoke,
    validate_dify_kit,
)
from inferdoctor.core.dify_reliability import (
    render_reliability_report,
    run_dify_connectivity_check,
    run_dify_evidence_collect,
    run_dify_evidence_explain,
    run_dify_selfhost_inspect,
    run_dify_selfhost_preflight,
)
from inferdoctor.core.config import (
    Config,
    ConfigError,
    load_config,
    normalize_endpoint,
)
from inferdoctor.core.explain import explain_topics, render_explanation
from inferdoctor.core.endpoint_safety import classify_endpoint, render_endpoint_safety_error
from inferdoctor.core.experience import (
    apply_profile_to_optimization_report,
    apply_profile_to_perf_result,
    get_profile,
    profile_names,
    render_profile as render_experience_profile,
)
from inferdoctor.core.model_fit import estimate_model_fit, render_model_fit
from inferdoctor.core.optimize import advise_endpoint, advise_rag, render_optimization_report
from inferdoctor.core.optimization_plan import build_optimization_plan, render_optimization_plan
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.core.profile import render_profile_json, render_profile_markdown
from inferdoctor.core.perf import render_perf_json, render_perf_markdown, render_perf_result, run_endpoint_smoke, run_streaming_smoke
from inferdoctor.core.perf_baseline import (
    create_baseline_from_report_file,
    delete_baseline,
    list_baselines,
    load_report_or_baseline,
    render_baseline_list,
    render_baseline_markdown,
    render_baseline_summary,
)
from inferdoctor.core.perf_compare import compare_performance_files, render_comparison
from inferdoctor.core.recommendations import recommend_stack, render_recommendation
from inferdoctor.core.rag import (
    RagError,
    compare_rag,
    diagnose_rag,
    init_case_template,
    load_case,
    load_trace,
    render_rag_result,
    run_gold_context_probe,
    validate_case_file,
    validate_trace_file,
)
from inferdoctor.core.rag_dify import (
    capture_dify_knowledge_trace,
)
from inferdoctor.core.quickstart import (
    QUICKSTART_GOALS,
    QUICKSTART_HARDWARE,
    QUICKSTART_LOCATIONS,
    QUICKSTART_PREFERENCES,
    QUICKSTART_RUNTIMES,
    build_quickstart_plan,
    render_quickstart_plan,
)
from inferdoctor.core.runner import run_checks
from inferdoctor.core.scenarios import evaluate_scenarios, render_scenarios, scenario_names
from inferdoctor.core.setup import GOALS, PREFERENCES, RUNTIMES, recommend_setup, render_setup_plan
from inferdoctor.core.stack_plan import (
    build_stack_bootstrap_plan,
    create_stack_bootstrap_project,
    build_stack_plan,
    render_stack_bootstrap_files,
    render_stack_bootstrap_plan,
    render_stack_plan,
)
from inferdoctor.core.template_validation import (
    render_template_smoke_test,
    render_template_validation,
    smoke_test_template_project,
    validate_template_project,
)
from inferdoctor.core.templates import (
    compose_template_names,
    create_compose_project,
    create_template_project,
    render_compose_create_summary,
    render_template_create_summary,
    render_template_detail,
    render_template_list,
    render_template_registry,
    template_names,
)
from inferdoctor.reporters import render_dashboard, render_json, render_markdown


def _model_size(value: str) -> str:
    stripped = value.strip().lower()
    number = stripped[:-1] if stripped.endswith("b") else stripped
    try:
        parsed = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a size like 7b, 14b, or 32b") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return stripped if stripped.endswith("b") else "{0:g}b".format(parsed)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed



def _bounded_int(value: str, minimum: int, maximum: int, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("{0} must be an integer".format(label)) from exc
    if parsed < minimum or parsed > maximum:
        raise argparse.ArgumentTypeError("{0} must be between {1} and {2}".format(label, minimum, maximum))
    return parsed


def _perf_runs(value: str) -> int:
    return _bounded_int(value, 1, 3, "--runs")


def _perf_warmup(value: str) -> int:
    return _bounded_int(value, 0, 1, "--warmup")


def _add_language_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--language",
        choices=("auto", "en", "zh", "ja"),
        default=None,
        help=(
            "Output language for the health dashboard and console summary. "
            "Other commands may remain English in this first i18n release; auto follows the system locale."
        ),
    )


def _add_runtime_options(parser: argparse.ArgumentParser, *, include_language: bool = True) -> None:
    parser.add_argument("--config", help="Path to a JSON or simple YAML config")
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        help="HTTP timeout in seconds; overrides the config value",
    )
    if include_language:
        _add_language_option(parser)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include raw diagnostic data in console or Markdown output",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferdoctor",
        description="Diagnose your local AI stack and get practical next steps for local AI apps.",
        epilog=(
            "Start here: inferdoctor | inferdoctor recommend --goal customer-service | "
            "inferdoctor template create customer-service --output ./customer-service-demo | inferdoctor template smoke-test ./customer-service-demo"
        ),
    )
    parser.add_argument(
        "--language",
        choices=("auto", "en", "zh", "ja"),
        default=None,
        help=(
            "Output language for the health dashboard and console summary. "
            "Other commands may remain English in this first i18n release; auto follows the system locale."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    check = subparsers.add_parser(
        "check",
        help="Check the whole stack or one component",
        description="Check local AI components without installing or running them.",
        epilog=(
            "Examples: inferdoctor check | inferdoctor check sglang "
            "--endpoint http://127.0.0.1:30000/v1"
        ),
    )
    check.add_argument(
        "target",
        nargs="?",
        choices=default_registry().names(),
        help="Component to check; omit to diagnose the full stack",
    )
    check.add_argument(
        "--endpoint",
        help="Override the selected service endpoint for this check",
    )
    _add_runtime_options(check)

    model = subparsers.add_parser(
        "model",
        help="Estimate local model fit",
        description="Lightweight model sizing helpers. No models are downloaded or run.",
    )
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_fit = model_subparsers.add_parser(
        "fit",
        help="Estimate whether a model size likely fits local VRAM",
        description="Estimate memory fit using simple heuristics, not benchmarks.",
        epilog="Examples: inferdoctor model fit --size 14b --quant q4 --vram 24 | inferdoctor model fit --size 32b --quant q4 --runtime vllm",
    )
    model_fit.add_argument(
        "--size",
        type=_model_size,
        default="7b",
        help="Model size such as 7b, 14b, or 32b",
    )
    model_fit.add_argument(
        "--quant",
        choices=("q4", "q8"),
        default="q4",
        help="Quantization heuristic",
    )
    model_fit.add_argument(
        "--vram",
        type=_positive_float,
        help="Override detected VRAM in GiB",
    )
    model_fit.add_argument(
        "--runtime",
        choices=("ollama", "vllm"),
        help="Runtime overhead heuristic",
    )

    optimize = subparsers.add_parser(
        "optimize",
        help="Get practical local AI performance optimization advice",
        description="Advice-only helpers for endpoint and RAG user experience. No inference is run.",
    )
    optimize_subparsers = optimize.add_subparsers(dest="optimize_command", required=True)
    optimize_endpoint = optimize_subparsers.add_parser(
        "endpoint",
        help="Suggest endpoint UX optimizations from supplied metrics",
        description="Analyze supplied runtime, hardware, TTFT, TPS, latency, and streaming hints. This command does not call endpoints.",
        epilog="Examples: inferdoctor optimize endpoint --runtime vllm --vram 24 --model-size 14b --quant q4 | inferdoctor optimize endpoint --runtime ollama --streaming --ttft 1.5 --tps 40",
    )
    optimize_endpoint.add_argument("--runtime", choices=("ollama", "vllm", "sglang", "openai-compatible"), default="openai-compatible")
    optimize_endpoint.add_argument("--vram", type=_positive_float, help="Available VRAM in GiB")
    optimize_endpoint.add_argument("--model-size", type=_model_size, help="Model size class such as 7b, 14b, or 32b")
    optimize_endpoint.add_argument("--quant", choices=("q4", "q8", "fp16"), help="Quantization or precision hint")
    optimize_endpoint.add_argument("--streaming", action="store_true", help="Whether the app already streams tokens to users")
    optimize_endpoint.add_argument("--ttft", type=_positive_float, help="Observed time to first token in seconds")
    optimize_endpoint.add_argument("--tps", type=_positive_float, help="Observed rough output tokens per second")
    optimize_endpoint.add_argument("--latency", type=_positive_float, help="Observed total response latency in seconds")
    optimize_endpoint.add_argument("--context-tokens", type=int, help="Approximate prompt/context tokens")
    optimize_endpoint.add_argument("--ttft-variance", type=_positive_float, help="Observed TTFT max/min ratio across runs")
    optimize_endpoint.add_argument("--containerized", action="store_true", help="Whether the app/runtime is containerized")
    optimize_endpoint.add_argument("--docker", action="store_true", help="Whether Docker is involved in the endpoint path")
    optimize_endpoint.add_argument("--cold-start", action="store_true", help="Whether the first request is noticeably slower")
    optimize_endpoint.add_argument("--cpu-fallback-suspected", action="store_true", help="Whether runtime logs or behavior suggest CPU fallback")
    optimize_endpoint.add_argument("--profile", choices=profile_names(), help="Application experience profile for advice context")
    optimize_rag = optimize_subparsers.add_parser(
        "rag",
        help="Suggest RAG user-experience optimizations",
        description="Advice-only RAG latency helper. It does not run retrieval, embeddings, rerankers, or inference.",
        epilog="Examples: inferdoctor optimize rag --top-k 8 --ttft 2.5 --streaming | inferdoctor optimize rag --retrieval-ms 900 --rerank-ms 1500 --top-k 12",
    )
    optimize_rag.add_argument("--docs", type=int, help="Approximate document count")
    optimize_rag.add_argument("--chunks", type=int, help="Approximate chunk count")
    optimize_rag.add_argument("--top-k", type=int, help="Chunks sent to generation")
    optimize_rag.add_argument("--rerank", action="store_true", help="Whether a reranker is used")
    optimize_rag.add_argument("--retrieval-ms", type=_positive_float, help="Observed retrieval latency in milliseconds")
    optimize_rag.add_argument("--rerank-ms", type=_positive_float, help="Observed rerank latency in milliseconds")
    optimize_rag.add_argument("--embedding-ms", type=_positive_float, help="Observed query embedding/encoding latency in milliseconds")
    optimize_rag.add_argument("--filter-ms", type=_positive_float, help="Observed metadata filtering latency in milliseconds")
    optimize_rag.add_argument("--doc-load-ms", type=_positive_float, help="Observed document loading latency in milliseconds")
    optimize_rag.add_argument("--context-build-ms", type=_positive_float, help="Observed context assembly latency in milliseconds")
    optimize_rag.add_argument("--generation-ms", type=_positive_float, help="Observed post-TTFT generation completion latency in milliseconds")
    optimize_rag.add_argument("--ttft", type=_positive_float, help="Observed time to first token in seconds")
    optimize_rag.add_argument("--streaming", action="store_true", help="Whether the app streams tokens to users")
    optimize_rag.add_argument("--model-size", type=_model_size, help="Model size class such as 7b, 14b, or 32b")
    optimize_rag.add_argument("--vram", type=_positive_float, help="Available VRAM in GiB")
    optimize_rag.add_argument("--profile", choices=profile_names(), help="Application experience profile for RAG UX context")
    optimize_plan = optimize_subparsers.add_parser(
        "plan",
        help="Generate an actionable optimization plan",
        description=(
            "Turn performance reports, comparisons, and supplied runtime facts into prioritized next steps. "
            "This command is advice-only and does not call endpoints."
        ),
        epilog="Examples: inferdoctor optimize plan --report perf.json | inferdoctor optimize plan --baseline before.json --candidate after.json --format markdown",
    )
    optimize_plan.add_argument("--report", help="Performance report or saved baseline JSON to analyze")
    optimize_plan.add_argument("--baseline", help="Baseline JSON path or saved baseline name")
    optimize_plan.add_argument("--candidate", help="Candidate JSON path or saved baseline name")
    optimize_plan.add_argument("--runtime", choices=("ollama", "vllm", "sglang", "openai-compatible"), help="Runtime hint")
    optimize_plan.add_argument("--model-size", type=_model_size, help="Model size class such as 7b, 14b, or 32b")
    optimize_plan.add_argument("--vram", type=_positive_float, help="Available VRAM in GiB")
    optimize_plan.add_argument("--goal", choices=GOALS, help="Application goal")
    optimize_plan.add_argument("--streaming", action="store_true", help="Whether the app is intended to stream output")
    optimize_plan.add_argument("--retrieval-ms", type=_positive_float, help="User-provided retrieval latency in milliseconds")
    optimize_plan.add_argument("--rerank-ms", type=_positive_float, help="User-provided rerank latency in milliseconds")
    optimize_plan.add_argument("--ttft", type=_positive_float, help="Observed TTFT in seconds")
    optimize_plan.add_argument("--profile", choices=profile_names(), help="Application experience profile for optimization priorities")
    optimize_plan.add_argument("--format", choices=("console", "json", "markdown"), default="console", help="Output format")
    optimize_plan.add_argument("--output", help="Write plan output to this file")

    perf = subparsers.add_parser(
        "perf",
        help="Run lightweight local AI performance UX smoke tests",
        description=(
            "Measure endpoint reachability, tiny chat latency, and streaming TTFT. "
            "These are timeout-bounded smoke tests, not benchmarks."
        ),
    )
    perf_subparsers = perf.add_subparsers(dest="perf_command", required=True)
    perf_endpoint = perf_subparsers.add_parser(
        "endpoint",
        help="Smoke-test OpenAI-compatible endpoint latency",
        description=(
            "Check /models and optionally run one tiny chat completion request. "
            "No models are downloaded or services started."
        ),
        epilog="Example: inferdoctor perf endpoint --endpoint http://127.0.0.1:8000/v1 --model local-model --timeout 30",
    )
    perf_endpoint.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, usually ending in /v1")
    perf_endpoint.add_argument("--model", help="Model name to use for a tiny chat completion smoke request")
    perf_endpoint.add_argument("--timeout", type=_positive_float, default=30.0, help="Strict request timeout in seconds")
    perf_endpoint.add_argument("--runs", type=_perf_runs, default=1, help="Measured request count, bounded to 1-3")
    perf_endpoint.add_argument("--warmup", type=_perf_warmup, default=0, help="Warmup request count, bounded to 0-1 and excluded from metrics")
    perf_endpoint.add_argument("--format", choices=("console", "json", "markdown"), default="console", help="Output format")
    perf_endpoint.add_argument("--output", help="Write report to a file instead of stdout")
    perf_endpoint.add_argument("--profile", choices=profile_names(), help="Application experience profile for readiness guidance")
    perf_endpoint.add_argument("--allow-non-local", action="store_true", help="Allow a tiny live smoke-test prompt to a LAN/private endpoint you control")
    perf_streaming = perf_subparsers.add_parser(
        "streaming",
        help="Smoke-test streaming TTFT for an OpenAI-compatible endpoint",
        description=(
            "Send one tiny stream=true chat completion request and measure time to first streamed chunk. "
            "This is a smoke test, not a benchmark."
        ),
        epilog="Example: inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model",
    )
    perf_streaming.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, usually ending in /v1")
    perf_streaming.add_argument("--model", required=True, help="Model name to use for the tiny streaming smoke request")
    perf_streaming.add_argument("--timeout", type=_positive_float, default=30.0, help="Strict request timeout in seconds")
    perf_streaming.add_argument("--runs", type=_perf_runs, default=1, help="Measured request count, bounded to 1-3")
    perf_streaming.add_argument("--warmup", type=_perf_warmup, default=0, help="Warmup request count, bounded to 0-1 and excluded from metrics")
    perf_streaming.add_argument("--format", choices=("console", "json", "markdown"), default="console", help="Output format")
    perf_streaming.add_argument("--output", help="Write report to a file instead of stdout")
    perf_streaming.add_argument("--profile", choices=profile_names(), help="Application experience profile for readiness guidance")
    perf_streaming.add_argument("--allow-non-local", action="store_true", help="Allow a tiny live smoke-test prompt to a LAN/private endpoint you control")

    perf_compare = perf_subparsers.add_parser(
        "compare",
        help="Compare two performance smoke-test reports or baselines",
        description=(
            "Compare before-and-after InferDoctor performance smoke-test JSON files. "
            "The comparison is heuristic and warns when inputs are not directly comparable."
        ),
        epilog="Examples: inferdoctor perf compare before.json after.json | inferdoctor perf compare --baseline before.json --candidate after.json --format markdown",
    )
    perf_compare.add_argument("paths", nargs="*", help="Optional positional baseline and candidate JSON paths")
    perf_compare.add_argument("--baseline", help="Baseline JSON path or saved baseline name")
    perf_compare.add_argument("--candidate", help="Candidate JSON path or saved baseline name")
    perf_compare.add_argument("--format", choices=("console", "json", "markdown"), default="console", help="Output format")
    perf_compare.add_argument("--output", help="Write comparison output to this file")

    perf_baseline = perf_subparsers.add_parser(
        "baseline",
        help="Save, inspect, and delete sanitized performance baselines",
        description=(
            "Manage user-local performance smoke-test baselines. Baselines store sanitized metrics, "
            "not response text, API keys, or authorization headers."
        ),
    )
    baseline_subparsers = perf_baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_create = baseline_subparsers.add_parser(
        "create",
        help="Create a sanitized baseline from a performance JSON report",
        epilog="Example: inferdoctor perf baseline create --report perf.json --name before",
    )
    baseline_create.add_argument("--report", required=True, help="Performance JSON report created by inferdoctor perf endpoint/streaming")
    baseline_create.add_argument("--name", help="Human-friendly baseline name; used for user-local storage")
    baseline_create.add_argument("--runtime", help="Runtime label such as ollama, vllm, sglang, or lmstudio")
    baseline_create.add_argument("--output", help="Write baseline JSON to this path instead of the user-local baseline directory")
    baseline_show = baseline_subparsers.add_parser(
        "show",
        help="Show a baseline by name or JSON path",
        epilog="Example: inferdoctor perf baseline show before --format markdown",
    )
    baseline_show.add_argument("baseline", help="Baseline name or JSON path")
    baseline_show.add_argument("--format", choices=("console", "json", "markdown"), default="console", help="Output format")
    baseline_show.add_argument("--output", help="Write rendered baseline output to this file")
    baseline_subparsers.add_parser(
        "list",
        help="List user-local performance baselines",
        epilog="Example: inferdoctor perf baseline list",
    )
    baseline_delete = baseline_subparsers.add_parser(
        "delete",
        help="Delete a user-local baseline by name or path",
        epilog="Example: inferdoctor perf baseline delete before --yes",
    )
    baseline_delete.add_argument("baseline", help="Baseline name or JSON path")
    baseline_delete.add_argument("--yes", action="store_true", help="Confirm deletion")

    experience = subparsers.add_parser(
        "experience",
        help="Explain local AI user-experience profiles",
        description="Show what matters for a specific local AI application goal.",
    )
    experience_subparsers = experience.add_subparsers(dest="experience_command", required=True)
    experience_profile = experience_subparsers.add_parser(
        "profile",
        help="Show an application experience profile",
        epilog="Example: inferdoctor experience profile customer-service",
    )
    experience_profile.add_argument("name", choices=profile_names(), help="Experience profile name")


    dify = subparsers.add_parser(
        "dify",
        help="Validate, smoke-test, and optimize Dify application kits",
        description=(
            "Dify integration helpers for published app APIs, Local/Private RAG kits, "
            "safe smoke tests, and performance UX guidance."
        ),
    )
    dify_subparsers = dify.add_subparsers(dest="dify_command", required=True)

    def add_dify_app_options(command):
        command.add_argument("--base-url", help="Dify application API base URL, for example http://127.0.0.1:5001/v1")
        command.add_argument("--app-key-env", default="DIFY_APP_API_KEY", help="Environment variable containing the Dify application API key")
        command.add_argument("--timeout", type=_positive_float, default=30.0, help="Strict request timeout in seconds")
        command.add_argument("--allow-non-local", action="store_true", help="Allow a tiny live request to a LAN/private Dify endpoint you control")
        command.add_argument("--allow-public", action="store_true", help="Allow a tiny live request to an explicitly supplied public Dify endpoint")
        command.add_argument("--format", choices=("console", "json", "markdown"), default="console", help="Output format")
        command.add_argument("--output", help="Write output to this file")

    dify_check = dify_subparsers.add_parser("check", help="Check Dify app API configuration and /info readiness")
    add_dify_app_options(dify_check)

    dify_template = dify_subparsers.add_parser("template", help="List, show, and export Dify application kits")
    dify_template_subparsers = dify_template.add_subparsers(dest="dify_template_command", required=True)
    dify_template_subparsers.add_parser("list", help="List built-in Dify kits")
    dify_template_show = dify_template_subparsers.add_parser("show", help="Show one Dify kit")
    dify_template_show.add_argument("name", choices=available_dify_template_names() + ["local-rag", "private-rag"])
    dify_template_export = dify_template_subparsers.add_parser("export", help="Export a Dify kit to a directory")
    dify_template_export.add_argument("name", choices=available_dify_template_names() + ["local-rag", "private-rag"])
    dify_template_export.add_argument("--output", required=True, help="Directory where kit files should be written")
    dify_template_export.add_argument("--overwrite", action="store_true", help="Overwrite generated kit files in a non-empty output directory")

    dify_validate = dify_subparsers.add_parser("validate", help="Offline-validate a Dify kit or DSL file")
    dify_validate.add_argument("path", help="Dify kit directory or dify_app.yaml path")
    dify_validate.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    dify_validate.add_argument("--output", help="Write output to this file")

    dify_smoke = dify_subparsers.add_parser("smoke", help="Run an offline or tiny live Dify app smoke test")
    dify_smoke.add_argument("--kit", help="Dify kit directory for offline dry-run validation")
    dify_smoke.add_argument("--dry-run", action="store_true", help="Do not contact Dify; validate the kit only")
    dify_smoke.add_argument("--query", help="Harmless non-sensitive query for live mode")
    dify_smoke.add_argument("--show-answer", action="store_true", help="Show a short answer preview in live mode")
    add_dify_app_options(dify_smoke)

    dify_perf = dify_subparsers.add_parser("perf", help="Run bounded Dify application performance smoke tests")
    add_dify_app_options(dify_perf)
    dify_perf.add_argument("--runs", type=_perf_runs, default=1, help="Measured request count, bounded to 1-3")
    dify_perf.add_argument("--warmup", type=_perf_warmup, default=0, help="Warmup request count, bounded to 0-1")
    dify_perf.add_argument("--query", help="Harmless non-sensitive query for the smoke test")
    dify_perf.add_argument("--profile", choices=profile_names(), help="Experience profile for readiness context")

    dify_optimize = dify_subparsers.add_parser("optimize", help="Generate Dify-specific performance optimization guidance")
    dify_optimize.add_argument("--report", help="Dify performance JSON report")
    dify_optimize.add_argument("--kit", help="Dify kit directory or DSL path for static analysis")
    dify_optimize.add_argument("--retrieval-ms", type=_positive_float, help="User-supplied retrieval latency in milliseconds")
    dify_optimize.add_argument("--rerank-ms", type=_positive_float, help="User-supplied rerank latency in milliseconds")
    dify_optimize.add_argument("--profile", choices=profile_names(), help="Experience profile for optimization priorities")
    dify_optimize.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    dify_optimize.add_argument("--output", help="Write output to this file")

    dify_knowledge = dify_subparsers.add_parser("knowledge", help="Read-only Dify knowledge retrieval checks")
    dify_knowledge_subparsers = dify_knowledge.add_subparsers(dest="dify_knowledge_command", required=True)
    dify_knowledge_check = dify_knowledge_subparsers.add_parser("check", help="Run a read-only knowledge retrieval check")
    dify_knowledge_check.add_argument("--base-url", help="Dify knowledge API base URL")
    dify_knowledge_check.add_argument("--knowledge-key-env", default="DIFY_KNOWLEDGE_API_KEY", help="Environment variable containing the Dify knowledge API key")
    dify_knowledge_check.add_argument("--dataset-id", help="Dify dataset ID; defaults to DIFY_DATASET_ID")
    dify_knowledge_check.add_argument("--query", default="fictional return policy", help="Harmless retrieval query")
    dify_knowledge_check.add_argument("--show-content", action="store_true", help="Show short retrieved content previews")
    dify_knowledge_check.add_argument("--timeout", type=_positive_float, default=30.0)
    dify_knowledge_check.add_argument("--allow-non-local", action="store_true")
    dify_knowledge_check.add_argument("--allow-public", action="store_true")
    dify_knowledge_check.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    dify_knowledge_check.add_argument("--output", help="Write output to this file")

    rag = subparsers.add_parser("rag", help="Diagnose RAG answer quality with deterministic layered evidence")
    rag_subparsers = rag.add_subparsers(dest="rag_command", required=True)
    rag_case = rag_subparsers.add_parser("case", help="Create and validate RAG Case files")
    rag_case_subparsers = rag_case.add_subparsers(dest="rag_case_command", required=True)
    rag_case_init = rag_case_subparsers.add_parser("init", help="Write a fictional RAG Case JSONL template")
    rag_case_init.add_argument("--output", required=True)
    rag_case_validate = rag_case_subparsers.add_parser("validate", help="Validate a RAG Case JSON or JSONL file")
    rag_case_validate.add_argument("path")
    rag_case_validate.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    rag_case_validate.add_argument("--output")
    rag_capture = rag_subparsers.add_parser(
        "capture",
        help="Capture framework evidence into a standard RAG Trace",
    )
    rag_capture_subparsers = rag_capture.add_subparsers(
        dest="rag_capture_command",
        required=True,
    )
    rag_capture_dify = rag_capture_subparsers.add_parser(
        "dify-knowledge",
        help="Capture Dify Knowledge API retrieval evidence",
    )
    rag_capture_dify.add_argument(
        "--base-url",
        help="Dify knowledge API base URL",
    )
    rag_capture_dify.add_argument(
        "--knowledge-key-env",
        default="DIFY_KNOWLEDGE_API_KEY",
        help="Environment variable containing the Dify knowledge API key",
    )
    rag_capture_dify.add_argument(
        "--dataset-id",
        help="Dify dataset ID; defaults to DIFY_DATASET_ID",
    )
    rag_capture_dify.add_argument(
        "--query",
        required=True,
        help="Query to send to the Dify Knowledge API",
    )
    rag_capture_dify.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Requested Dify retrieval top_k",
    )
    rag_capture_dify.add_argument(
        "--case-id",
        help="Optional RAG Case ID to associate with this trace",
    )
    rag_capture_dify.add_argument(
        "--include-content",
        action="store_true",
        help="Explicitly retain query and retrieved chunk text in the trace",
    )
    rag_capture_dify.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
    )
    rag_capture_dify.add_argument(
        "--allow-non-local",
        action="store_true",
    )
    rag_capture_dify.add_argument(
        "--allow-public",
        action="store_true",
    )
    rag_capture_dify.add_argument(
        "--output",
        required=True,
        help="Write the captured RAG Trace JSON to this path",
    )

    rag_trace = rag_subparsers.add_parser("trace", help="Validate RAG Trace files")
    rag_trace_subparsers = rag_trace.add_subparsers(dest="rag_trace_command", required=True)
    rag_trace_validate = rag_trace_subparsers.add_parser("validate", help="Validate a RAG Trace JSON file")
    rag_trace_validate.add_argument("path")
    rag_trace_validate.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    rag_trace_validate.add_argument("--output")
    rag_diagnose = rag_subparsers.add_parser("diagnose", help="Diagnose one RAG Case + Trace with layered deterministic evidence")
    rag_diagnose.add_argument("--case", required=True)
    rag_diagnose.add_argument("--trace", required=True)
    rag_diagnose.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    rag_diagnose.add_argument("--output")
    rag_compare = rag_subparsers.add_parser("compare", help="Compare before/after RAG traces for one Case")
    rag_compare.add_argument("--case", required=True)
    rag_compare.add_argument("--before", required=True)
    rag_compare.add_argument("--after", required=True)
    rag_compare.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    rag_compare.add_argument("--output")
    rag_probe = rag_subparsers.add_parser("probe", help="Run bounded RAG diagnostic probes")
    rag_probe_subparsers = rag_probe.add_subparsers(dest="rag_probe_command", required=True)
    rag_gold = rag_probe_subparsers.add_parser("gold-context", help="Probe whether a model can answer when explicit gold context is supplied")
    rag_gold.add_argument("--case", required=True)
    rag_gold.add_argument("--context-file", required=True)
    rag_gold.add_argument("--endpoint", required=True)
    rag_gold.add_argument("--model", required=True)
    rag_gold.add_argument("--api-key-env", help="Environment variable containing an API key, if needed")
    rag_gold.add_argument("--timeout", type=_positive_float, default=30.0)
    rag_gold.add_argument("--dry-run", action="store_true")
    rag_gold.add_argument("--allow-non-local", action="store_true")
    rag_gold.add_argument("--allow-public", action="store_true")
    rag_gold.add_argument("--retain-answer", action="store_true", help="Retain a short generated answer preview in the report")
    rag_gold.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    rag_gold.add_argument("--output")

    def add_dify_compose_options(command):
        command.add_argument("--compose-file", help="Dify Docker Compose file to inspect read-only")
        command.add_argument("--project-directory", help="Directory containing compose.yaml or docker-compose.yaml")
        command.add_argument("--project-name", help="Optional Compose project name used for report context")
        command.add_argument("--format", choices=("console", "json", "markdown"), default="console")
        command.add_argument("--output", help="Write output to this file")

    dify_selfhost = dify_subparsers.add_parser("selfhost", help="Read-only self-hosted Dify deployment diagnostics")
    dify_selfhost_subparsers = dify_selfhost.add_subparsers(dest="dify_selfhost_command", required=True)
    dify_selfhost_preflight = dify_selfhost_subparsers.add_parser("preflight", help="Check host, Docker, Compose, and service readiness without starting anything")
    add_dify_compose_options(dify_selfhost_preflight)
    dify_selfhost_inspect = dify_selfhost_subparsers.add_parser("inspect", help="Inspect an existing selected Dify Compose deployment read-only")
    add_dify_compose_options(dify_selfhost_inspect)
    dify_selfhost_inspect.add_argument("--since", default="10m", help="Bounded log window for --details, for example 10m")
    dify_selfhost_inspect.add_argument("--services", help="Comma-separated service names or roles to inspect")
    dify_selfhost_inspect.add_argument("--details", action="store_true", help="Collect bounded redacted log signatures for failed or selected services")

    dify_connectivity = dify_subparsers.add_parser("connectivity", help="Diagnose model endpoint connectivity across host, containers, and Dify layers")
    dify_connectivity_subparsers = dify_connectivity.add_subparsers(dest="dify_connectivity_command", required=True)
    dify_connectivity_check = dify_connectivity_subparsers.add_parser("check", help="Run layered model connectivity diagnosis")
    add_dify_compose_options(dify_connectivity_check)
    dify_connectivity_check.add_argument("--endpoint", help="Model endpoint URL to test, for example http://192.168.1.20:8000/v1")
    dify_connectivity_check.add_argument("--runtime", choices=("auto", "openai-compatible", "ollama", "vllm", "sglang", "xinference"), default="auto")
    dify_connectivity_check.add_argument("--role", choices=("chat", "embedding", "rerank", "tool"), default="chat")
    dify_connectivity_check.add_argument("--services", help="Comma-separated service names or roles for container probes")
    dify_connectivity_check.add_argument("--path", help="Optional endpoint route to probe instead of /models")
    dify_connectivity_check.add_argument("--through-dify", action="store_true", help="Include Dify-mediated provider path evidence when app API options are supplied")
    dify_connectivity_check.add_argument("--app-api-base", help="Published Dify app API base for Dify-mediated checks")
    dify_connectivity_check.add_argument("--app-key-env", default="DIFY_APP_API_KEY", help="Environment variable containing the Dify app API key")
    dify_connectivity_check.add_argument("--allow-non-local", action="store_true", help="Allow a bounded probe to a LAN/private endpoint you control")
    dify_connectivity_check.add_argument("--allow-public", action="store_true", help="Allow a bounded probe to an explicitly supplied public endpoint")
    dify_connectivity_check.add_argument("--details", action="store_true", help="Attempt bounded read-only probes from detected containers")

    dify_evidence = dify_subparsers.add_parser("evidence", help="Collect and explain bounded redacted Dify evidence bundles")
    dify_evidence_subparsers = dify_evidence.add_subparsers(dest="dify_evidence_command", required=True)
    dify_evidence_collect = dify_evidence_subparsers.add_parser("collect", help="Collect a bounded redacted self-host evidence bundle")
    add_dify_compose_options(dify_evidence_collect)
    dify_evidence_collect.add_argument("--since", default="10m")
    dify_evidence_collect.add_argument("--services", help="Comma-separated service names or roles")
    dify_evidence_collect.add_argument("--details", action="store_true", help="Collect bounded redacted log signatures")
    dify_evidence_explain = dify_evidence_subparsers.add_parser("explain", help="Explain a saved Dify evidence bundle")
    dify_evidence_explain.add_argument("bundle", help="Evidence bundle JSON path")
    dify_evidence_explain.add_argument("--format", choices=("console", "json", "markdown"), default="console")
    dify_evidence_explain.add_argument("--output", help="Write output to this file")

    report = subparsers.add_parser(
        "report",
        help="Generate a JSON or Markdown diagnostic report",
        description="Run all checks and create a shareable diagnostic report.",
    )
    report.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Report output format",
    )
    report.add_argument("--output", help="Write the report to this file")
    _add_runtime_options(report, include_language=False)

    profile = subparsers.add_parser(
        "profile",
        help="Generate a safe, redacted diagnostic profile",
        description=(
            "Create a shareable local AI environment profile with secrets, "
            "endpoint credentials, query strings, and home paths redacted."
        ),
        epilog="Examples: inferdoctor profile --format markdown | inferdoctor profile --format json --output profile.json",
    )
    profile.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Profile output format",
    )
    profile.add_argument("--output", help="Write the profile to this file")
    _add_runtime_options(profile, include_language=False)

    explain = subparsers.add_parser(
        "explain",
        help="Explain a common local AI failure",
        description="Show a short troubleshooting guide for a known InferDoctor topic.",
        epilog="Example: inferdoctor explain openai-compatible-404",
    )
    explain.add_argument(
        "topic",
        choices=explain_topics(),
        help="Troubleshooting topic to explain",
    )

    recommend = subparsers.add_parser(
        "recommend",
        help="Recommend a local AI stack path",
        description=(
            "Suggest a runtime, model size class, and starter template using "
            "lightweight hardware heuristics."
        ),
        epilog="Examples: inferdoctor recommend --goal customer-service --vram 24 | inferdoctor recommend --goal document-qa --preference easiest",
    )
    recommend.add_argument(
        "--goal",
        choices=GOALS,
        help="What you want to build",
    )
    recommend.add_argument(
        "--preference",
        choices=PREFERENCES,
        default="easiest",
        help="Optimize for easiest setup or performance",
    )
    recommend.add_argument(
        "--hardware",
        choices=("auto",),
        default="auto",
        help="Hardware source; currently auto only",
    )
    recommend.add_argument(
        "--vram",
        type=_positive_float,
        help="Override detected VRAM in GiB",
    )

    quickstart = subparsers.add_parser(
        "quickstart",
        help="Plan a guided local or private AI app quickstart",
        description=(
            "Recommend a stack, template, endpoint configuration path, validation commands, "
            "and performance verification steps. No installation is performed."
        ),
        epilog="Examples: inferdoctor quickstart customer-service --preference easiest | inferdoctor quickstart rag --endpoint http://192.168.1.20:8000/v1",
    )
    quickstart.add_argument("goal", nargs="?", choices=QUICKSTART_GOALS, help="What you want to build")
    quickstart.add_argument("--preference", choices=QUICKSTART_PREFERENCES, default="easiest", help="Optimize for easiest setup or performance")
    quickstart.add_argument("--endpoint", help="Existing local, LAN, or private OpenAI-compatible endpoint")
    quickstart.add_argument("--location", choices=QUICKSTART_LOCATIONS, help="Where the endpoint runs: local, lan, or endpoint")
    quickstart.add_argument("--hardware", choices=QUICKSTART_HARDWARE, default="auto", help="Hardware hint")
    quickstart.add_argument("--runtime", choices=QUICKSTART_RUNTIMES, help="Existing runtime if known")

    init = subparsers.add_parser(
        "init",
        help="Get a guided local AI setup recommendation",
        description=(
            "Ask a few lightweight questions and recommend a runtime path, "
            "template, and next commands. No installation is performed."
        ),
        epilog="Examples: inferdoctor init --goal customer-service --preference easiest | inferdoctor init --goal document-qa --preference gpu",
    )
    init.add_argument(
        "--goal",
        choices=GOALS,
        help="What you want to build",
    )
    init.add_argument(
        "--preference",
        choices=PREFERENCES,
        help="Optimize for easiest setup, performance, CPU, or GPU",
    )
    init.add_argument(
        "--runtime",
        choices=RUNTIMES,
        help="Local runtime you already have, if known",
    )

    capacity = subparsers.add_parser(
        "capacity",
        help="Preview local AI workload capacity",
        description=(
            "Estimate local AI hardware readiness with lightweight heuristics. "
            "No models are downloaded or run."
        ),
        epilog="Examples: inferdoctor capacity --vram 24 --model-size 14b --quant q4 | inferdoctor capacity --gpu 'RTX 3090'",
    )
    capacity.add_argument(
        "--vram",
        type=_positive_float,
        help="Override detected NVIDIA VRAM in GiB",
    )
    capacity.add_argument(
        "--gpu",
        help="GPU name to display or infer common VRAM from, for example 'RTX 3090'",
    )
    capacity.add_argument(
        "--model-size",
        type=_model_size,
        help="Optional model size heuristic, for example 7b, 14b, or 32b",
    )
    capacity.add_argument(
        "--quant",
        choices=("q4", "q8"),
        default="q4",
        help="Quantization heuristic to use with --model-size",
    )
    capacity.add_argument(
        "--runtime",
        choices=("ollama", "vllm"),
        help="Runtime heuristic to apply",
    )

    stack = subparsers.add_parser(
        "stack",
        help="Plan a beginner-friendly local AI app stack",
        description="Create a read-only plan for building a local AI app on this machine.",
    )
    stack_subparsers = stack.add_subparsers(dest="stack_command", required=True)
    stack_plan = stack_subparsers.add_parser(
        "plan",
        help="Create a local AI app stack plan",
        description=(
            "Recommend a runtime path, model size class, starter template, required "
            "components, and next commands. This command is advisory and read-only."
        ),
        epilog="Examples: inferdoctor stack plan --goal customer-service --vram 24 | inferdoctor stack plan --goal restaurant-ordering --preference easiest",
    )
    stack_plan.add_argument("--goal", choices=GOALS, help="What you want to build")
    stack_plan.add_argument(
        "--preference",
        choices=PREFERENCES,
        default="easiest",
        help="Optimize for easiest setup, performance, CPU, or GPU",
    )
    stack_plan.add_argument(
        "--hardware",
        choices=("auto",),
        default="auto",
        help="Hardware source; currently auto only",
    )
    stack_plan.add_argument("--vram", type=_positive_float, help="Override detected VRAM in GiB")
    stack_bootstrap = stack_subparsers.add_parser(
        "bootstrap",
        help="Show a dry-run bootstrap plan for a local AI app",
        description=(
            "Print the exact beginner commands for creating, validating, and smoke-testing "
            "a local AI starter project. This is a plan, not an executor."
        ),
        epilog="Example: inferdoctor stack bootstrap --goal customer-service --dry-run",
    )
    stack_bootstrap.add_argument("--goal", choices=GOALS, help="What you want to build")
    stack_bootstrap.add_argument(
        "--preference",
        choices=PREFERENCES,
        default="easiest",
        help="Optimize for easiest setup, performance, CPU, or GPU",
    )
    stack_bootstrap.add_argument(
        "--hardware",
        choices=("auto",),
        default="auto",
        help="Hardware source; currently auto only",
    )
    stack_bootstrap.add_argument("--vram", type=_positive_float, help="Override detected VRAM in GiB")
    stack_bootstrap.add_argument("--output", help="Project path to show in the plan")
    stack_bootstrap.add_argument(
        "--dry-run",
        action="store_true",
        help="Required safety flag; only print the plan and do not execute it",
    )

    template = subparsers.add_parser(
        "template",
        help="Explore, create, and validate local AI starter templates",
        description=(
            "List, inspect, create, and validate local AI app templates. Template commands do not "
            "download models or install runtimes."
        ),
        epilog="Beginner flow: inferdoctor template list | inferdoctor template create customer-service --output ./demo | inferdoctor template validate ./demo | inferdoctor template smoke-test ./demo",
    )
    template_subparsers = template.add_subparsers(
        dest="template_command", required=True
    )
    template_subparsers.add_parser(
        "list",
        help="List available starter templates",
        description="Show beginner-friendly local AI app templates.",
    )
    template_subparsers.add_parser(
        "registry",
        help="Show built-in template source and future registry safety rules",
        description="Explain built-in templates and future community template registry principles.",
    )
    template_show = template_subparsers.add_parser(
        "show",
        help="Show details for one starter template",
        description="Explain what a template builds, what it needs, and how to start.",
    )
    template_show.add_argument(
        "template",
        choices=template_names(),
        help="Template name to inspect",
    )
    template_create = template_subparsers.add_parser(
        "create",
        help="Create a lightweight starter project",
        description=(
            "Generate a local starter project. This writes files only to the "
            "explicit --output directory and does not install dependencies."
        ),
        epilog="Examples: inferdoctor template create customer-service --output ./customer-service-demo | inferdoctor template create local-doc-qa --output ./docqa-demo",
    )
    template_create.add_argument(
        "template",
        choices=template_names(),
        help="Template name to generate",
    )
    template_create.add_argument(
        "--output",
        required=True,
        help="Directory where the starter project should be written",
    )
    template_validate = template_subparsers.add_parser(
        "validate",
        help="Validate a generated starter project",
        description=(
            "Read a generated template directory and check required files, "
            "endpoint configuration, and obvious secret-like values. No dependencies "
            "are installed and no endpoints are called."
        ),
        epilog="Examples: inferdoctor template validate ./customer-service-demo | inferdoctor template smoke-test ./customer-service-demo",
    )
    template_validate.add_argument(
        "path",
        help="Generated template project directory to validate",
    )
    template_smoke = template_subparsers.add_parser(
        "smoke-test",
        help="Run safe dry-run checks for a generated starter project",
        description=(
            "Run only allowlisted help, dry-run, and config-check commands inside a generated "
            "template directory. No dependencies are installed and no endpoints are called."
        ),
        epilog="Example: inferdoctor template smoke-test ./customer-service-demo",
    )
    template_smoke.add_argument(
        "path",
        help="Generated template project directory to smoke-test",
    )
    template_smoke.add_argument(
        "--timeout",
        type=_positive_float,
        default=5.0,
        help="Per-command timeout in seconds",
    )

    template_compose = template_subparsers.add_parser(
        "compose",
        help="Generate optional Docker Compose files for a starter template",
        description=(
            "Generate Docker Compose starter files only. This does not pull images, "
            "start containers, install runtimes, or call endpoints."
        ),
        epilog="Example: inferdoctor template compose customer-service --output ./compose-customer-service",
    )
    template_compose.add_argument(
        "template",
        choices=compose_template_names(),
        help="Template name for Compose guidance",
    )
    template_compose.add_argument(
        "--output",
        required=True,
        help="Directory where Compose starter files should be written",
    )

    def add_scenario_parser(name: str):
        scenario_parser = subparsers.add_parser(
            name,
            help="Show goal-oriented scenario readiness",
            description="Summarize readiness for common local AI goals using existing checks.",
            epilog="Examples: inferdoctor scenario | inferdoctor scenario openai-compatible-server",
        )
        scenario_parser.add_argument(
            "target",
            nargs="?",
            choices=scenario_names(),
            help="Scenario to show; omit to show all scenarios",
        )
        _add_runtime_options(scenario_parser, include_language=False)

    add_scenario_parser("scenario")
    add_scenario_parser("scenarios")
    return parser


def _load(path: Optional[str]):
    try:
        return load_config(path)
    except ConfigError as exc:
        raise SystemExit(
            "inferdoctor: configuration error: {0}. "
            "Check the path and the documented endpoints/timeout format.".format(exc)
        )


def _results_for_target(
    target: Optional[str],
    config_path: Optional[str],
    timeout: Optional[float] = None,
    endpoint: Optional[str] = None,
    language: Optional[str] = None,
) -> Tuple[List[CheckResult], Config]:
    registry = default_registry()
    checkers = [registry.get(target)] if target else registry.all()
    config = _load(config_path)
    if timeout is not None:
        config.timeout = timeout
    if language is not None:
        config.language = language
    if endpoint is not None:
        if target is None:
            raise SystemExit("inferdoctor: --endpoint requires a component name")
        if target not in config.endpoints:
            raise SystemExit(
                "inferdoctor: {0} does not use an HTTP endpoint".format(target)
            )
        try:
            config.endpoints[target] = normalize_endpoint(target, endpoint)
        except ConfigError as exc:
            raise SystemExit("inferdoctor: {0}".format(exc)) from exc
    return run_checks(checkers, config), config



def _render_perf_output(result, output_format: str) -> str:
    if output_format == "json":
        return render_perf_json(result)
    if output_format == "markdown":
        return render_perf_markdown(result)
    return render_perf_result(result)


def _emit_output(content: str, output: Optional[str]) -> None:
    if output:
        Path(output).write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


def _exit_code(results: List[CheckResult]) -> int:
    return 1 if any(result.status == Status.FAIL for result in results) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        arguments = ["check"]
    args = parser.parse_args(arguments)
    if args.command is None:
        args = parser.parse_args(["check"] + arguments)
    if getattr(args, "language", None) is not None and args.command != "check":
        parser.error(
            "--language currently applies only to the default health dashboard and inferdoctor check; "
            "other commands may remain English in this first i18n release"
        )

    if args.command == "explain":
        print(render_explanation(args.topic))
        return 0
    if args.command == "experience":
        if args.experience_command == "profile":
            print(render_experience_profile(get_profile(args.name)))
            return 0
    if args.command == "optimize":
        if args.optimize_command == "endpoint":
            print(render_optimization_report(apply_profile_to_optimization_report(advise_endpoint(
                runtime=args.runtime,
                vram_gib=args.vram,
                model_size=args.model_size,
                quant=args.quant,
                streaming=args.streaming,
                ttft=args.ttft,
                tps=args.tps,
                latency=args.latency,
                context_tokens=args.context_tokens,
                ttft_variance=args.ttft_variance,
                containerized=args.containerized,
                docker=args.docker,
                cold_start=args.cold_start,
                cpu_fallback_suspected=args.cpu_fallback_suspected,
            ), args.profile)))
            return 0
        if args.optimize_command == "rag":
            print(render_optimization_report(apply_profile_to_optimization_report(advise_rag(
                docs=args.docs,
                chunks=args.chunks,
                top_k=args.top_k,
                rerank=args.rerank,
                retrieval_ms=args.retrieval_ms,
                rerank_ms=args.rerank_ms,
                embedding_ms=args.embedding_ms,
                filter_ms=args.filter_ms,
                doc_load_ms=args.doc_load_ms,
                context_build_ms=args.context_build_ms,
                generation_ms=args.generation_ms,
                ttft=args.ttft,
                streaming=args.streaming,
                model_size=args.model_size,
                vram_gib=args.vram,
            ), args.profile)))
            return 0
        if args.optimize_command == "plan":
            if (args.baseline and not args.candidate) or (args.candidate and not args.baseline):
                print("inferdoctor: optimize plan requires both --baseline and --candidate when comparing", file=sys.stderr)
                return 2
            try:
                plan = build_optimization_plan(
                    report_path=args.report,
                    baseline_path=args.baseline,
                    candidate_path=args.candidate,
                    runtime=args.runtime,
                    model_size=args.model_size,
                    vram_gib=args.vram,
                    goal=args.goal,
                    streaming=args.streaming,
                    retrieval_ms=args.retrieval_ms,
                    rerank_ms=args.rerank_ms,
                    ttft=args.ttft,
                    profile=args.profile,
                )
            except ValueError as exc:
                print("inferdoctor: {0}".format(exc), file=sys.stderr)
                return 2
            _emit_output(render_optimization_plan(plan, args.format), args.output)
            return 0

    if args.command == "perf":
        if args.perf_command == "endpoint":
            safety = classify_endpoint(args.endpoint)
            if safety.category == "invalid" or (safety.requires_explicit_allow and not args.allow_non_local):
                print("inferdoctor: {0}".format(render_endpoint_safety_error(safety)), file=sys.stderr)
                return 2
            result = apply_profile_to_perf_result(run_endpoint_smoke(args.endpoint, args.model, args.timeout, runs=args.runs, warmup=args.warmup), args.profile)
            _emit_output(_render_perf_output(result, args.format), args.output)
            return 0
        if args.perf_command == "streaming":
            safety = classify_endpoint(args.endpoint)
            if safety.category == "invalid" or (safety.requires_explicit_allow and not args.allow_non_local):
                print("inferdoctor: {0}".format(render_endpoint_safety_error(safety)), file=sys.stderr)
                return 2
            result = apply_profile_to_perf_result(run_streaming_smoke(args.endpoint, args.model, args.timeout, runs=args.runs, warmup=args.warmup), args.profile)
            _emit_output(_render_perf_output(result, args.format), args.output)
            return 0
        if args.perf_command == "compare":
            paths = list(args.paths or [])
            if len(paths) > 2:
                print("inferdoctor: perf compare accepts at most two positional paths", file=sys.stderr)
                return 2
            baseline_path = args.baseline or (paths[0] if len(paths) >= 1 else None)
            candidate_path = args.candidate or (paths[1] if len(paths) >= 2 else None)
            if not baseline_path or not candidate_path:
                print("inferdoctor: perf compare requires a baseline and candidate JSON path", file=sys.stderr)
                return 2
            try:
                comparison = compare_performance_files(baseline_path, candidate_path)
            except ValueError as exc:
                print("inferdoctor: {0}".format(exc), file=sys.stderr)
                return 2
            _emit_output(render_comparison(comparison, args.format), args.output)
            return 0
        if args.perf_command == "baseline":
            try:
                if args.baseline_command == "create":
                    baseline, path = create_baseline_from_report_file(
                        args.report,
                        name=args.name,
                        runtime=args.runtime,
                        output=args.output,
                    )
                    print(render_baseline_summary(baseline, path))
                    return 0
                if args.baseline_command == "show":
                    baseline = load_report_or_baseline(args.baseline)
                    if args.format == "json":
                        rendered = json.dumps(baseline, indent=2, sort_keys=True)
                    elif args.format == "markdown":
                        rendered = render_baseline_markdown(baseline, args.baseline)
                    else:
                        rendered = render_baseline_summary(baseline, args.baseline)
                    _emit_output(rendered, args.output)
                    return 0
                if args.baseline_command == "list":
                    print(render_baseline_list(list_baselines()))
                    return 0
                if args.baseline_command == "delete":
                    if not args.yes:
                        print("inferdoctor: refusing to delete baseline without --yes", file=sys.stderr)
                        return 2
                    deleted = delete_baseline(args.baseline)
                    print("Deleted performance baseline: {0}".format(deleted))
                    return 0
            except ValueError as exc:
                print("inferdoctor: {0}".format(exc), file=sys.stderr)
                return 2


    if args.command == "rag":
        try:
            if args.rag_command == "case":
                if args.rag_case_command == "init":
                    output = init_case_template(args.output)
                    print("Wrote RAG Case template: {0}".format(output))
                    return 0
                if args.rag_case_command == "validate":
                    result = validate_case_file(args.path)
                    _emit_output(render_rag_result(result, args.format), args.output)
                    return 1 if result.get("status") == "FAIL" else 0
            if (
                args.rag_command == "capture"
                and args.rag_capture_command == "dify-knowledge"
            ):
                config = load_dify_config(
                    knowledge_base_url=args.base_url,
                    knowledge_key_env=args.knowledge_key_env,
                    dataset_id=args.dataset_id,
                    timeout=args.timeout,
                    allow_non_local=args.allow_non_local,
                    allow_public=args.allow_public,
                )

                result = capture_dify_knowledge_trace(
                    config,
                    query=args.query,
                    top_k=args.top_k,
                    include_content=args.include_content,
                    case_id=args.case_id,
                )

                Path(args.output).write_text(
                    json.dumps(
                        result,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                print(
                    "Wrote Dify RAG Trace: {0}".format(
                        args.output
                    )
                )

                return 0

            if args.rag_command == "trace" and args.rag_trace_command == "validate":
                result = validate_trace_file(args.path)
                _emit_output(render_rag_result(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.rag_command == "diagnose":
                result = diagnose_rag(load_case(args.case), load_trace(args.trace))
                _emit_output(render_rag_result(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.rag_command == "compare":
                result = compare_rag(load_case(args.case), load_trace(args.before), load_trace(args.after))
                _emit_output(render_rag_result(result, args.format), args.output)
                return 1 if result.get("verdict") in {"regressed", "incompatible"} else 0
            if args.rag_command == "probe" and args.rag_probe_command == "gold-context":
                api_key = None
                if args.api_key_env:
                    import os
                    api_key = os.environ.get(args.api_key_env)
                context_text = Path(args.context_file).read_text(encoding="utf-8")
                result = run_gold_context_probe(load_case(args.case), context_text=context_text, endpoint=args.endpoint, model=args.model, timeout=args.timeout, dry_run=args.dry_run, allow_non_local=args.allow_non_local, allow_public=args.allow_public, api_key=api_key, retain_answer=args.retain_answer)
                _emit_output(render_rag_result(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
        except (RagError, OSError, json.JSONDecodeError) as exc:
            print("inferdoctor: {0}".format(exc), file=sys.stderr)
            return 2

    if args.command == "dify":
        try:
            if args.dify_command == "check":
                config = load_dify_config(base_url=args.base_url, app_key_env=args.app_key_env, timeout=args.timeout, allow_non_local=args.allow_non_local, allow_public=args.allow_public)
                result = run_dify_check(config)
                _emit_output(render_dify_check(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.dify_command == "template":
                if args.dify_template_command == "list":
                    print(render_dify_template_list())
                    return 0
                if args.dify_template_command == "show":
                    print(render_dify_template_show(args.name))
                    return 0
                if args.dify_template_command == "export":
                    written = export_dify_template(args.name, args.output, overwrite=args.overwrite)
                    print(render_dify_template_export(args.name, args.output, written))
                    return 0
            if args.dify_command == "validate":
                result = validate_dify_kit(args.path)
                _emit_output(render_dify_validation(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.dify_command == "smoke":
                config = load_dify_config(base_url=args.base_url, app_key_env=args.app_key_env, timeout=args.timeout, allow_non_local=args.allow_non_local, allow_public=args.allow_public)
                result = run_dify_smoke(config, kit_path=args.kit, dry_run=args.dry_run, query=args.query, show_answer=args.show_answer)
                _emit_output(render_dify_smoke(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.dify_command == "perf":
                config = load_dify_config(base_url=args.base_url, app_key_env=args.app_key_env, timeout=args.timeout, allow_non_local=args.allow_non_local, allow_public=args.allow_public)
                result = run_dify_perf(config, runs=args.runs, warmup=args.warmup, profile=args.profile, query=args.query)
                _emit_output(render_dify_perf(result, args.format), args.output)
                return 1 if result.get("failed_runs", 0) and not result.get("successful_runs", 0) else 0
            if args.dify_command == "optimize":
                result = optimize_dify(report_path=args.report, kit_path=args.kit, retrieval_ms=args.retrieval_ms, rerank_ms=args.rerank_ms, profile=args.profile)
                _emit_output(render_dify_optimize(result, args.format), args.output)
                return 0
            if args.dify_command == "knowledge" and args.dify_knowledge_command == "check":
                config = load_dify_config(knowledge_base_url=args.base_url, knowledge_key_env=args.knowledge_key_env, dataset_id=args.dataset_id, timeout=args.timeout, allow_non_local=args.allow_non_local, allow_public=args.allow_public)
                result = run_dify_knowledge_check(config, query=args.query, show_content=args.show_content)
                _emit_output(render_dify_knowledge(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.dify_command == "selfhost":
                if args.dify_selfhost_command == "preflight":
                    result = run_dify_selfhost_preflight(compose_file=args.compose_file, project_directory=args.project_directory, project_name=args.project_name)
                    _emit_output(render_reliability_report(result, args.format), args.output)
                    return 1 if result.get("status") == "BLOCKED" else 0
                if args.dify_selfhost_command == "inspect":
                    services = args.services.split(",") if args.services else None
                    result = run_dify_selfhost_inspect(compose_file=args.compose_file, project_directory=args.project_directory, project_name=args.project_name, since=args.since, services=services, details=args.details)
                    _emit_output(render_reliability_report(result, args.format), args.output)
                    return 1 if result.get("status") == "BLOCKED" else 0
            if args.dify_command == "connectivity" and args.dify_connectivity_command == "check":
                services = args.services.split(",") if args.services else None
                result = run_dify_connectivity_check(endpoint=args.endpoint, runtime=args.runtime, role=args.role, compose_file=args.compose_file, project_directory=args.project_directory, project_name=args.project_name, services=services, path=args.path, through_dify=args.through_dify, app_api_base=args.app_api_base, app_key_env=args.app_key_env, allow_non_local=args.allow_non_local, allow_public=args.allow_public, details=args.details)
                _emit_output(render_reliability_report(result, args.format), args.output)
                return 1 if result.get("status") == "FAIL" else 0
            if args.dify_command == "evidence":
                if args.dify_evidence_command == "collect":
                    services = args.services.split(",") if args.services else None
                    result = run_dify_evidence_collect(compose_file=args.compose_file, project_directory=args.project_directory, project_name=args.project_name, since=args.since, services=services, details=args.details)
                    _emit_output(render_reliability_report(result, args.format), args.output)
                    return 0
                if args.dify_evidence_command == "explain":
                    result = run_dify_evidence_explain(args.bundle)
                    _emit_output(render_reliability_report(result, args.format), args.output)
                    return 0
        except (DifyError, KeyError, OSError, json.JSONDecodeError) as exc:
            print("inferdoctor: {0}".format(exc), file=sys.stderr)
            return 2

    if args.command == "recommend":
        print(
            render_recommendation(
                recommend_stack(
                    goal=args.goal,
                    preference=args.preference,
                    hardware=args.hardware,
                    vram_gib=args.vram,
                )
            )
        )
        return 0
    if args.command == "quickstart":
        goal = args.goal
        preference = args.preference
        location = args.location
        hardware = args.hardware
        runtime = args.runtime
        endpoint = args.endpoint
        interactive = sys.stdin.isatty() and goal is None and endpoint is None and runtime is None
        if interactive:
            goal = input("What do you want to build? [customer-service/restaurant-ordering/document-qa/rag/local-api/not-sure]: " ).strip() or None
            preference = input("Prefer easiest setup or performance? [easiest/performance]: " ).strip() or preference
            location = input("Endpoint location? [local/lan/endpoint]: " ).strip() or location
            hardware = input("Hardware? [auto/cpu/gpu]: " ).strip() or hardware
            runtime = input("Existing runtime? [ollama/vllm/sglang/xinference/openai-compatible/not-sure]: " ).strip() or runtime
        print(render_quickstart_plan(build_quickstart_plan(
            goal=goal,
            preference=preference,
            endpoint=endpoint,
            location=location,
            hardware=hardware,
            runtime=runtime,
        )))
        return 0
    if args.command == "init":
        goal = args.goal
        preference = args.preference
        runtime = args.runtime
        interactive = sys.stdin.isatty() and goal is None and preference is None and runtime is None
        if interactive:
            goal = input("What do you want to build? [chatbot/document-qa/customer-service/restaurant-ordering/local-api/not-sure]: ").strip() or None
            preference = input("What do you prefer? [easiest/performance/cpu/gpu]: ").strip() or None
            runtime = input("Existing runtime? [ollama/vllm/sglang/xinference/not-sure]: ").strip() or None
        print(render_setup_plan(recommend_setup(goal, preference, runtime)))
        return 0
    if args.command == "model":
        if args.model_command == "fit":
            print(
                render_model_fit(
                    estimate_model_fit(
                        size=args.size,
                        quant=args.quant,
                        runtime=args.runtime,
                        vram_gib=args.vram,
                    )
                )
            )
            return 0
    if args.command == "capacity":
        print(
            render_capacity(
                vram_gib=args.vram,
                gpu_name=args.gpu,
                model_size_b=args.model_size,
                quant=args.quant,
                runtime=args.runtime,
            )
        )
        return 0
    if args.command == "stack":
        if args.stack_command == "plan":
            print(
                render_stack_plan(
                    build_stack_plan(
                        goal=args.goal,
                        preference=args.preference,
                        hardware=args.hardware,
                        vram_gib=args.vram,
                    )
                )
            )
            return 0
        if args.stack_command == "bootstrap":
            if args.dry_run:
                print(
                    render_stack_bootstrap_plan(
                        build_stack_bootstrap_plan(
                            goal=args.goal,
                            preference=args.preference,
                            hardware=args.hardware,
                            vram_gib=args.vram,
                            output_dir=args.output,
                        )
                    )
                )
                return 0
            if args.output:
                print(
                    render_stack_bootstrap_files(
                        create_stack_bootstrap_project(
                            goal=args.goal,
                            preference=args.preference,
                            hardware=args.hardware,
                            vram_gib=args.vram,
                            output_dir=args.output,
                        )
                    )
                )
                return 0
            print("inferdoctor: stack bootstrap requires --dry-run or --output; no commands were executed.", file=sys.stderr)
            return 2
    if args.command == "template":
        try:
            if args.template_command == "list":
                print(render_template_list())
            elif args.template_command == "registry":
                print(render_template_registry())
            elif args.template_command == "show":
                print(render_template_detail(args.template))
            elif args.template_command == "create":
                written = create_template_project(args.template, args.output)
                print(render_template_create_summary(args.template, args.output, written))
            elif args.template_command == "validate":
                print(render_template_validation(validate_template_project(args.path)))
            elif args.template_command == "smoke-test":
                print(render_template_smoke_test(smoke_test_template_project(args.path, timeout=args.timeout)))
            elif args.template_command == "compose":
                written = create_compose_project(args.template, args.output)
                print(render_compose_create_summary(args.template, args.output, written))
        except (KeyError, OSError) as exc:
            print("inferdoctor: {0}".format(exc), file=sys.stderr)
            return 2
        return 0

    if args.command in ("scenario", "scenarios"):
        results, _ = _results_for_target(
            None,
            getattr(args, "config", None),
            getattr(args, "timeout", None),
            None,
        )
        print(render_scenarios(evaluate_scenarios(results, args.target)))
        return _exit_code(results)

    if args.command == "profile":
        results, config = _results_for_target(
            None,
            getattr(args, "config", None),
            getattr(args, "timeout", None),
            None,
        )
        rendered = (
            render_profile_json(results, config)
            if args.format == "json"
            else render_profile_markdown(results, config)
        )
        if args.output:
            try:
                Path(args.output).write_text(rendered + "\n", encoding="utf-8")
            except OSError as exc:
                print(
                    "inferdoctor: could not write profile to '{0}': {1}. "
                    "Check that the parent directory exists and is writable.".format(
                        args.output, exc
                    ),
                    file=sys.stderr,
                )
                return 2
        else:
            print(rendered)
        return _exit_code(results)

    if args.command == "check":
        language = getattr(args, "language", None)
        results, config = (
            _results_for_target(
                getattr(args, "target", None),
                getattr(args, "config", None),
                getattr(args, "timeout", None),
                getattr(args, "endpoint", None),
                language,
            )
            if language is not None
            else _results_for_target(
                getattr(args, "target", None),
                getattr(args, "config", None),
                getattr(args, "timeout", None),
                getattr(args, "endpoint", None),
            )
        )
        print(render_dashboard(results, config, verbose=args.verbose, language=config.language))
        return _exit_code(results)

    results, config = _results_for_target(
        None,
        getattr(args, "config", None),
        getattr(args, "timeout", None),
        None,
    )
    rendered = (
        render_json(results)
        if args.format == "json"
        else render_markdown(results, verbose=args.verbose)
    )
    if args.output:
        try:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(
                "inferdoctor: could not write report to '{0}': {1}. "
                "Check that the parent directory exists and is writable.".format(
                    args.output, exc
                ),
                file=sys.stderr,
            )
            return 2
    else:
        print(rendered)
    return _exit_code(results)
