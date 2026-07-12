# InferDoctor

[日本語クイックスタート](https://github.com/anguoyang/inferdoctor/blob/main/README.ja.md)

[![Tests](https://github.com/anguoyang/inferdoctor/actions/workflows/tests.yml/badge.svg)](https://github.com/anguoyang/inferdoctor/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**InferDoctor helps developers diagnose, build, measure, compare, and optimize local or self-hosted AI apps.**

Diagnose what is broken. Choose a reasonable stack. Generate a starter app. Validate it. Measure responsiveness. Save a baseline. Compare after a change. Then verify whether the user experience actually improved.

Local AI setup often fails for unclear reasons: ports, CUDA, drivers, runtimes,
OpenAI-compatible endpoints, model size, app scaffolding, streaming behavior, and
RAG latency all interact. InferDoctor gives you one command for stack health, then
a beginner-friendly path from diagnosis to a small generated app and a bounded
performance feedback loop.

Use it to:

- find why Ollama, vLLM, SGLang, Xinference, Dify, CUDA, NVIDIA, Docker, or local endpoints are not working;
- estimate what your machine can realistically run with clear heuristic caveats;
- choose a practical local AI stack for a goal such as customer service, document Q&A, or a local API;
- generate, validate, and smoke-test starter projects without contacting a model endpoint;
- measure local, LAN, or private endpoint responsiveness with bounded smoke tests;
- save sanitized performance baselines and compare before/after changes;
- get practical TTFT, streaming, TPS, cold/warm, endpoint, and RAG UX optimization advice.

It is lightweight and read-only by default. It does not install AI runtimes,
download models, run inference, publish data, or modify system settings.

## Install

InferDoctor is available on PyPI:

```bash
pip install inferdoctor
```

Alternative developer install from GitHub:

```bash
python -m pip install "git+https://github.com/anguoyang/inferdoctor.git@dev"
```

For local development:

```bash
git clone https://github.com/anguoyang/inferdoctor.git
cd inferdoctor
python -m pip install -e ".[dev]"
```

## Quick Start

```bash
inferdoctor
inferdoctor quickstart customer-service
inferdoctor template list
inferdoctor stack plan --goal customer-service --vram 24
inferdoctor template create customer-service --output ./customer-service-demo
inferdoctor template validate ./customer-service-demo
inferdoctor template smoke-test ./customer-service-demo
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --runs 2 --warmup 1 --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --report before.json --goal customer-service
```

Starter workflows also include optional file generation commands:

```bash
inferdoctor template compose customer-service --output ./compose-customer-service
inferdoctor stack bootstrap --goal customer-service --output ./customer-service-bootstrap
inferdoctor template registry
```

These commands generate local files only. They do not pull Docker images, start containers, install runtimes, call endpoints, download models, or run inference.

## Closed-Loop Performance UX

Reachable endpoints are not enough. Local AI apps also need acceptable user experience: first token should appear quickly, streaming should work when enabled, RAG retrieval should show progress, and demos should avoid cold-start surprises.

```bash
inferdoctor perf endpoint --endpoint http://127.0.0.1:8000/v1 --model local-model
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --runs 2 --warmup 1 --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
# make one runtime, model, prompt, streaming, or RAG change
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output after.json
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --baseline before.json --candidate after.json --goal customer-service
inferdoctor optimize endpoint --runtime vllm --vram 24 --model-size 14b --streaming
inferdoctor optimize rag --top-k 8 --ttft 2.5 --streaming
```

These are smoke tests and heuristic suggestions, not formal benchmarks. InferDoctor does not download models, start runtimes, run long load tests, evaluate model quality, or modify system settings.

For LAN or private endpoints you control, live `inferdoctor perf` smoke tests require explicit `--allow-non-local`. InferDoctor redacts endpoint credentials in saved reports and baselines.

## Two Main Tracks

Application track:

```bash
inferdoctor quickstart customer-service
inferdoctor template create customer-service --output ./customer-service-demo
inferdoctor template validate ./customer-service-demo
inferdoctor template smoke-test ./customer-service-demo
```

Performance track:

```bash
inferdoctor perf baseline create --report before.json --name before
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --report after.json
```

## Language Support

InferDoctor v0.5 starts with first-step localization for the health dashboard and `inferdoctor check` console summary.

```bash
inferdoctor --language zh
inferdoctor --language ja
inferdoctor check --language en
```

Supported values are `auto`, `en`, `zh`, and `ja`. `auto` follows the local environment when it can be detected. Other commands, generated templates, JSON schemas, Markdown reports, and structured field names may remain English in this first i18n release so scripts and issue reports stay stable. Unsupported language values are rejected instead of silently falling back.

See [Internationalization](https://github.com/anguoyang/inferdoctor/blob/main/docs/i18n.md) for current scope and contribution notes.


Model recommendation tools help you choose a model. InferDoctor helps you
understand why your local AI stack is broken and what a practical next setup
step could look like.

## Example Output

```text
InferDoctor - Local AI Stack Health Check
=========================================================
Overall Health: 88 / 100  (Mostly healthy)
Stack Summary: 3 working | 1 needs attention | 4 optional/offline | 0 failed
Doctor's read: Some components need attention. Start with the first fix below.
PASS 100 | WARN 60 | FAIL 0 | SKIP 85  (heuristic)

Component   Status   Summary
----------- -------- --------------------------------------------------
System      PASS     System information collected
NVIDIA      PASS     1 NVIDIA GPU(s) detected
CUDA        SKIP     CUDA compiler was not found
Ollama      PASS     Ollama CLI and API are available
vLLM        WARN     vLLM models route returned HTTP 404
SGLang      SKIP     SGLang endpoint is not reachable
Xinference  SKIP     Xinference endpoint is not reachable
Dify        SKIP     Dify endpoint is not reachable

Top recommended fixes (most useful first):
1. vLLM: vLLM models route returned HTTP 404
   Likely cause: The service responded, but /v1/models was not found. The base URL may be missing or duplicating /v1.
   Impact: Needs attention if your app depends on vLLM.
   Try: inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1
   Config: endpoints.vllm: http://127.0.0.1:8000/v1
```

More screenshot-friendly samples:

- [`examples/console_cpu_only.txt`](https://github.com/anguoyang/inferdoctor/blob/main/examples/console_cpu_only.txt)
- [`examples/console_with_ollama.txt`](https://github.com/anguoyang/inferdoctor/blob/main/examples/console_with_ollama.txt)
- [`examples/console_with_gpu.txt`](https://github.com/anguoyang/inferdoctor/blob/main/examples/console_with_gpu.txt)
- [`examples/demo_health_dashboard.txt`](https://github.com/anguoyang/inferdoctor/blob/main/examples/demo_health_dashboard.txt)
- [`examples/demo_scenarios.txt`](https://github.com/anguoyang/inferdoctor/blob/main/examples/demo_scenarios.txt)
- [`examples/demo_profile.md`](https://github.com/anguoyang/inferdoctor/blob/main/examples/demo_profile.md)
- [`examples/demo_outputs/`](https://github.com/anguoyang/inferdoctor/tree/main/examples/demo_outputs) - v0.4 health, capacity, recommendation, stack plan, bootstrap, validation, smoke-test, and model-fit demos

Beginner setup docs and template examples:

- [`docs/beginner_guide.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/beginner_guide.md)
- [`docs/local_ai_stacks.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/local_ai_stacks.md)
- [`docs/openai_compatible_endpoints.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/openai_compatible_endpoints.md)
- [`docs/template_projects.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/template_projects.md)
- [`docs/template_registry.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/template_registry.md)
- [`docs/hardware_and_model_fit.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/hardware_and_model_fit.md)
- [`examples/templates/`](https://github.com/anguoyang/inferdoctor/tree/main/examples/templates)
- [`docs/performance/local_ai_user_experience.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/local_ai_user_experience.md)
- [`docs/performance/ttft_tps_streaming.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/ttft_tps_streaming.md)
- [`docs/performance/rag_latency.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/rag_latency.md)
- [`docs/performance/demo_readiness.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/demo_readiness.md)
- [`docs/performance/metric_definitions.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/metric_definitions.md)
- [`docs/performance/performance_reports.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/performance_reports.md)
- [`examples/reference_apps/customer_service/`](https://github.com/anguoyang/inferdoctor/tree/main/examples/reference_apps/customer_service)
- [`examples/reference_apps/local_doc_qa/`](https://github.com/anguoyang/inferdoctor/tree/main/examples/reference_apps/local_doc_qa)
- [`docs/performance/customer_experience_checklist.md`](https://github.com/anguoyang/inferdoctor/blob/main/docs/performance/customer_experience_checklist.md)

## From Broken Stack to Working App

InferDoctor starts with diagnosis, then guides the next step and helps you improve the early user experience. The project is
evolving toward hardware-aware stack recommendations, app templates, generated
configuration, and dry-run bootstrap plans that help beginners move from a
failing local setup to a small working local AI application.

The default behavior remains lightweight and read-only. Setup guidance and
template generation should be explicit user actions, not hidden side effects.

## Diagnose, Don't Guess

Use the commands that match the question you have:

```bash
inferdoctor                              # health score, component table, top fixes
inferdoctor explain openai-compatible-404
inferdoctor capacity --vram 24
inferdoctor profile --format markdown
inferdoctor report --format markdown
```

InferDoctor does not choose models for you. It tells you whether the local stack
that should serve those models is actually healthy.

## Why InferDoctor?

A local AI request can fail because of the operating system, driver, CUDA
toolkit, runtime process, endpoint URL, authentication, reverse proxy, or API
response format. These failures often look identical from the application.

InferDoctor checks each layer separately and turns low-level symptoms into
short, practical next steps.

## One Command, One Answer

Running `inferdoctor` is the same as running `inferdoctor check`:

```bash
inferdoctor
```

The dashboard immediately shows:

- an overall health score;
- which components pass, warn, fail, or are optional and skipped;
- a short explanation for every component;
- the top three recommended fixes;
- exact commands and configuration hints to try next.

The score is a transparent heuristic, not a benchmark. `PASS` contributes 100,
`WARN` 60, `FAIL` 0, and optional `SKIP` results contribute 85 so an unused
runtime does not heavily penalize an otherwise healthy machine.

## What Problems Does InferDoctor Diagnose?

| Component | What InferDoctor checks |
| --- | --- |
| System | OS, Python version, CPU architecture, available memory |
| NVIDIA | `nvidia-smi`, driver version, GPU name, total VRAM |
| CUDA | `nvcc`, toolkit version, CUDA environment variables |
| Ollama | CLI discovery and `/api/tags` connectivity |
| vLLM | OpenAI-compatible `/v1/models` connectivity and response shape |
| SGLang | OpenAI-compatible `/v1/models` connectivity and response shape |
| llama.cpp server | OpenAI-compatible `/v1/models` connectivity when server mode is enabled |
| LM Studio | OpenAI-compatible `/v1/models` connectivity |
| Xinference | Supervisor endpoint connectivity without the SDK |
| Dify | Configurable Dify endpoint connectivity without the SDK |
| Open WebUI | Lightweight web endpoint reachability |
| Docker | Docker CLI discovery and daemon reachability without starting containers |

OpenAI-compatible checks distinguish connection refusal, timeout, unauthorized
responses, wrong base URLs, invalid JSON, and non-compatible response shapes.

## Top Fixes

For the highest-priority problems, InferDoctor shows:

- the observed issue;
- the likely cause;
- whether the issue is optional or likely blocking;
- the next command to run;
- the relevant `inferdoctor.yaml` setting.

Example:

```text
1. SGLang: SGLang models route returned HTTP 404
   Likely cause: The service responded, but /v1/models was not found. The base URL may be missing or duplicating /v1.
   Impact: Needs attention if your app depends on SGLang.
   Try: inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1
   Config: endpoints.sglang: http://127.0.0.1:30000/v1
```

## Command Reference Quick Start

InferDoctor requires Python 3.9 or newer. Install the published package and run the health check:

```bash
pip install inferdoctor
inferdoctor
```

For development from source, use the GitHub install path shown above.

Check one component or override its endpoint:

```bash
inferdoctor check nvidia
inferdoctor check ollama
inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1
inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1
inferdoctor check llamacpp --endpoint http://127.0.0.1:8080
inferdoctor check lmstudio --endpoint http://127.0.0.1:1234/v1
inferdoctor check openwebui
inferdoctor check docker
inferdoctor check xinference
inferdoctor check dify
```

Show detailed raw diagnostic data or allow more time for remote endpoints:

```bash
inferdoctor check --verbose
inferdoctor check vllm --timeout 5
```

## Scenario Readiness

Use `inferdoctor scenario` when you want the answer framed around a goal instead
of a component:

```bash
inferdoctor scenario
inferdoctor scenario openai-compatible-server
```

Initial scenarios include local chatbot, RAG app, OpenAI-compatible server,
Dify local RAG, GPU inference, and CPU-only fallback. See
[`examples/scenario_readiness.txt`](https://github.com/anguoyang/inferdoctor/blob/main/examples/scenario_readiness.txt).

## Troubleshooting Explain

Use `inferdoctor explain` when you want a short, focused explanation for a
common local AI failure:

```bash
inferdoctor explain openai-compatible-404
inferdoctor explain cuda-toolkit-missing
inferdoctor explain vllm-endpoint-not-reachable
```

Each explanation includes what the symptom means, common causes, what to try
next, and the related InferDoctor command.

## Configuration

```yaml
endpoints:
  ollama: http://127.0.0.1:11434
  vllm: http://127.0.0.1:8000/v1
  sglang: http://127.0.0.1:30000/v1
  llamacpp: http://127.0.0.1:8080
  lmstudio: http://127.0.0.1:1234/v1
  xinference: http://127.0.0.1:9997
  dify: http://127.0.0.1:5001
  openwebui: http://127.0.0.1:3000
timeout: 2
```

```bash
inferdoctor check --config inferdoctor.yaml
```

The built-in YAML reader intentionally supports this small mapping format so
the runtime remains dependency-free. `--timeout` and `--endpoint` provide safe,
temporary overrides without editing configuration.




## Stack Recommendations

Use `inferdoctor recommend` for a lightweight recommendation that connects your
goal, preference, hardware, runtime path, model size class, and starter
template:

```bash
inferdoctor recommend
inferdoctor recommend --goal customer-service --vram 24
inferdoctor recommend --goal document-qa --preference easiest
inferdoctor recommend --goal local-api --preference performance --vram 24
```

Recommendations are heuristics, not benchmarks. InferDoctor does not rank
specific model names, download models, or start runtimes for you.

## Guided Init

Use `inferdoctor init` when you know what you want to build but not which local
AI stack to start with:

```bash
inferdoctor init
inferdoctor init --goal customer-service --preference easiest
inferdoctor init --goal document-qa --preference gpu
```

The command recommends a runtime path, template, diagnosis command, and template
creation command. It does not install runtimes, download models, or modify
system settings.

## Template Catalog

InferDoctor is growing from diagnosis into guided local AI setup. Start by
inspecting lightweight template metadata:

```bash
inferdoctor template list
inferdoctor template show customer-service
inferdoctor template show restaurant-ordering
inferdoctor template create customer-service --output ./customer-service-demo
inferdoctor template create local-doc-qa --output ./local-doc-qa-demo
```

The catalog describes the intended app, required stack, optional stack, hardware
fit, difficulty, planned generated files, and next command. Listing or showing a template does
not install runtimes, download models, or create files. Creating a template
writes only to the explicit `--output` directory.


## Model Fit Advisor

Use `inferdoctor model fit` when you want a direct memory-fit estimate for a
model size, quantization, runtime, and VRAM value:

```bash
inferdoctor model fit --size 14b --quant q4 --vram 24
inferdoctor model fit --size 32b --quant q4 --vram 24
inferdoctor model fit --size 14b --quant q4 --runtime ollama
```

Model fit estimates are rough heuristics, not benchmarks. Context length, KV
cache, runtime options, CPU offload, batch size, and model architecture can move
real memory usage significantly.

## Capacity Preview

Use `inferdoctor capacity` for a lightweight hardware readiness estimate:

```bash
inferdoctor capacity
inferdoctor capacity --gpu "RTX 3090"
inferdoctor capacity --vram 24 --model-size 14b --quant q4
inferdoctor capacity --runtime vllm --model-size 32b
```

Capacity estimates are rough heuristics, not benchmarks. InferDoctor can infer
common VRAM sizes from names such as RTX 4090, RTX 3090, or RTX 3060 12GB, but
it does not download models, run inference, or recommend a model leaderboard.

## Reports

```bash
inferdoctor report --format json
inferdoctor report --format markdown
inferdoctor report --format json --output report.json
inferdoctor report --format markdown --output report.md
```

See the sanitized samples in
[`examples/report_cpu_only.json`](https://github.com/anguoyang/inferdoctor/blob/main/examples/report_cpu_only.json) and
[`examples/report_cpu_only.md`](https://github.com/anguoyang/inferdoctor/blob/main/examples/report_cpu_only.md).

## Safe Diagnostic Profile

Use `inferdoctor profile` when you need to share environment details in an issue,
chat, or support thread without leaking secrets by default:

```bash
inferdoctor profile --format markdown
inferdoctor profile --format json
inferdoctor profile --format markdown --output inferdoctor-profile.md
```

The profile includes OS, Python version, CPU architecture, RAM summary, detected
GPU names and VRAM, command availability, configured endpoints, checker summary,
and Top Fixes. It redacts suspicious keys such as tokens, passwords, API keys,
authorization values, endpoint credentials, query strings, and home-directory
paths.

```markdown
# InferDoctor Safe Diagnostic Profile

## Checker Summary

| Check | Status | Summary |
| --- | --- | --- |
| system | **PASS** | System information collected |
| vllm | **WARN** | vLLM models route returned HTTP 404 |

## Top Fixes

1. **vLLM**: vLLM models route returned HTTP 404
   - Try: `inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1`
```

## InferDoctor vs Model Recommendation Tools

| Question | Model recommendation tools | InferDoctor |
| --- | --- | --- |
| Which model should I use? | Yes | No |
| Why is my local endpoint failing? | Usually no | Yes |
| Is my NVIDIA driver visible? | Sometimes | Yes |
| Is `/v1/models` valid and compatible? | Usually no | Yes |
| Does it download or run models? | Sometimes | Never |

InferDoctor is not a model recommender. It does not choose, download, or run
models for you. It diagnoses the stack you already operate.

## What InferDoctor Does Not Do

- It does not install AI runtimes, drivers, CUDA packages, or models.
- It does not run inference or load model weights.
- It does not modify system settings or services.
- It does not require a GPU.
- It is read-only by default.

## Status Meaning

- `PASS`: the component responded as expected.
- `WARN`: the component is reachable or present, but needs attention.
- `FAIL`: a discovered component is broken or a diagnostic cannot complete.
- `SKIP`: an optional component is absent or offline; this is not automatically
  a system failure.

Only `FAIL` causes a non-zero diagnostic exit code.

## Demo Outputs

Generate safe demo artifacts for screenshots, README updates, launch posts, or
social sharing:

```bash
bash scripts/generate_demo_outputs.sh
```

The generated files live under `examples/` and use fixed sample values instead
of probing the current machine.


## More Guides

- [Getting started](https://github.com/anguoyang/inferdoctor/blob/main/docs/getting_started.md)
- [Templates](https://github.com/anguoyang/inferdoctor/blob/main/docs/templates.md)
- [Stack recommendations](https://github.com/anguoyang/inferdoctor/blob/main/docs/recommendations.md)
- [Model fit advisor](https://github.com/anguoyang/inferdoctor/blob/main/docs/model_fit.md)
- [PyPI release workflow](https://github.com/anguoyang/inferdoctor/blob/main/docs/pypi_release.md)

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Tests use subprocess and HTTP mocks. They require no GPU, CUDA installation,
local inference runtime, or internet access.

See [CONTRIBUTING.md](https://github.com/anguoyang/inferdoctor/blob/main/CONTRIBUTING.md) and [SECURITY.md](https://github.com/anguoyang/inferdoctor/blob/main/SECURITY.md).

## Roadmap

- guided local AI setup with `inferdoctor init`
- hardware-aware stack recommendations
- practical app templates and a template generator
- dry-run bootstrap scripts for explicit setup flows
- model and runtime fit advisor heuristics
- richer capacity heuristics for more hardware profiles
- deeper llama.cpp build and backend diagnostics
- ONNX Runtime provider diagnostics
- TensorRT library and version diagnostics
- RKNN and other edge accelerator checks
- optional local container and service discovery
- stable third-party checker entry points
- richer report summaries without adding heavy runtime dependencies

## License

Licensed under the Apache License 2.0. See [LICENSE](https://github.com/anguoyang/inferdoctor/blob/main/LICENSE).
