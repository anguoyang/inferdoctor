# Changelog

## Unreleased

## v0.8.0

### Highlights

- Added evidence sufficiency classification so missing, redacted, or insufficient evidence is not silently converted into a confident diagnosis.
- Added structured Minimal Next Probe guidance for the smallest useful evidence-gathering or controlled experiment.
- Added provider-neutral Provider Compare for bounded same-workload checks across OpenAI-compatible inference targets.
- Added `inferdoctor rag gate` for CI-friendly before/after RAG regression gating with PASS / BLOCKED / INCONCLUSIVE semantics and exit codes 0 / 1 / 2.
- Added a Quality Gate comparison policy so model, provider, and pipeline implementation changes can be evaluated without treating ordinary latency jitter as a quality regression.
- Repositioned the GitHub and PyPI first screen around evidence-driven diagnosis, First Broken Layer, Minimal Next Probe, Fix Verification, and repeated regression gating.
- Updated the Japanese first screen and added the lightweight animated Quality Gate demo.

### Evidence and Safety Boundaries

- Missing or redacted evidence remains UNKNOWN / INCONCLUSIVE.
- First Broken Layer is established only when supported by available evidence.
- Provider Compare produces bounded evidence, not a model benchmark or leaderboard.
- Observed latency samples are not benchmark claims.
- Provider and partner metadata does not influence diagnosis, comparison, ranking, or recommendation logic.
- Live public, LAN, or private endpoint access continues to require explicit opt-in.
- Provider Compare does not intentionally retain API keys or raw response content.
- Quality Gate PASS is scoped to the evaluated Cases and available evidence; it is not a universal guarantee that an application has no bugs.

## v0.7.1

### Documentation and Metadata Hotfix

- Made OrcaRouter visible in the PyPI summary and README first screen as InferDoctor's first built-in hosted-provider preset.
- Added current RAG, agent, OpenInference, OpenTelemetry, OpenAI-compatible, and OrcaRouter package keywords.
- Clarified that OrcaRouter uses the provider-neutral diagnostic layer and does not affect diagnostic independence or recommendation logic.
- No diagnostic, provider-transport, security, or runtime behavior changed.

## v0.7.0


#### Highlights

- Added Dify application diagnostics, safe orchestration tracing, Local / Private RAG kit workflows, performance smoke reports, and read-only knowledge retrieval checks.
- Added framework-neutral RAG Intelligence with evidence normalization, layered diagnosis, comparison, and First Broken Layer attribution.
- Added cognitive-path diagnosis across intent, route, plan, action, retrieval, context, generation, and postprocessing.
- Added the Minimal Next Probe Planner, Controlled Cognitive Replay, and Gold Context capability-isolation probes.
- Added OpenInference / OTLP trace adaptation, including observable Agent-to-Tool plan evidence.
- Added provider metadata and one shared OpenAI-compatible transport, with OrcaRouter as the first provider preset.
- Added Provider Check for bounded connectivity, authentication, model-catalog, and optional model-invocation evidence.

#### Safety and Evidence Boundaries

- Dify integration uses published APIs and remains read-only by default; it does not install Dify, start containers, import DSLs automatically, create knowledge bases, or upload documents.
- API keys and optional chat response content are never retained or rendered in Provider Check results.
- Public endpoints require explicit `--allow-public`; LAN/private endpoints require explicit `--allow-non-local`; URL credentials are rejected.
- A model catalog entry does not prove invocation access, and HTTP 401 authentication failures remain distinct from HTTP 403 model-access denials.
- Unsupported probes and missing evidence remain `UNKNOWN`, including TTFT, pricing, and total compute cost.
- Partner metadata cannot influence diagnosis, scores, comparisons, rankings, or recommendations.
- Live checks and performance reports are bounded smoke tests, not formal benchmarks.

## v0.6.0

### Highlights

- Added closed-loop local AI app optimization workflow: diagnose, plan, build, validate, measure, compare, optimize, and verify improvement.
- Added reusable sanitized performance baselines with create, show, list, and delete commands.
- Added before-and-after performance comparison for TTFT, total latency, generation duration, TPS, success rate, streaming state, and readiness category.
- Added evidence-based optimization plans with observations, evidence level, verification commands, expected impact category, and limitations.
- Added application experience profiles for interactive chat, customer service, restaurant ordering, document Q&A, RAG, local APIs, batch processing, and internal prototypes.
- Improved guided quickstart output with validation, baseline, comparison, and optimization-plan commands.
- Added safer local, LAN, and private endpoint workflow guidance with explicit non-local opt-in.
- Added performance verification guidance to principal starter templates.
- Added customer-service and local document Q&A reference apps.

### Safety

- Performance commands remain bounded smoke tests, not formal benchmarks.
- No model downloads, runtime installation, service startup, system modification, concurrency benchmark, or sustained load test was added.
- Public endpoints are not contacted automatically.
- Dify integration is not included in v0.6.

## v0.5.1

### Documentation Hotfix

- Fixed repository-relative links in the PyPI long description by converting README links to stable GitHub URLs.
- Fixed the Japanese quickstart link when rendered on PyPI.
- Updated Japanese installation instructions to use `pip install inferdoctor` as the primary path.
- Updated Japanese documentation for v0.5 performance UX and first-step i18n features.
- Added offline README link validation to prevent future PyPI long-description link regressions.

### Safety

- No runtime, API, metric-schema, or performance behavior changes.
- No model downloads, runtime installation, or inference execution features were added.

## v0.5.0

### Development Notes

- Performance UX smoke tests for OpenAI-compatible endpoints, including TTFT, streaming, bounded cold/warm runs, structured reports, and endpoint redaction.
- Endpoint and RAG optimization advice for local AI user experience.
- Streaming-first starter templates with safe dry-run and config-check paths.
- Docker Compose starter file generation for selected templates.
- Safe stack bootstrap file generation for starter projects and setup plans.
- Improved Dify, Open WebUI, and Ollama + Open WebUI starter guidance.
- Local template registry foundation with conservative safety rules.
- Project readiness scoring for template validation and smoke tests.
- More real-world template examples for Dify, Open WebUI, Compose, and bootstrap workflows.
- Added first-step i18n for the health dashboard and `inferdoctor check` console summary in English, Chinese, and Japanese. Full CLI localization remains future work, and machine-readable schemas keep stable English field names.

### Safety

- Compose generation writes files only; it does not pull images or start containers.
- Bootstrap generation writes starter files only; it does not install dependencies, call endpoints, or run inference.
- Template registry support is local-only; no remote template execution is enabled.
- Recommendations, performance readings, and readiness scores remain heuristics, not benchmarks.

## v0.4.1

### Highlights

- First PyPI release of InferDoctor.
- Published package name: `inferdoctor`.
- Default install command is now `pip install inferdoctor`.
- Verified the PyPI package in a clean environment with the health dashboard, template list, and stack plan commands.
- Fixed a terminal-width-sensitive local-doc-qa template help test before publication.

### Safety

- No model downloads, runtime installation, or inference execution were added.
- Diagnosis remains read-only by default.
- Template validation and smoke tests remain safe local checks.

## v0.4.0

### Highlights

- Beginner setup journeys polished across diagnosis, recommendation, stack planning, bootstrap dry-run, template creation, validation, and smoke testing.
- Generated starter apps now make dry-run and config-check paths clearer before any live model endpoint is used.
- `inferdoctor template smoke-test` support is emphasized in recommended next steps and generated project flow.
- Golden demo outputs added for health check, capacity, recommendations, stack plan, bootstrap dry-run, template validation, template smoke-test, and model fit.
- README first screen updated around the setup-assistant workflow.
- v0.4.0 release notes drafted for a future public release.

### Safety

- No heavy AI runtime dependencies were added.
- No model download or model execution commands were added.
- Template smoke tests remain read-only and do not call endpoints.
- Stack bootstrap remains dry-run guidance, not an installer.
- Recommendations and model-fit estimates remain heuristics, not benchmarks.

## v0.3.0

### Highlights

- Template catalog for common local AI app goals.
- Starter project generation for customer service, restaurant ordering, and local document Q&A demos.
- Guided `inferdoctor init` setup path.
- Hardware-aware `inferdoctor recommend` stack recommendations.
- Heuristic `inferdoctor model fit` advisor.
- Improved generated starter templates with local endpoint config and troubleshooting.
- Beginner documentation for getting started, templates, recommendations, model fit, and local AI concepts.
- Public v0.3.0 release readiness checks and install smoke test coverage.

### Safety

- No heavy AI runtime dependencies were added.
- No model download or model execution commands were added.
- Template generation writes only to the explicit output directory.
- Recommendations and model-fit estimates remain heuristics, not benchmarks.

## v0.2.0

### Highlights

- Screenshot-friendly default health dashboard for `inferdoctor`.
- Heuristic overall health score and stack summary.
- Top Fixes with likely cause, impact, next command, and config hint.
- Reusable OpenAI-compatible endpoint diagnostics for `/v1/models`.
- First-class SGLang checker.
- Improved vLLM, Xinference, Dify, CUDA, NVIDIA, and HTTP failure suggestions.
- `inferdoctor explain <topic>` troubleshooting guides for common local AI failures.
- `inferdoctor capacity` lightweight local AI hardware readiness preview.
- Updated README, console examples, launch post draft, and release checklist.

### Safety

- No heavy AI runtime dependencies were added.
- No model download or inference execution features were added.
- Diagnostics remain lightweight and read-only by default.
- Tests continue to use mocks and do not require GPU, CUDA, local runtimes, or internet access.


