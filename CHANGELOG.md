# Changelog

## Unreleased

- No unreleased changes.

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


