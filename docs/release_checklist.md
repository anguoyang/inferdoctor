# v0.7.0 Release Candidate Checklist

This checklist prepares InferDoctor v0.7.0 for a human-controlled `dev` to `main` release. It intentionally excludes private credentials, live provider evidence, request IDs, and publication actions.

## Release Scope

- [x] Dify application diagnostics and safe orchestration tracing are documented.
- [x] RAG Intelligence, cognitive-path diagnosis, and First Broken Layer attribution are documented.
- [x] Controlled Cognitive Replay, Gold Context probes, and the Minimal Next Probe Planner are included.
- [x] OpenInference / OTLP trace adaptation is included.
- [x] Provider metadata reuses the shared OpenAI-compatible transport.
- [x] OrcaRouter remains a provider preset rather than an architectural dependency.
- [x] Provider Check is bounded and does not include Provider Compare, pricing data, routing, or recommendation ranking.

## Security and Evidence Boundaries

- [x] API keys and optional response content are not rendered or retained.
- [x] Malformed API keys fail before a request is sent.
- [x] Credentials embedded in provider endpoint URLs are rejected.
- [x] Public endpoints require `--allow-public` and private/LAN endpoints require `--allow-non-local`.
- [x] Model catalog presence does not prove model invocation access.
- [x] HTTP 401 authentication failure and HTTP 403 model-access denial remain distinct.
- [x] Unsupported probes and missing evidence remain `UNKNOWN`.
- [x] Partner metadata cannot influence diagnostic results or recommendations.
- [x] No live external provider call is required by automated validation.

## Validation

- [x] Python source compilation passes.
- [x] Full `pytest` suite passes.
- [x] Existing README/PyPI link validation passes.
- [x] `git diff --check` passes.
- [x] Package build succeeds with the existing `python -m build` workflow.
- [x] Existing `twine check` package metadata validation passes.
- [x] Built wheel installs and passes CLI/version smoke tests in a clean temporary environment.
- [x] Distribution artifacts contain no obvious credentials or private release artifacts.

## Release Preparation

- [x] Canonical package and module versions are synchronized at `0.7.0`.
- [x] `CHANGELOG.md` summarizes the v0.7 milestone and evidence boundaries.
- [x] Public release notes exist at `docs/releases/v0.7.0.md`.
- [x] README links use the existing absolute-link convention for GitHub and PyPI rendering.
- [ ] Confirm GitHub Actions passes on the final `dev` commit.
- [ ] Review the final `dev` to `main` diff.
- [ ] Merge `dev` to `main` only after release approval.
- [ ] Create and push tag `v0.7.0` only after the release merge.
- [ ] Create the GitHub release from `docs/releases/v0.7.0.md`.
- [ ] Publish to PyPI only after explicit human approval and credential availability.
