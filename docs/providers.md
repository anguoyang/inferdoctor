# Provider Foundation, Provider Check, and Provider Compare

InferDoctor treats a provider preset as metadata and a comparison target as runtime configuration. Neither gets a provider-specific SDK or transport. Built-in presets, custom endpoints, local vLLM, Provider Check, Provider Compare, and the RAG Gold Context probe all use the same bounded standard-library OpenAI-compatible transport.

Provider preset metadata can describe:

- an identifier and display name;
- protocol and API base URL;
- documentation and signup URLs;
- an optional partner URL;
- the API-key environment variable;
- an optional default model;
- advertised capabilities.

A runtime target adds the selected model and whether an API key is required. Custom OpenAI-compatible targets can omit API-key configuration, which supports localhost vLLM and other endpoints that do not require authentication. No fake key is needed.

Partner metadata is informational and optional navigation metadata only. It never participates in diagnosis, deterministic quality evaluation, latency interpretation, First Broken Layer attribution, target ordering, provider comparison, ranking, scoring, or recommendations.

## Provider Check

```bash
inferdoctor provider list
inferdoctor provider show orcarouter
inferdoctor provider check --provider orcarouter --allow-public
```

The default check sends one authenticated, read-only `GET /v1/models` request. A chat request is not sent by default. To make a bounded, optional chat invocation:

```bash
inferdoctor provider check --provider orcarouter --allow-public --smoke
```

The smoke prompt is fixed and harmless, output is not retained, and the request uses a small output-token limit. Provider Check is not a benchmark or a quality evaluation.

## Same-workload Provider Compare

Provider Compare executes the same non-streaming messages, temperature, output-token bound, and deterministic RAG Case expectations against both targets. Only the endpoint, model, and configured authentication differ.

Example: OrcaRouter versus a no-key local vLLM endpoint:

```bash
export ORCAROUTER_API_KEY=your-orcarouter-key

inferdoctor provider compare \
  --provider orcarouter \
  --provider-model orcarouter/free \
  --custom-endpoint http://127.0.0.1:8000/v1 \
  --custom-model your-local-model \
  --custom-label "Local vLLM" \
  --case ./case.json \
  --allow-public \
  --format json \
  --output provider-compare.json
```

`case.json` uses the existing [`inferdoctor.rag.case.v1`](https://github.com/anguoyang/inferdoctor/blob/main/docs/rag/case_schema.md) schema. Provider Compare reuses:

- `question` as the shared user prompt;
- `required_facts` for deterministic required evidence;
- `forbidden_claims` for explicit prohibited assertions;
- the existing `all_terms`, `any_term`, `exact_phrase`, and `human_review` semantics.

This is deterministic term verification, not an LLM-as-a-judge evaluation. Human-review-only rules remain `UNKNOWN` rather than being guessed.

For a private/LAN custom endpoint, add `--allow-non-local`. For a custom endpoint that requires a key, name its environment variable without placing the key on the command line:

```bash
export LOCAL_OPENAI_API_KEY=your-local-key

inferdoctor provider compare \
  --provider orcarouter \
  --custom-endpoint http://192.168.1.20:8000/v1 \
  --custom-model your-local-model \
  --custom-api-key-env LOCAL_OPENAI_API_KEY \
  --case ./case.json \
  --allow-public \
  --allow-non-local
```

## Comparison Evidence

The canonical JSON result records transparent evidence for each target:

- redacted endpoint, endpoint category, selected model, and key presence only;
- `/models` HTTP and catalog evidence when available;
- same-workload invocation HTTP status and model-access evidence;
- usable generation evidence;
- one bounded observed total-latency sample;
- required facts found or missing;
- forbidden claims found;
- deterministic quality `PASS`, `FAIL`, or `UNKNOWN`;
- first failed observable layer;
- explicit limitations and unknown fields.

The raw answer is evaluated in memory. By default only its hash, character count, and derived deterministic evidence are retained. API keys are never retained.

The provider-specific observable path is deliberately narrower than InferDoctor's cognitive and RAG causal schemas:

```text
endpoint
  -> connectivity
  -> authentication
  -> billing / quota when observed
  -> model access
  -> generation
  -> deterministic verification
```

The first check with direct `FAIL` evidence becomes the target's `first_broken_layer`. `UNKNOWN` is not treated as a broken layer. Model catalog evidence is diagnostic context, not a causal prerequisite: catalog presence does not prove invocation access, and an unsupported catalog route does not prove unavailability.

The comparison produces no winner badge, provider ranking, composite score, routing decision, price comparison, or Doctor Recommendation. A lower observed latency in one request is not evidence that a provider is universally faster.

## OrcaRouter Preset

The built-in OrcaRouter preset uses:

- API base: `https://api.orcarouter.ai/v1`
- protocol: OpenAI-compatible
- API key environment variable: `ORCAROUTER_API_KEY`
- default routing model: `orcarouter/auto`
- an approved optional partner URL stored only in preset metadata

Official references:

- [OrcaRouter quickstart](https://docs.orcarouter.ai/getting-started/quickstart)
- [API key and Bearer authentication](https://docs.orcarouter.ai/getting-started/get-api-key)
- [OpenAI-compatible HTTP](https://docs.orcarouter.ai/native-formats/openai-compat)
- [`orcarouter/auto`](https://docs.orcarouter.ai/routing/auto-router)

The partner URL can be replaced through metadata without changing request, evaluation, comparison, attribution, or future recommendation logic. Provider Compare does not call or resolve it.

## Safety and Conservative Semantics

Localhost follows existing local rules. Public endpoints require explicit `--allow-public`; LAN/private endpoints require `--allow-non-local`; URLs containing credentials are rejected. Endpoints are redacted before output.

Evidence remains conservative:

- HTTP 401 is authentication failure;
- HTTP 403 is permission/model-access failure and does not overwrite already established authentication evidence;
- HTTP 402 is billing/credit failure;
- HTTP 429 is quota/rate-limit failure;
- unsupported `/models` remains `UNKNOWN`;
- a listed model proves catalog presence, not key-specific invocation access;
- successful invocation establishes model access for that request;
- TTFT is `null` / `UNKNOWN` because Provider Compare is non-streaming;
- pricing and API cost are `null` / `UNKNOWN` without evidence;
- local total compute cost is never described as zero and remains `null` / `UNKNOWN`;
- one observed total latency is a bounded sample, not a benchmark.

Automated tests use mocked HTTP responses and do not contact external providers or local inference services.
