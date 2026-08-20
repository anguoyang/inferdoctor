# Provider Foundation and Provider Check

InferDoctor treats a provider as metadata plus a protocol preset. Providers do not get separate SDKs or transports. Every OpenAI-compatible preset uses the same bounded standard-library transport that is also used by the RAG Gold Context probe.

Provider metadata can describe:

- an identifier and display name;
- protocol and API base URL;
- documentation and signup URLs;
- an optional partner URL;
- the API-key environment variable;
- an optional default model;
- advertised capabilities.

Partner metadata is informational only. It never participates in diagnosis, quality scoring, ranking, provider comparison, or recommendations.

## Commands

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

## OrcaRouter preset

The built-in OrcaRouter preset uses facts documented by OrcaRouter:

- API base: `https://api.orcarouter.ai/v1`
- protocol: OpenAI-compatible
- API key environment variable: `ORCAROUTER_API_KEY`
- default routing model: `orcarouter/auto`
- partner URL: `None`

Official references:

- [OrcaRouter quickstart](https://docs.orcarouter.ai/getting-started/quickstart)
- [API key and Bearer authentication](https://docs.orcarouter.ai/getting-started/get-api-key)
- [OpenAI-compatible HTTP](https://docs.orcarouter.ai/native-formats/openai-compat)
- [`orcarouter/auto`](https://docs.orcarouter.ai/routing/auto-router)

No referral URL is invented. A future partner URL can be changed in metadata without changing request or diagnostic logic.

## Safety and evidence semantics

Public endpoints require explicit `--allow-public`. LAN/private presets require `--allow-non-local`. URLs containing credentials are rejected. API keys are read from the preset's environment variable, sent only as a Bearer header, and never included in reports.

Evidence remains conservative:

- unsupported `/models` means model availability is `UNKNOWN`, not `FAIL`;
- a model listed by `/models` proves catalog presence, not key-specific invocation access; actual model access remains `UNKNOWN` until invocation succeeds;
- TTFT is `null` / `UNKNOWN` because Provider Check does not perform a streaming TTFT measurement;
- pricing and API cost are `null` / `UNKNOWN` without direct evidence;
- total compute cost is always `null` / `UNKNOWN` in this MVP;
- the optional chat response body is evaluated in memory for protocol success and is not retained.

Tests use mocked HTTP responses and do not require external services or API keys.


A successful `/models` request establishes connectivity and authentication but does not prove that the same key can invoke every listed model.

During an explicit smoke request:

- HTTP 401 is authentication failure;
- HTTP 403 is permission/model-access failure;
- HTTP 402 is billing/credit failure;
- HTTP 429 is quota/rate-limit failure.

Later evidence must not incorrectly overwrite an already established authentication result with an unrelated permission failure.
