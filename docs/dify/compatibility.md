# Dify Integration Compatibility

InferDoctor v0.7 adds a Dify-first workflow for validating, smoke-testing, measuring, and optimizing Dify applications connected to local, LAN, private, self-hosted, or explicitly approved public endpoints.

This integration uses published Dify APIs only. It does not depend on Dify database tables, Redis queues, Celery internals, Studio browser requests, Docker volumes, or undocumented admin APIs.

## Supported Integration Level

InferDoctor supports:

- Dify application API preflight checks with `GET /info`
- Dify Chatflow, Chatbot, Agent, New Agent, and Workflow app smoke-test routing where the published API mode is supported
- Dify Server-Sent Events parsing for application streaming responses
- Local / Private RAG Starter Kit export
- Offline kit and DSL validation
- Dify performance smoke reports
- Baseline, comparison, and optimization guidance for Dify performance reports
- Read-only knowledge retrieval checks when the user supplies a knowledge API key and dataset ID

InferDoctor does not:

- Install Dify
- Start or stop Dify containers
- Import DSLs automatically
- Create, update, or delete applications
- Create, update, or delete knowledge bases
- Upload documents
- Modify model-provider configuration
- Store API keys
- Treat smoke-test results as formal benchmarks

## Official API Assumptions

InferDoctor follows the current public Dify documentation for:

- Application API `GET /info`
- Chat message API `POST /chat-messages`
- Workflow run API `POST /workflows/run`
- Dify SSE streaming responses
- Knowledge retrieval/test API `POST /datasets/{dataset_id}/retrieve`

Supported app mode names are based on the official `/info` response documentation:

- `chat`
- `advanced-chat`
- `agent-chat`
- `agent`
- `workflow`
- `completion`

`completion` apps are recognized, but the initial live smoke-test path focuses on chat-like and workflow apps.

Compatibility can vary by Dify version, deployment, and app design. InferDoctor prefers capability detection and explicit warnings over strict version gates.

## Credentials

Dify application API keys and knowledge API keys are separate credentials.

Use:

- `DIFY_API_BASE_URL`
- `DIFY_APP_API_KEY`
- `DIFY_KNOWLEDGE_API_BASE_URL`
- `DIFY_KNOWLEDGE_API_KEY`
- `DIFY_DATASET_ID`

Do not pass raw API keys on the command line. Command-line values can remain in shell history.

InferDoctor redacts:

- Authorization headers
- URL userinfo
- secret-like query parameters
- API-key-like values in errors and reports

## Endpoint Safety

Localhost endpoints may be smoke-tested directly with harmless built-in prompts.

LAN/private endpoints require `--allow-non-local`.

Public endpoints require `--allow-public`.

InferDoctor never scans for Dify, never follows a live endpoint inferred from a DSL file, and never sends private documents in smoke tests.


## Verified Import Compatibility Profile

The current `local-private-rag` kit uses a v2 Dify Chatflow-style DSL structure with top-level `app`, `dependencies`, `kind`, `version`, and a React Flow-style `workflow.graph`.

This structure has been manually imported into a current Dify Cloud workspace. The import completed, the Chatflow canvas rendered, and the four-node canary flow loaded:

- Start
- Knowledge Retrieval
- LLM
- Answer

The model provider and knowledge-base references intentionally remain unresolved after import. Users must bind their own model provider, model, and knowledge base inside Dify before publishing or testing a live app.

This verification applies to the tested current Dify Cloud DSL structure. It does not guarantee compatibility with every Dify Cloud, Community, or self-hosted release. InferDoctor still reports offline validation separately from live import verification, and generated future DSLs must not claim live import verification unless they are verified.

## Validation Levels

The Dify kit workflow uses explicit validation levels:

- `package-structure validated`
- `DSL syntax/static compatibility checked`
- `live application API verified`
- `live import verified`

InferDoctor does not claim `live import verified` unless a user has actually imported the DSL into a Dify workspace and verified it.

## References

- Dify API Get Started: https://docs.dify.ai/en/api-reference/guides/get-started
- Dify Get App Info: https://docs.dify.ai/en/api-reference/applications/get-app-info
- Dify Chat Messages: https://docs.dify.ai/en/api-reference/chat-messages/send-chat-message
- Dify Workflow Runs: https://docs.dify.ai/en/api-reference/workflow-runs/run-workflow
- Dify SSE Streaming: https://docs.dify.ai/en/api-reference/guides/streaming
- Dify Knowledge Retrieval: https://docs.dify.ai/en/api-reference/knowledge-bases/retrieve-chunks-from-a-knowledge-base-test-retrieval
- Official Dify repository: https://github.com/langgenius/dify
