# Local / Private RAG Starter Kit for Dify

`local-private-rag` is the first Dify starter kit in InferDoctor v0.7.

It is designed for teams that want Dify orchestration with a local, LAN, private, or self-hosted model endpoint.

## What It Contains

The exported kit includes:

- `manifest.yaml`
- `dify_app.yaml`
- `README.md`
- `.env.example`
- `preflight.yaml`
- `smoke_cases.yaml`
- `experience_profile.yaml`
- `performance_guidance.yaml`
- `optimization_notes.md`
- `sample_docs/return_policy.md`

The sample document is fictional.

## Intended Flow

```text
User question
-> knowledge retrieval
-> optional rerank guidance
-> context budget
-> LLM answer
-> streamed answer
-> no-answer fallback guidance
```

The DSL is a starter draft for manual review and import. Offline validation checks package structure and static graph shape, but it does not prove live import success.

## Commands

```bash
inferdoctor dify template export local-private-rag --output ./dify-local-private-rag
inferdoctor dify validate ./dify-local-private-rag
inferdoctor dify smoke --kit ./dify-local-private-rag --dry-run
```

After manual Dify setup:

```bash
export DIFY_API_BASE_URL=http://127.0.0.1:5001/v1
export DIFY_APP_API_KEY=your-app-api-key

inferdoctor dify check --base-url "$DIFY_API_BASE_URL"
inferdoctor dify smoke --base-url "$DIFY_API_BASE_URL"
inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --format json --output dify-after.json
inferdoctor dify optimize --report dify-after.json --kit ./dify-local-private-rag
```

## Safety

Do not use private documents in smoke-test queries. Do not commit API keys. Do not treat static DSL validation as import verification.
