# Dify Local / Private RAG Reference Example

This example demonstrates a safe InferDoctor workflow for a Dify RAG application connected to a local, LAN, private, or self-hosted model endpoint.

It is an educational reference, not production software.

## 1. Export the Starter Kit

```bash
inferdoctor dify template export local-private-rag --output ./dify-local-private-rag
```

## 2. Validate Offline

```bash
inferdoctor dify validate ./dify-local-private-rag
inferdoctor dify smoke --kit ./dify-local-private-rag --dry-run
```

No Dify endpoint is contacted in this phase.

## 3. Manual Dify Setup

1. Review `dify_app.yaml`.
2. Import or recreate the flow manually in Dify.
3. Configure your local/private model provider in Dify.
4. Select a knowledge base.
5. Publish the app.
6. Create an application API key.

## 4. Optional Live Smoke Test

```bash
export DIFY_API_BASE_URL=http://127.0.0.1:5001/v1
export DIFY_APP_API_KEY=your-app-api-key

inferdoctor dify check --base-url "$DIFY_API_BASE_URL"
inferdoctor dify smoke --base-url "$DIFY_API_BASE_URL"
```

Use `--allow-non-local` for private LAN endpoints you control. Use `--allow-public` only when you intentionally test a public Dify endpoint with a harmless query.

## 5. Measure, Baseline, Compare, Optimize

```bash
inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --runs 2 --warmup 1 --format json --output before.json
inferdoctor perf baseline create --report before.json --name dify-before

# Change one thing in Dify, model provider, top_k, context budget, or runtime.

inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --runs 2 --warmup 1 --format json --output after.json
inferdoctor perf compare before.json after.json
inferdoctor dify optimize --report after.json --kit ./dify-local-private-rag
```

These are bounded smoke tests, not formal benchmarks.

## Privacy Boundaries

- Do not use private documents in smoke-test queries.
- Do not commit API keys.
- Do not store answer text unless explicitly needed.
- Do not treat static DSL validation as live import verification.
