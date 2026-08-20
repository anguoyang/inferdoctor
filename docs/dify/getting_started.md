# Dify Integration Getting Started

InferDoctor v0.7 adds a Dify-first workflow for local, LAN, private, self-hosted, and explicitly approved Dify Cloud applications.

The goal is not to replace Dify. Dify remains the orchestration layer for Chatflow, Workflow, application APIs, knowledge APIs, and manual DSL import/export. InferDoctor helps you validate the kit, run safe smoke tests, measure perceived latency, compare changes, and generate Dify-specific optimization guidance.

## Install

```bash
pip install inferdoctor
```

## Offline First

Start without a Dify instance:

```bash
inferdoctor dify template list
inferdoctor dify template show local-private-rag
inferdoctor dify template export local-private-rag --output ./dify-local-private-rag
inferdoctor dify validate ./dify-local-private-rag
inferdoctor dify smoke --kit ./dify-local-private-rag --dry-run
```

These commands do not contact Dify, import a DSL, create a knowledge base, upload documents, install runtimes, or download models.

## Live App API Checks

After you manually import or recreate the app in Dify, publish it, and create an app API key:

```bash
export DIFY_API_BASE_URL=http://127.0.0.1:5001/v1
export DIFY_APP_API_KEY=your-app-api-key

inferdoctor dify check --base-url "$DIFY_API_BASE_URL"
inferdoctor dify smoke --base-url "$DIFY_API_BASE_URL"
inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --runs 2 --warmup 1 --format json --output dify-perf.json
inferdoctor perf baseline create --report dify-perf.json --name dify-before
inferdoctor dify optimize --report dify-perf.json --kit ./dify-local-private-rag
```

For private LAN endpoints, add `--allow-non-local`. For public endpoints such as Dify Cloud, add `--allow-public` only when you intentionally want to send a harmless smoke-test query.

## Credential Separation

Application API keys and knowledge API keys are separate.

Use:

- `DIFY_API_BASE_URL`
- `DIFY_APP_API_KEY`
- `DIFY_KNOWLEDGE_API_BASE_URL`
- `DIFY_KNOWLEDGE_API_KEY`
- `DIFY_DATASET_ID`

Do not pass raw keys on the command line.

## Limits

InferDoctor smoke tests are not formal benchmarks. They do not measure model quality, concurrency, sustained throughput, or production reliability.
