# Local Document Q&A Reference App

A small reference app for local document Q&A with keyword retrieval and optional OpenAI-compatible generation.

This example is intentionally lightweight:

- no vector database
- no embedding model
- no model download
- no runtime installation
- dry-run and config-check modes do not call any endpoint

## Configure

```bash
LOCAL_AI_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_AI_MODEL=local-model
LOCAL_AI_STREAMING=true
LOCAL_AI_TOP_K=4
LOCAL_AI_CONTEXT_BUDGET=4000
```

## Build Local Index

```bash
python ingest.py --dry-run
python ingest.py
```

`ingest.py` reads Markdown files from `docs/` and writes a tiny local keyword index. It does not call a model.

## Safe Smoke Modes

```bash
python query.py --help
python query.py --check-config
python query.py --dry-run
```

These modes do not call a model endpoint.

## Optional Live Generation

```bash
python query.py "What is InferDoctor?" --generate
```

Live generation sends selected local context and your question to the configured endpoint. Streaming is enabled by default.

## Validate With InferDoctor

```bash
inferdoctor template validate .
inferdoctor template smoke-test .
```

## Performance Verification

For RAG apps, measure endpoint responsiveness separately from retrieval stage latency:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
inferdoctor optimize rag --top-k 4 --retrieval-ms 700 --ttft 2.5 --streaming
inferdoctor optimize plan --report before.json --goal document-qa --retrieval-ms 700
```

After a change:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output after.json
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --baseline before.json --candidate after.json --goal document-qa
```

For a LAN/private endpoint you control, add `--allow-non-local` to `inferdoctor perf` commands.

## Stage Latency Notes

A document Q&A app can feel slow before generation starts. Track these stages separately:

- query preprocessing
- retrieval
- optional rerank
- context assembly
- LLM TTFT
- streamed generation

Show progress before generation so users know the app is searching local documents.

## Files

```text
.
├── README.md
├── ingest.py
├── query.py
├── config.yaml
├── .env.example
├── requirements.txt
└── docs/sample.md
```

## Limitations

This is a readable reference app, not a production RAG system. Add real chunking, embeddings, access control, evaluation, and document security before production use.
