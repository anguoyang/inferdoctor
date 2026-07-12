# Customer Service Reference App

A small reference app for a local or private OpenAI-compatible customer-service assistant.

This example is intentionally lightweight:

- no model download
- no runtime installation
- no cloud API key required by default
- dry-run and config-check modes do not call any endpoint
- live mode sends only the prompt you type to the configured endpoint

## Configure

Copy `.env.example` to `.env` or edit `config.yaml`:

```bash
LOCAL_AI_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_AI_MODEL=local-model
LOCAL_AI_STREAMING=true
LOCAL_AI_TIMEOUT=30
```

Use localhost for local runtimes. For LAN or private endpoints, make sure you control the endpoint and do not send private customer data in smoke tests.

## Safe Smoke Modes

```bash
python app.py --help
python app.py --check-config
python app.py --dry-run
```

These modes do not call the model endpoint.

## Optional Live Modes

```bash
python app.py --check-endpoint
python app.py --message "What is the return policy?"
```

Live mode calls the configured endpoint. It supports streaming by default and falls back to full JSON responses when streaming is not returned.

## Validate With InferDoctor

```bash
inferdoctor template validate .
inferdoctor template smoke-test .
```

## Performance Verification

Create a baseline before changing runtime, model, quantization, prompt, or networking:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
```

After a change, run another tiny smoke test and compare:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output after.json
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --baseline before.json --candidate after.json --goal customer-service
```

For a LAN/private endpoint you control, add `--allow-non-local` to `inferdoctor perf` commands.

## Files

```text
.
├── README.md
├── app.py
├── config.yaml
├── .env.example
├── data/faq.md
└── instructions.md
```

## Limitations

This is a readable reference app, not production customer support software. Add auth, logging policy, rate limits, prompt injection defenses, and real evaluation before production use.
