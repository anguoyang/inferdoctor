# Gold Context Probe

`inferdoctor rag probe gold-context` separates retrieval and context failures from prompt/model limitations.

The command supplies an explicit gold context to an OpenAI-compatible endpoint and checks deterministic required facts and forbidden claims. It is a diagnostic probe, not a benchmark.

Safety defaults:

- one request only;
- strict timeout and response-size limit;
- no model download;
- no runtime installation;
- no endpoint discovery;
- localhost is allowed;
- LAN/private endpoints require `--allow-non-local`;
- public endpoints require `--allow-public`;
- API keys are read only from a named environment variable;
- URL credentials are rejected;
- full answer retention is off unless explicitly requested.

Use `--dry-run` to validate request construction without contacting a model endpoint.
