# Dify Self-Host Reliability Doctor

InferDoctor v0.7 adds read-only reliability diagnostics for self-hosted Dify deployments. The goal is not to replace Dify. Dify builds and runs the application; InferDoctor collects bounded evidence about the host, Docker Compose deployment, Plugin Daemon, Sandbox, SSRF Proxy, model endpoints, and knowledge infrastructure so users can localize the failing layer.

## What It Checks

Self-host preflight:

```bash
inferdoctor dify selfhost preflight --compose-file ./docker-compose.yaml
```

Checks host resources, Docker/Compose availability, Compose service inventory, detected Dify roles, declared host ports, and obvious missing core roles. It does not start containers or pull images.

Read-only deployment inspection:

```bash
inferdoctor dify selfhost inspect --compose-file ./docker-compose.yaml --details
```

Inspects selected Compose state, detected roles, bounded container state, and redacted log signatures when `--details` is requested. It does not dump full Docker inspect output, full logs, or environment values.

Model connectivity doctor:

```bash
inferdoctor dify connectivity check   --compose-file ./docker-compose.yaml   --endpoint http://192.168.1.20:8000/v1   --runtime openai-compatible   --role chat   --allow-non-local
```

Separates host direct path, optional container direct path, SSRF/security evidence, and Dify-mediated provider evidence. Direct network success is not treated as proof that Dify Provider configuration works.

Evidence bundles:

```bash
inferdoctor dify evidence collect --compose-file ./docker-compose.yaml --since 10m --output evidence.json
inferdoctor dify evidence explain evidence.json --format markdown --output diagnosis.md
```

Bundles contain versioned, bounded, redacted observations. They are designed for sharing in a GitHub issue or team debugging thread without exposing keys, environment values, private documents, or complete logs.

## First-Class Components

InferDoctor treats these Dify layers as separate components:

- API
- Web
- Worker
- Worker Beat or scheduler
- Plugin Daemon
- Sandbox
- SSRF Proxy
- PostgreSQL
- Redis
- vector store
- model endpoint
- Dify application API

This separation matters because a symptom such as `Internal Server Error`, `model unavailable`, or `plugin invoke error` may be downstream of a different failing component.

## Root-Cause Patterns

InferDoctor currently reasons about patterns such as:

- host can reach a model endpoint but containers cannot;
- DNS/TCP passes but HTTP route returns 404;
- raw network path works but Dify-mediated Provider path fails;
- Plugin Daemon is restarting and Provider errors appear downstream;
- Worker/indexing failures appear before empty retrieval results;
- SSRF Proxy evidence appears with 403-style failures;
- chat capability works but embedding or rerank is unavailable.

Each diagnosis states whether the evidence is observed, strongly indicated, possible, or unknown. It also gives a bounded verification command. It does not promise an exact repair.

## Safety Boundaries

InferDoctor does not:

- install Dify;
- start, stop, restart, or remove containers;
- pull Docker images;
- modify Compose files or `.env`;
- dump complete logs;
- collect full environment values;
- read database rows;
- upload documents;
- store API keys;
- run load tests.

LAN/private endpoints require `--allow-non-local`. Public endpoints require `--allow-public`. API keys should be supplied through environment-variable names, not raw command-line values.

## Cloud Compatibility

For Dify Cloud, InferDoctor can run external application API checks when the user explicitly supplies the API base URL and credentials. Deep root-cause localization requires self-host evidence such as Compose roles, container state, Plugin Daemon state, Sandbox state, and SSRF Proxy evidence.

## Related Docs

- Compatibility: https://github.com/anguoyang/inferdoctor/blob/main/docs/dify/compatibility.md
- Diagnostic matrix: https://github.com/anguoyang/inferdoctor/blob/main/docs/dify/diagnostic_matrix.md
- Security: https://github.com/anguoyang/inferdoctor/blob/main/docs/dify/security.md
- Local / Private RAG Kit: https://github.com/anguoyang/inferdoctor/blob/main/docs/dify/local_private_rag.md
