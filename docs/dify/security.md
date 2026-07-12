# Dify Security and Privacy

InferDoctor Dify commands are designed to be read-only and explicit by default.

## API Keys

Use environment variables:

- `DIFY_APP_API_KEY`
- `DIFY_KNOWLEDGE_API_KEY`

Application API keys and knowledge API keys are separate. Knowledge API keys can have broader access and should be handled with extra care.

Do not pass raw API keys on the command line.

## Endpoint Opt-In

Localhost endpoints may be checked directly.

Private or LAN endpoints require:

```bash
--allow-non-local
```

Public endpoints require:

```bash
--allow-public
```

InferDoctor never scans for Dify and never contacts an endpoint inferred from a DSL file.

## Content Handling

By default, Dify smoke tests suppress answer text and knowledge retrieval content.

Use explicit flags only with non-sensitive data:

```bash
inferdoctor dify smoke --show-answer
inferdoctor dify knowledge check --show-content
```

## What Is Not Automated

InferDoctor does not import DSL files, create apps, create knowledge bases, upload documents, modify Dify settings, start containers, or install model runtimes.
