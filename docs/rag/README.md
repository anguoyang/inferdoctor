# RAG Intelligence Doctor

InferDoctor RAG Intelligence is a framework-neutral diagnostic foundation for cases where a RAG application runs but gives poor, incomplete, ungrounded, or apparently wrong answers.

The diagnostic sequence is explicit:

1. Does the correct source exist?
2. Was it retrieved?
3. Was it ranked high enough?
4. Was it selected into the final context?
5. Was it truncated or diluted?
6. Did the prompt clearly instruct grounding?
7. Did the raw model answer use the evidence?
8. Did post-processing damage the answer?
9. Did conversation history contaminate the result?
10. Does the same model answer correctly when given gold context?

InferDoctor does not start with one opaque LLM-generated score. Deterministic evidence and explicit uncertainty come first.

## Commands

```bash
inferdoctor rag case init --output rag-cases.jsonl
inferdoctor rag case validate rag-cases.jsonl
inferdoctor rag trace validate trace.json
inferdoctor rag diagnose --case case.json --trace trace.json --format markdown --output diagnosis.md
inferdoctor rag compare --case case.json --before before-trace.json --after after-trace.json
inferdoctor rag probe gold-context --case case.json --context-file gold-context.md --endpoint http://127.0.0.1:8000/v1 --model local-model --dry-run
```

The schemas are framework-neutral. Adapters can be written for Dify, LangChain, LlamaIndex, RAGFlow, or custom applications as long as they emit the public trace schema.

## Privacy

A trace can omit source text, prompt text, and answer text while preserving IDs, hashes, ranks, scores, lengths, timings, and status. Public examples must use fictional data only.
