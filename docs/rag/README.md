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
inferdoctor rag capture dify-knowledge --base-url http://127.0.0.1:5001/v1 --dataset-id DATASET_ID --query "fictional return policy" --output trace.json
inferdoctor rag trace validate trace.json
inferdoctor rag diagnose --case case.json --trace trace.json --format markdown --output diagnosis.md
inferdoctor rag compare --case case.json --before before-trace.json --after after-trace.json
inferdoctor rag probe gold-context --case case.json --context-file gold-context.md --endpoint http://127.0.0.1:8000/v1 --model local-model --dry-run
```

The schemas are framework-neutral. Adapters can be written for Dify, LangChain, LlamaIndex, RAGFlow, or custom applications as long as they emit the public trace schema.

## Regression quality gate

Know whether an AI change is safe to ship from the evidence you already collect.

The gate reuses existing RAG Cases, Traces, `compare_rag()`, and causal diagnosis:

```bash
inferdoctor rag gate \
  --cases cases.jsonl \
  --before traces/before \
  --after traces/after
```

Trace filenames do not need to match. The gate matches Cases and Traces by `case_id`, rejects ambiguous duplicates, and treats missing, invalid, redacted, incompatible, or non-evaluable evidence as inconclusive rather than passing it silently.

Model, provider, and pipeline metadata changes are recorded as candidate implementation changes rather than treated as Gate incompatibilities. Observed latency deltas remain evidence, but without an explicit performance policy they do not decide this quality verdict; workload identity changes still make the comparison inconclusive.

Exit codes are suitable for CI:

- `0`: all evaluated Cases are improved or unchanged.
- `1`: at least one established regression blocks the change.
- `2`: no regression was established, but input or comparison evidence is inconclusive.

A minimal GitHub Actions step for a repository checkout is:

```yaml
- name: Install InferDoctor checkout
  run: python -m pip install -e .
- name: Gate RAG change
  run: |
    inferdoctor rag gate \
      --cases cases.jsonl \
      --before traces/before \
      --after traces/after \
      --format markdown \
      --output rag-quality-gate.md
```

## Privacy

A trace can omit source text, prompt text, and answer text while preserving IDs, hashes, ranks, scores, lengths, timings, and status. Public examples must use fictional data only.


## Dify retrieval capture

`rag capture dify-knowledge` reuses InferDoctor's existing Dify Knowledge API client and converts the returned retrieval records into `inferdoctor.rag.trace.v1`.

By default, query text and retrieved chunk text are not retained. Their hashes and safe structural metadata are exported instead. Use `--include-content` only for synthetic or explicitly approved diagnostic content.

The Dify Knowledge API adapter is intentionally retrieval-only. It does not claim that retrieved chunks were selected into the final application context, and it does not infer prompt, generation, tool, or post-processing behavior that the Knowledge API did not expose.
