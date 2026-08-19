# RAG Failure Diagnosis

`inferdoctor rag diagnose` compares one Case with one Trace and produces `inferdoctor.rag.diagnosis.v1`.

Diagnosis categories include:

- `retrieval_failure`
- `ranking_failure`
- `context_selection_failure`
- `context_truncation`
- `prompt_grounding_failure`
- `model_reasoning_limitation`
- `conversation_memory_contamination`
- `answer_postprocessing_failure`
- `insufficient_evidence`
- `no_clear_failure`

A category is only emitted when evidence supports it. Missing trace fields produce `insufficient_evidence` instead of invented certainty.

## Ranking attribution

InferDoctor distinguishes retrieval, ranking, and context selection instead of treating them as one failure.

- `retrieval_failure`: the required source is absent from captured retrieval candidates.
- `ranking_failure`: the required source was retrieved, but evidence shows ranking or an observed top-k cutoff excluded it.
- `context_selection_failure`: the required source was retrieved but was not selected, while available evidence does not prove ranking caused the exclusion.

Ranking attribution is deliberately conservative. InferDoctor only reports `ranking_failure` when the trace contains an explicit ranking/cutoff drop reason or when selected chunks form an observable top-ranked prefix and the required source falls below that prefix. Otherwise it does not infer a ranking cause.

## Layered attribution

Diagnosis reports include a `layer_chain` in pipeline order.

A layer can be:

- `FIRST_BROKEN`: the earliest abnormal layer supported by evidence.
- `ESTABLISHED_UPSTREAM`: an upstream condition that is directly established by the evidence needed for the first-broken diagnosis.
- `DOWNSTREAM_OBSERVATION`: a later abnormal observation. InferDoctor does not automatically claim the first broken layer caused it.
- `NOT_ATTRIBUTED`: available evidence does not support a conclusion for that layer.

InferDoctor does not mark every layer before the first failure as PASS. A PASS is only emitted when that upstream condition is directly established by the diagnostic evidence.
