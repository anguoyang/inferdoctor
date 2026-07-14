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
