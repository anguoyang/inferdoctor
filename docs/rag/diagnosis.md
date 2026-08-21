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

## Evidence sufficiency

`inferdoctor.rag.diagnosis.v1` includes an additive
`evidence_sufficiency` summary:

- `SUFFICIENT`: observed evidence supports the reported First Broken Layer at
  its stated confidence, or available deterministic checks support the current
  no-failure conclusion.
- `INSUFFICIENT`: a required observation or answer-evaluation criterion is
  missing, redacted, or unavailable, so causal attribution is blocked.
- `PARTIAL`: useful observations are established, but one controlled probe or
  human review is still needed for a stronger conclusion.

The summary also reports concise `known` and `unknown` facts and whether the
evidence supports the reported First Broken Layer. It does not use
`evidence_completeness_score` as a causality threshold. That score remains only
for backward compatibility.

InferDoctor does not guess when evidence is missing:

- a missing retrieval candidate list is `UNKNOWN`, not retrieval failure;
- an explicitly observed empty candidate list is different and can support
  retrieval failure when a required source is defined;
- a redacted context is distinguishable from an observed empty context;
- a failed answer after correct context reached generation is an observed
  symptom, not proof that the underlying model is incapable.

## Minimal Next Probe

When causal attribution is blocked or an important uncertainty remains, the
diagnosis includes one structured `minimal_next_probe`. It contains a probe
type, target layer, required evidence, action, reason, and the outcomes that the
probe distinguishes.

Selection follows the earliest unresolved causal boundary, not a static list of
missing telemetry. For example:

- if fictional policy retrieval candidates are absent, capture retrieval
  evidence;
- if the required policy was retrieved but selected chunk IDs are absent,
  capture selection evidence;
- if selected sources and the failed final answer are known but final model
  context is absent, capture or evaluate that context;
- if correct context, prompt grounding, raw output, and a deterministic failed
  answer are observed, use the existing Gold Context Probe for controlled
  isolation;
- if the Case has no usable deterministic or human-review criterion, define the
  smallest answer-evaluation criterion first.

An established upstream First Broken Layer does not trigger a probe aimed only
at a downstream symptom. A complete successful case has
`minimal_next_probe: null`.

Probe recommendations do not change privacy defaults. Content may be evaluated
in memory or captured only in an explicitly approved export mode; hashes and
redaction states remain supported, and chain-of-thought is never required.

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
