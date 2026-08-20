# Cognitive Path Diagnosis

InferDoctor separates execution evidence from semantic correctness.

A workflow can execute successfully while still making the wrong intent, routing, planning, tool-selection, retrieval, or generation decision.

The initial cognitive path is:

`Intent -> Route -> Plan -> Action -> Retrieval -> Context -> Generation -> Post-processing`

For Dify, native workflow node types are projected conservatively:

- `question-classifier` -> Intent
- `if-else` -> Route
- `agent` / `agent-v2` -> Plan
- `tool` / `http-request` / `code` -> Action
- `knowledge-retrieval` -> Retrieval
- `llm` -> Generation
- `answer` -> Post-processing

Unknown Dify node types remain unmapped rather than being guessed.

## Execution versus semantics

InferDoctor deliberately asks two different questions:

1. Did the layer execute successfully?
2. Was the layer's semantic decision correct?

For example, a successful Question Classifier currently means:

- execution: PASS
- semantic: UNKNOWN

It does not mean the classified intent was correct.

`first_execution_failure` may therefore be established from runtime evidence while semantic `first_broken_layer` remains unset.

Semantic attribution requires expected outcomes or controlled probes.

## Dify capture

A live Dify application can be captured with:

`inferdoctor dify trace capture --query "synthetic question" --output cognitive.json`

The adapter reuses Dify's existing streaming events.

Raw query text, node inputs, node outputs, reasoning text, tool input, and retrieved content are not retained in the cognitive trace.

## Semantic cases

Runtime success is not enough to diagnose cognitive correctness.

A Cognitive Case can define known-good expectations:

- `expected_intent`
- `expected_route`
- `expected_tool`
- `expected_sources`

Intent and route values are compared by SHA256. A case may contain either a plain expected value or a precomputed `sha256`.

The semantic evaluator reports the earliest supported mismatch as `first_broken_layer`.

Example interpretation:

`Intent PASS -> Route FAIL -> Action UNKNOWN -> Retrieval UNKNOWN`

In that case Route is the first semantic broken layer. Later problems must not automatically be blamed on retrieval or model capability.

Missing semantic evidence produces `UNKNOWN`, not failure.
