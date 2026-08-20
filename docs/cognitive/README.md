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
