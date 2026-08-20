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

## Minimal next probe

After semantic attribution, InferDoctor recommends only the smallest controlled experiment needed to move the diagnosis forward.

Examples:

- Intent failure -> Gold Intent
- Route failure -> Gold Route
- Plan failure -> Gold Plan
- Action/tool failure -> Gold Tool Result
- Retrieval/context failure -> Gold Context
- Generation failure -> model capability probe only after upstream evidence is fixed
- Post-processing failure -> raw-vs-final comparison

The planner follows a one-variable rule: do not change intent, routing, retrieval, prompt, and model at the same time.

A Gold Probe is evidence isolation, not an optimization engine. InferDoctor does not implement another agent framework or workflow engine to perform the override.

### CLI

Plan the next controlled experiment from an existing Cognitive Case and Trace:

`inferdoctor cognitive probe next --case case.json --trace cognitive.json`

JSON output is also available:

`inferdoctor cognitive probe next --case case.json --trace cognitive.json --format json --output probe.json`

## Controlled replay comparison

A Gold Probe becomes stronger evidence when its replay is compared with the original trace.

InferDoctor checks whether:

1. the replay target was the original first broken layer;
2. the target changed from semantic FAIL to PASS;
3. the first broken layer moved downstream, or all evaluated failures cleared.

A downstream move supports `VALIDATED_UPSTREAM_BOTTLENECK`.

Example:

`Before: Route FAIL -> After Gold Route: Route PASS, Retrieval FAIL`

This supports Route as an upstream bottleneck and exposes Retrieval as the next failure.

The comparison does not prove that only one external variable changed. Experimental control must come from the replay procedure; InferDoctor reports that causal boundary explicitly.

### Controlled replay CLI

Compare a baseline trace with a controlled Gold Probe replay:

`inferdoctor cognitive replay compare --case case.json --before before.json --after after.json --target-layer route --probe-name gold_route`

JSON output:

`inferdoctor cognitive replay compare --case case.json --before before.json --after after.json --target-layer route --probe-name gold_route --format json --output replay.json`
