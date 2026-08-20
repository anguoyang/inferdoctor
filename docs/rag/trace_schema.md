# RAG Trace Schema

Schema version: `inferdoctor.rag.trace.v1`

A Trace describes one RAG execution. Required top-level sections are:

- `schema_version`
- `trace_id`
- `timestamp`
- `system`
- `pipeline`
- `input`
- `retrieval`
- `context_selection`
- `generation`
- `postprocessing`
- `timings`
- `privacy`

Trace content is optional. When content is omitted, adapters should still emit useful IDs, ranks, scores, hashes, lengths, timings, and status values.
