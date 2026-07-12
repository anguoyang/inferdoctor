# Dify Performance UX

Dify application performance is not only endpoint reachability. Users notice:

- first SSE event latency
- first visible answer text latency
- total response latency
- whether streaming is working
- whether retrieval progress is visible
- failed nodes or stream errors
- cold versus warm behavior

InferDoctor Dify performance checks are bounded smoke tests, not benchmarks.

## Measure

```bash
inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --runs 2 --warmup 1 --format json --output before.json
```

The report uses schema:

```text
inferdoctor.dify.performance.v1
```

It can be used with existing v0.6 workflows:

```bash
inferdoctor perf baseline create --report before.json --name dify-before
inferdoctor perf compare before.json after.json
inferdoctor dify optimize --report after.json --kit ./dify-local-private-rag
```

## Interpret

Low first visible answer text latency usually matters more than total latency for interactive Dify apps.

RAG applications should show retrieval progress before the model starts generating. A slow total answer can still feel acceptable if the first visible signal arrives quickly and progress is clear.

## Limitations

InferDoctor does not run concurrency tests, sustained load tests, model-quality evaluation, or Dify worker profiling.
