# Gold Context Probe

`inferdoctor rag probe gold-context` helps separate retrieval and context failures
from prompt or model limitations. It sends an explicit gold context to an
OpenAI-compatible endpoint and evaluates the answer against deterministic Case
checks where possible.

This is a diagnostic probe, not a benchmark and not an LLM judge. InferDoctor
does not infer semantic equivalence unless the Case author provides deterministic
terms or exact phrases that make the check possible.

## Status Semantics

The report separates transport from answer evaluation:

- `request_status`: whether a request was sent or skipped.
- `transport_status`: whether the endpoint request completed successfully.
- `evaluation_status`: whether deterministic required facts and forbidden claims
  passed, failed, were skipped, or were inconclusive.
- `review_status`: whether human review is still required.
- `overall_status`: the diagnostic result: `pass`, `fail`, `inconclusive`,
  `request_failed`, or `dry_run`.
- `status`: a backward-compatible uppercase summary derived from
  `overall_status`.

A successful HTTP request is not enough for an overall pass. If the model reaches
the endpoint but misses deterministic required facts, the report uses
`transport_status: pass`, `evaluation_status: fail`, and `overall_status: fail`.

If the Case only contains `human_review` facts, InferDoctor returns
`overall_status: inconclusive` because no deterministic answer-quality decision is
available. Mixed deterministic and human-review cases report the deterministic
result separately and keep `review_status: required` when human judgment remains
necessary.

## Deterministic Matching

Deterministic checks support the Case match modes:

- `exact_phrase`
- `all_terms`
- `any_term`
- `human_review`

Matching uses Unicode NFKC normalization, case folding, whitespace normalization,
and conservative punctuation handling. It does not automatically treat `14`,
`fourteen`, and `十四` as equivalent. Add explicit alternatives in the Case if
those should all pass.

The answer is evaluated in memory before optional redaction. `answer_retained:
false` prevents answer preview retention in the report, but it does not prevent
deterministic checks from running.

## Safety Defaults

- one request only;
- strict timeout and response-size limit;
- no model download;
- no runtime installation;
- no endpoint discovery;
- localhost is allowed;
- LAN/private endpoints require `--allow-non-local`;
- public endpoints require `--allow-public`;
- API keys are read only from a named environment variable;
- URL credentials are rejected;
- full answer retention is off unless explicitly requested.

Use `--dry-run` to validate request construction without contacting a model
endpoint.
