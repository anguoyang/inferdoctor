# RAG Case Schema

Schema version: `inferdoctor.rag.case.v1`

A Case describes one question that should be evaluated. Required fields are `schema_version`, `case_id`, `question`, `language`, `category`, and `why_bad`.

Supported evidence fields include:

- `expected_sources`
- `required_facts`
- `forbidden_claims`
- `expected_answer`
- `current_answer`
- `expected_behavior`
- `metadata`

Fact and claim matching supports `all_terms`, `any_term`, `exact_phrase`, and `human_review`. `human_review` explicitly means deterministic verification is unavailable.
