import json

from inferdoctor.core.cognitive_semantics import (
    COGNITIVE_CASE_SCHEMA_VERSION,
    evaluate_cognitive_case,
)
from inferdoctor.core.openinference_adapter import (
    extract_openinference_spans,
    project_openinference_trace,
)


def span(
    kind,
    *,
    status="OK",
    attributes=None,
):
    attrs = {
        "openinference.span.kind": (
            kind
        ),
    }

    attrs.update(
        attributes or {}
    )

    return {
        "name": (
            "private span name"
        ),
        "context": {
            "trace_id": "trace-1",
            "span_id": (
                "span-" + kind.lower()
            ),
        },
        "status_code": status,
        "attributes": attrs,
    }


def test_openinference_maps_standard_span_kinds():
    trace = (
        project_openinference_trace([
            span("AGENT"),
            span(
                "TOOL",
                attributes={
                    "tool.name": (
                        "crm_lookup"
                    )
                },
            ),
            span(
                "RETRIEVER",
                attributes={
                    (
                        "retrieval.documents."
                        "0.document.id"
                    ): "doc-1"
                },
            ),
            span(
                "RERANKER",
                attributes={
                    (
                        "reranker.documents."
                        "0.document.id"
                    ): "doc-2"
                },
            ),
            span(
                "LLM",
                attributes={
                    "llm.model_name": (
                        "local-model"
                    )
                },
            ),
        ])
    )

    assert [
        item["layer"]
        for item
        in trace["observations"]
    ] == [
        "plan",
        "action",
        "retrieval",
        "retrieval",
        "generation",
    ]

    assert (
        trace["adapter"][
            "mapped_span_count"
        ]
        == 5
    )


def test_openinference_does_not_retain_raw_input_or_output():
    trace = (
        project_openinference_trace([
            span(
                "LLM",
                attributes={
                    "input.value": (
                        "SECRET INPUT"
                    ),
                    "output.value": (
                        "SECRET OUTPUT"
                    ),
                },
            )
        ])
    )

    dumped = json.dumps(
        trace,
        ensure_ascii=False,
    )

    assert (
        "SECRET INPUT"
        not in dumped
    )

    assert (
        "SECRET OUTPUT"
        not in dumped
    )

    observation = trace[
        "observations"
    ][0]

    assert observation[
        "input_sha256"
    ]

    assert observation[
        "output_sha256"
    ]


def test_openinference_tool_reuses_existing_action_semantics():
    trace = (
        project_openinference_trace([
            span(
                "TOOL",
                attributes={
                    "tool.name": (
                        "crm_lookup"
                    )
                },
            )
        ])
    )

    case = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": (
            "openinference-tool"
        ),
        "expected_tool": (
            "crm_lookup"
        ),
    }

    result = evaluate_cognitive_case(
        case,
        trace,
    )

    assert (
        result["semantic_status"]
        == "PASS"
    )


def test_openinference_retriever_reuses_source_semantics():
    trace = (
        project_openinference_trace([
            span(
                "RETRIEVER",
                attributes={
                    (
                        "retrieval.documents."
                        "0.document.id"
                    ): "doc-policy",
                    (
                        "retrieval.documents."
                        "1.document.id"
                    ): "doc-other",
                },
            )
        ])
    )

    case = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": (
            "openinference-rag"
        ),
        "expected_sources": [
            "doc-policy"
        ],
    }

    result = evaluate_cognitive_case(
        case,
        trace,
    )

    assert (
        result["semantic_status"]
        == "PASS"
    )


def test_unknown_openinference_kinds_are_preserved_not_guessed():
    trace = (
        project_openinference_trace([
            span("PROMPT"),
            span("GUARDRAIL"),
            span("EVALUATOR"),
        ])
    )

    assert (
        trace["observations"]
        == []
    )

    assert (
        trace[
            "unmapped_span_kinds"
        ]
        == [
            "EVALUATOR",
            "GUARDRAIL",
            "PROMPT",
        ]
    )


def test_openinference_error_status_maps_to_execution_failure():
    trace = (
        project_openinference_trace([
            span(
                "TOOL",
                status="ERROR",
                attributes={
                    "tool.name": (
                        "crm_lookup"
                    )
                },
            )
        ])
    )

    assert (
        trace["observations"][0][
            "status"
        ]
        == "failed"
    )


def test_otlp_json_shape_is_supported():
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": (
                                    "trace-otlp"
                                ),
                                "spanId": (
                                    "span-tool"
                                ),
                                "name": (
                                    "tool-call"
                                ),
                                "status": {
                                    "code": (
                                        "STATUS_CODE_OK"
                                    )
                                },
                                "attributes": [
                                    {
                                        "key": (
                                            "openinference.span.kind"
                                        ),
                                        "value": {
                                            "stringValue": (
                                                "TOOL"
                                            )
                                        },
                                    },
                                    {
                                        "key": (
                                            "tool.name"
                                        ),
                                        "value": {
                                            "stringValue": (
                                                "crm_lookup"
                                            )
                                        },
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    spans = (
        extract_openinference_spans(
            payload
        )
    )

    assert len(spans) == 1

    trace = (
        project_openinference_trace(
            payload
        )
    )

    assert (
        trace["trace_id"]
        == "trace-otlp"
    )

    assert (
        trace["observations"][0][
            "tool_name"
        ]
        == "crm_lookup"
    )
