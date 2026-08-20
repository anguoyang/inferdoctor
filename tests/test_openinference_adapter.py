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



def _agent_span(
    span_id="agent-1",
):
    return {
        "name": "agent",
        "context": {
            "trace_id": "trace-plan",
            "span_id": span_id,
        },
        "status_code": "OK",
        "attributes": {
            "openinference.span.kind": (
                "AGENT"
            ),
        },
    }


def _tool_span(
    span_id,
    tool_name,
    *,
    parent_id,
    start_time=None,
):
    item = {
        "name": "tool",
        "context": {
            "trace_id": "trace-plan",
            "span_id": span_id,
        },
        "parent_id": parent_id,
        "status_code": "OK",
        "attributes": {
            "openinference.span.kind": (
                "TOOL"
            ),
            "tool.name": tool_name,
        },
    }

    if start_time is not None:
        item[
            "start_time"
        ] = start_time

    return item


def test_agent_tool_children_feed_existing_plan_semantics():
    payload = [
        _agent_span(),
        _tool_span(
            "tool-2",
            "send_email",
            parent_id="agent-1",
            start_time=(
                "2026-08-20T01:00:02+00:00"
            ),
        ),
        _tool_span(
            "tool-1",
            "crm_lookup",
            parent_id="agent-1",
            start_time=(
                "2026-08-20T01:00:01+00:00"
            ),
        ),
    ]

    trace = (
        project_openinference_trace(
            payload
        )
    )

    case = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": (
            "openinference-plan"
        ),
        "expected_plan": [
            "crm_lookup",
            "send_email",
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

    plan = [
        item
        for item
        in trace["observations"]
        if (
            item.get("source")
            == "openinference_agent_tool_sequence"
        )
    ]

    assert [
        item["planned_tool"]
        for item in plan
    ] == [
        "crm_lookup",
        "send_email",
    ]

    assert (
        trace["adapter"][
            "derived_plan_observation_count"
        ]
        == 2
    )


def test_nested_tool_descendant_can_feed_agent_plan():
    agent = _agent_span()

    chain = {
        "name": "internal-chain",
        "context": {
            "trace_id": "trace-plan",
            "span_id": "chain-1",
        },
        "parent_id": "agent-1",
        "status_code": "OK",
        "attributes": {
            "openinference.span.kind": (
                "CHAIN"
            ),
        },
    }

    tool = _tool_span(
        "tool-1",
        "crm_lookup",
        parent_id="chain-1",
    )

    trace = (
        project_openinference_trace([
            agent,
            chain,
            tool,
        ])
    )

    case = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": (
            "nested-plan"
        ),
        "expected_plan": [
            "crm_lookup"
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


def test_multi_tool_plan_without_timestamps_remains_unknown():
    trace = (
        project_openinference_trace([
            _agent_span(),
            _tool_span(
                "tool-1",
                "crm_lookup",
                parent_id="agent-1",
            ),
            _tool_span(
                "tool-2",
                "send_email",
                parent_id="agent-1",
            ),
        ])
    )

    case = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": (
            "ambiguous-plan"
        ),
        "expected_plan": [
            "crm_lookup",
            "send_email",
        ],
    }

    result = evaluate_cognitive_case(
        case,
        trace,
    )

    assert (
        result["semantic_status"]
        == "INCOMPLETE"
    )

    assert (
        result["first_broken_layer"]
        is None
    )

    assert (
        trace["adapter"][
            "derived_plan_observation_count"
        ]
        == 0
    )


def test_tool_outside_agent_is_action_not_plan():
    trace = (
        project_openinference_trace([
            _tool_span(
                "tool-1",
                "crm_lookup",
                parent_id="root-chain",
            )
        ])
    )

    plan = [
        item
        for item
        in trace["observations"]
        if (
            item.get("source")
            == "openinference_agent_tool_sequence"
        )
    ]

    action = [
        item
        for item
        in trace["observations"]
        if item["layer"]
        == "action"
    ]

    assert plan == []

    assert len(action) == 1

    assert (
        action[0]["tool_name"]
        == "crm_lookup"
    )


def test_agent_plan_can_be_first_broken_with_openinference():
    trace = (
        project_openinference_trace([
            _agent_span(),
            _tool_span(
                "tool-1",
                "web_search",
                parent_id="agent-1",
            ),
        ])
    )

    case = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": (
            "openinference-plan-fail"
        ),
        "expected_plan": [
            "crm_lookup"
        ],
        "expected_tool": (
            "crm_lookup"
        ),
    }

    result = evaluate_cognitive_case(
        case,
        trace,
    )

    assert (
        result["first_broken_layer"]
        == "plan"
    )

    plan = next(
        item
        for item in result["layers"]
        if item["layer"]
        == "plan"
    )

    action = next(
        item
        for item in result["layers"]
        if item["layer"]
        == "action"
    )

    assert (
        plan["semantic_role"]
        == "FIRST_BROKEN"
    )

    assert (
        action["semantic_role"]
        == "DOWNSTREAM_OBSERVATION"
    )
