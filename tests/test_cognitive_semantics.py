from inferdoctor.core.cognitive_semantics import (
    COGNITIVE_CASE_SCHEMA_VERSION,
    evaluate_cognitive_case,
    semantic_value_sha256,
    validate_cognitive_case,
)
from inferdoctor.core.dify_cognitive import (
    project_dify_cognitive_trace,
)


def case(**kwargs):
    item = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": "semantic-case-1",
    }

    item.update(kwargs)

    return item


def node(
    node_id,
    node_type,
    decision=None,
):
    data = {
        "node_id": node_id,
        "node_type": node_type,
        "status": "succeeded",
    }

    if decision is not None:
        data[
            "decision_sha256"
        ] = semantic_value_sha256(
            decision
        )

    return {
        "event": "node_finished",
        "data": data,
    }


def test_intent_mismatch_is_first_semantic_broken_layer():
    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
                "billing",
            ),
            node(
                "route",
                "if-else",
                "billing-route",
            ),
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent=(
                "refund"
            ),
            expected_route=(
                "billing-route"
            ),
        ),
        trace,
    )

    assert (
        result["semantic_status"]
        == "FAIL"
    )

    assert (
        result["first_broken_layer"]
        == "intent"
    )

    intent = next(
        item
        for item in result["layers"]
        if item["layer"]
        == "intent"
    )

    route = next(
        item
        for item in result["layers"]
        if item["layer"]
        == "route"
    )

    assert (
        intent["semantic_role"]
        == "FIRST_BROKEN"
    )

    assert (
        route["semantic_role"]
        == "DOWNSTREAM_OBSERVATION"
    )


def test_route_can_be_first_broken_after_intent_passes():
    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
                "refund",
            ),
            node(
                "route",
                "if-else",
                "wrong-route",
            ),
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent=(
                "refund"
            ),
            expected_route=(
                "refund-route"
            ),
        ),
        trace,
    )

    assert (
        result["first_broken_layer"]
        == "route"
    )

    intent = next(
        item
        for item in result["layers"]
        if item["layer"]
        == "intent"
    )

    assert (
        intent["semantic_status"]
        == "PASS"
    )

    assert (
        intent["semantic_role"]
        == "ESTABLISHED_UPSTREAM"
    )


def test_missing_decision_evidence_remains_unknown():
    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
            )
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent=(
                "refund"
            )
        ),
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


def test_hashed_expectation_is_supported():
    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
                "refund",
            )
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent={
                "sha256": (
                    semantic_value_sha256(
                        "refund"
                    )
                )
            }
        ),
        trace,
    )

    assert (
        result["semantic_status"]
        == "PASS"
    )


def test_wrong_tool_is_action_failure():
    trace = (
        project_dify_cognitive_trace([
            {
                "event": (
                    "agent_thought"
                ),
                "tool": "web_search",
            }
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_tool=(
                "crm_lookup"
            )
        ),
        trace,
    )

    assert (
        result["first_broken_layer"]
        == "action"
    )


def test_missing_expected_source_is_retrieval_failure():
    trace = (
        project_dify_cognitive_trace([
            {
                "event": "message_end",
                "retriever_resources": [
                    {
                        "document_id": (
                            "doc-other"
                        ),
                        "segment_id": (
                            "seg-1"
                        ),
                    }
                ],
            }
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_sources=[
                "doc-policy"
            ]
        ),
        trace,
    )

    assert (
        result["first_broken_layer"]
        == "retrieval"
    )


def test_all_available_expectations_can_pass():
    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
                "refund",
            ),
            node(
                "route",
                "if-else",
                "refund-route",
            ),
            {
                "event": (
                    "agent_thought"
                ),
                "tool": "crm_lookup",
            },
            {
                "event": "message_end",
                "retriever_resources": [
                    {
                        "document_id": (
                            "doc-policy"
                        ),
                        "segment_id": (
                            "seg-policy"
                        ),
                    }
                ],
            },
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent=(
                "refund"
            ),
            expected_route=(
                "refund-route"
            ),
            expected_tool=(
                "crm_lookup"
            ),
            expected_sources=[
                "doc-policy"
            ],
        ),
        trace,
    )

    assert (
        result["semantic_status"]
        == "PASS"
    )

    assert (
        result["first_broken_layer"]
        is None
    )


def test_invalid_case_requires_expectation():
    errors = validate_cognitive_case({
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": "empty",
    })

    assert errors


def test_renderer_marks_semantic_first_broken_layer():
    from inferdoctor.core.cognitive import (
        render_cognitive_analysis,
    )

    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
                "refund",
            ),
            node(
                "route",
                "if-else",
                "wrong-route",
            ),
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent="refund",
            expected_route="refund-route",
        ),
        trace,
    )

    rendered = (
        render_cognitive_analysis(
            result
        )
    )

    assert (
        "Semantic first broken layer: route"
        in rendered
    )

    assert (
        "FIRST SEMANTIC BROKEN"
        in rendered
    )



def test_plan_can_be_first_broken_after_intent_and_route_pass():
    trace = (
        project_dify_cognitive_trace([
            node(
                "intent",
                "question-classifier",
                "refund",
            ),
            node(
                "route",
                "if-else",
                "refund-route",
            ),
            {
                "event": "agent_thought",
                "position": 1,
                "tool": "web_search",
            },
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_intent="refund",
            expected_route=(
                "refund-route"
            ),
            expected_plan=[
                "crm_lookup"
            ],
            expected_tool=(
                "crm_lookup"
            ),
        ),
        trace,
    )

    assert (
        result["first_broken_layer"]
        == "plan"
    )

    plan = next(
        item
        for item in result["layers"]
        if item["layer"] == "plan"
    )

    action = next(
        item
        for item in result["layers"]
        if item["layer"] == "action"
    )

    assert (
        plan["semantic_status"]
        == "FAIL"
    )

    assert (
        plan["semantic_role"]
        == "FIRST_BROKEN"
    )

    assert (
        action["semantic_status"]
        == "FAIL"
    )

    assert (
        action["semantic_role"]
        == "DOWNSTREAM_OBSERVATION"
    )


def test_plan_sequence_can_pass():
    trace = (
        project_dify_cognitive_trace([
            {
                "event": "agent_thought",
                "position": 2,
                "tool": "send_email",
            },
            {
                "event": "agent_thought",
                "position": 1,
                "tool": "crm_lookup",
            },
        ])
    )

    result = evaluate_cognitive_case(
        case(
            expected_plan={
                "tool_sequence": [
                    "crm_lookup",
                    "send_email",
                ]
            }
        ),
        trace,
    )

    assert (
        result["semantic_status"]
        == "PASS"
    )


def test_missing_plan_evidence_is_unknown():
    trace = {
        "schema_version": (
            "inferdoctor.cognitive.trace.v1"
        ),
        "observations": [],
    }

    result = evaluate_cognitive_case(
        case(
            expected_plan=[
                "crm_lookup"
            ]
        ),
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
