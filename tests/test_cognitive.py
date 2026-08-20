from inferdoctor.core.cognitive import (
    analyze_cognitive_trace,
    render_cognitive_analysis,
)
from inferdoctor.core.dify_cognitive import (
    project_dify_cognitive_trace,
)


def _event(
    node_id,
    node_type,
    status,
    index,
    predecessor=None,
):
    return {
        "event": "node_finished",
        "data": {
            "node_id": node_id,
            "node_type": node_type,
            "status": status,
            "index": index,
            "predecessor_node_id": (
                predecessor
            ),
        },
    }


def test_dify_nodes_project_to_cognitive_layers():
    events = [
        _event(
            "intent",
            "question-classifier",
            "succeeded",
            1,
        ),
        _event(
            "route",
            "if-else",
            "succeeded",
            2,
            "intent",
        ),
        _event(
            "agent",
            "agent",
            "succeeded",
            3,
            "route",
        ),
        _event(
            "tool",
            "tool",
            "succeeded",
            4,
            "agent",
        ),
        _event(
            "rag",
            "knowledge-retrieval",
            "succeeded",
            5,
            "tool",
        ),
        _event(
            "llm",
            "llm",
            "succeeded",
            6,
            "rag",
        ),
        _event(
            "answer",
            "answer",
            "succeeded",
            7,
            "llm",
        ),
    ]

    trace = project_dify_cognitive_trace(
        events,
        app_mode="advanced-chat",
    )

    layers = [
        item["layer"]
        for item
        in trace["observations"]
    ]

    assert layers == [
        "intent",
        "route",
        "plan",
        "action",
        "retrieval",
        "generation",
        "postprocessing",
    ]


def test_first_execution_failure_is_earliest_failed_layer():
    events = [
        _event(
            "intent",
            "question-classifier",
            "succeeded",
            1,
        ),
        _event(
            "route",
            "if-else",
            "failed",
            2,
            "intent",
        ),
        _event(
            "llm",
            "llm",
            "failed",
            3,
            "route",
        ),
    ]

    trace = project_dify_cognitive_trace(
        events
    )

    result = analyze_cognitive_trace(
        trace
    )

    assert (
        result[
            "first_execution_failure"
        ]
        == "route"
    )

    assert (
        result["execution_status"]
        == "FAIL"
    )

    assert (
        result["first_broken_layer"]
        is None
    )


def test_execution_pass_does_not_claim_semantic_pass():
    events = [
        _event(
            "intent",
            "question-classifier",
            "succeeded",
            1,
        )
    ]

    trace = project_dify_cognitive_trace(
        events
    )

    result = analyze_cognitive_trace(
        trace
    )

    intent = next(
        item
        for item in result["layers"]
        if item["layer"] == "intent"
    )

    assert (
        intent["execution_status"]
        == "PASS"
    )

    assert (
        intent["semantic_status"]
        == "UNKNOWN"
    )

    assert (
        result["semantic_status"]
        == "NOT_EVALUATED"
    )

    assert (
        result[
            "first_broken_layer_status"
        ]
        == "NOT_ESTABLISHED"
    )


def test_unknown_dify_node_is_preserved_not_guessed():
    events = [
        _event(
            "mystery",
            "future-super-node",
            "succeeded",
            1,
        )
    ]

    trace = project_dify_cognitive_trace(
        events
    )

    assert (
        trace["observations"]
        == []
    )

    assert (
        trace[
            "unmapped_node_types"
        ]
        == ["future-super-node"]
    )


def test_agent_thought_projects_plan_and_action_without_content():
    events = [
        {
            "event": "agent_thought",
            "position": 1,
            "tool": "crm_lookup",
            "thought_present": True,
            "thought_length": 20,
            "thought_sha256": "abc",
            "tool_input_present": True,
            "tool_input_length": 30,
            "tool_input_sha256": "def",
        }
    ]

    trace = project_dify_cognitive_trace(
        events
    )

    assert [
        item["layer"]
        for item
        in trace["observations"]
    ] == [
        "plan",
        "action",
    ]

    dumped = str(trace)

    assert "thought_sha256" not in dumped
    assert "tool_input_sha256" not in dumped


def test_console_explains_execution_vs_semantics():
    trace = project_dify_cognitive_trace([
        _event(
            "intent",
            "question-classifier",
            "succeeded",
            1,
        )
    ])

    result = analyze_cognitive_trace(
        trace
    )

    rendered = (
        render_cognitive_analysis(
            result
        )
    )

    assert (
        "execution=PASS"
        in rendered
    )

    assert (
        "semantic=UNKNOWN"
        in rendered
    )

    assert (
        "Semantic first broken layer: not established"
        in rendered
    )
