from inferdoctor.core.cognitive import (
    COGNITIVE_TRACE_SCHEMA_VERSION,
)
from inferdoctor.core.cognitive_replay import (
    compare_controlled_replay,
    render_replay_comparison,
)
from inferdoctor.core.cognitive_semantics import (
    COGNITIVE_CASE_SCHEMA_VERSION,
    semantic_value_sha256,
)


def case():
    return {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": "replay-case",
        "expected_intent": "refund",
        "expected_route": "refund-route",
        "expected_sources": [
            "doc-policy"
        ],
    }


def trace(
    *,
    intent="refund",
    route="refund-route",
    sources=None,
):
    observations = [
        {
            "layer": "intent",
            "node_type": (
                "question-classifier"
            ),
            "status": "succeeded",
            "decision_sha256": (
                semantic_value_sha256(
                    intent
                )
            ),
        },
        {
            "layer": "route",
            "node_type": "if-else",
            "status": "succeeded",
            "decision_sha256": (
                semantic_value_sha256(
                    route
                )
            ),
        },
    ]

    if sources is not None:
        observations.append({
            "layer": "retrieval",
            "node_type": (
                "retriever_resources"
            ),
            "status": "succeeded",
            "source_ids": sources,
        })

    return {
        "schema_version": (
            COGNITIVE_TRACE_SCHEMA_VERSION
        ),
        "observations": observations,
    }


def test_route_fix_moves_failure_to_retrieval():
    before = trace(
        route="wrong-route",
        sources=[
            "doc-wrong"
        ],
    )

    after = trace(
        route="refund-route",
        sources=[
            "doc-wrong"
        ],
    )

    result = compare_controlled_replay(
        case(),
        before,
        after,
        target_layer="route",
        probe_name="gold_route",
    )

    assert (
        result["verdict"]
        == "VALIDATED_UPSTREAM_BOTTLENECK"
    )

    assert (
        result["before"][
            "first_broken_layer"
        ]
        == "route"
    )

    assert (
        result["after"][
            "first_broken_layer"
        ]
        == "retrieval"
    )

    assert (
        result[
            "first_broken_moved_downstream"
        ]
        is True
    )


def test_route_fix_can_clear_all_defined_failures():
    before = trace(
        route="wrong-route",
        sources=[
            "doc-policy"
        ],
    )

    after = trace(
        route="refund-route",
        sources=[
            "doc-policy"
        ],
    )

    result = compare_controlled_replay(
        case(),
        before,
        after,
        target_layer="route",
        probe_name="gold_route",
    )

    assert (
        result["verdict"]
        == "VALIDATED_AND_CLEARED"
    )

    assert (
        result[
            "all_evaluated_failures_cleared"
        ]
        is True
    )


def test_target_must_change_from_fail_to_pass():
    before = trace(
        route="wrong-route",
        sources=[
            "doc-wrong"
        ],
    )

    after = trace(
        route="wrong-route",
        sources=[
            "doc-policy"
        ],
    )

    result = compare_controlled_replay(
        case(),
        before,
        after,
        target_layer="route",
        probe_name="gold_route",
    )

    assert (
        result["verdict"]
        == "TARGET_NOT_ISOLATED"
    )


def test_target_must_be_baseline_first_broken_layer():
    before = trace(
        intent="wrong-intent",
        route="wrong-route",
        sources=[
            "doc-wrong"
        ],
    )

    after = trace(
        intent="wrong-intent",
        route="refund-route",
        sources=[
            "doc-wrong"
        ],
    )

    result = compare_controlled_replay(
        case(),
        before,
        after,
        target_layer="route",
        probe_name="gold_route",
    )

    assert (
        result["verdict"]
        == "INVALID_BASELINE"
    )


def test_renderer_shows_attribution_movement():
    result = compare_controlled_replay(
        case(),
        trace(
            route="wrong-route",
            sources=[
                "doc-wrong"
            ],
        ),
        trace(
            route="refund-route",
            sources=[
                "doc-wrong"
            ],
        ),
        target_layer="route",
        probe_name="gold_route",
    )

    rendered = (
        render_replay_comparison(
            result
        )
    )

    assert (
        "VALIDATED_UPSTREAM_BOTTLENECK"
        in rendered
    )

    assert (
        "first_broken_layer: route"
        in rendered
    )

    assert (
        "first_broken_layer: retrieval"
        in rendered
    )
