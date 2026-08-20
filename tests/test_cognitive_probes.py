from inferdoctor.core.cognitive_probes import (
    plan_next_cognitive_probe,
    render_probe_plan,
)


def analysis(
    *,
    first=None,
    semantic="FAIL",
    execution="PASS",
    first_execution=None,
):
    layers = []

    for layer in (
        "intent",
        "route",
        "plan",
        "action",
        "retrieval",
        "context",
        "generation",
        "postprocessing",
    ):
        layers.append({
            "layer": layer,
            "semantic_status": (
                "PASS"
                if (
                    first
                    and layer
                    in {
                        "intent",
                    }
                    and first != "intent"
                )
                else "NOT_EVALUATED"
            ),
        })

    return {
        "first_broken_layer": first,
        "semantic_status": semantic,
        "execution_status": execution,
        "first_execution_failure": (
            first_execution
        ),
        "layers": layers,
    }


def test_intent_failure_recommends_gold_intent():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first="intent"
            )
        )
    )

    assert (
        result["next_probe"]
        == "gold_intent"
    )

    assert (
        result["target_layer"]
        == "intent"
    )

    assert (
        result[
            "change_one_variable_only"
        ]
        is True
    )


def test_route_failure_recommends_gold_route():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first="route"
            )
        )
    )

    assert (
        result["next_probe"]
        == "gold_route"
    )


def test_action_failure_recommends_gold_tool_result():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first="action"
            )
        )
    )

    assert (
        result["next_probe"]
        == "gold_tool_result"
    )


def test_retrieval_failure_recommends_existing_gold_context_direction():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first="retrieval"
            )
        )
    )

    assert (
        result["next_probe"]
        == "gold_context"
    )


def test_generation_failure_does_not_immediately_blame_model():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first="generation"
            )
        )
    )

    assert (
        result["next_probe"]
        == "model_capability"
    )

    assert (
        "before attributing failure"
        in result["goal"]
    )


def test_incomplete_semantics_requests_evidence_first():
    item = analysis(
        first=None,
        semantic="INCOMPLETE",
    )

    item["layers"][0][
        "semantic_status"
    ] = "UNKNOWN"

    result = (
        plan_next_cognitive_probe(
            item
        )
    )

    assert (
        result["status"]
        == "MORE_EVIDENCE_REQUIRED"
    )

    assert (
        result["next_probe"]
        == "capture_missing_evidence"
    )


def test_all_semantic_expectations_pass_without_probe():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first=None,
                semantic="PASS",
                execution="PASS",
            )
        )
    )

    assert (
        result["status"]
        == "NO_PROBE_NEEDED"
    )

    assert (
        result["next_probe"]
        is None
    )


def test_execution_failure_blocks_gold_probe():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first=None,
                semantic="NOT_EVALUATED",
                execution="FAIL",
                first_execution="route",
            )
        )
    )

    assert (
        result["status"]
        in {
            "EXECUTION_FAILURE_FIRST",
            "MORE_EVIDENCE_REQUIRED",
        }
    )


def test_probe_renderer_is_human_readable():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first="route"
            )
        )
    )

    rendered = render_probe_plan(
        result
    )

    assert (
        "next_probe: gold_route"
        in rendered
    )

    assert (
        "change exactly one"
        in rendered
    )


def test_execution_failure_strictly_blocks_semantic_probe():
    result = (
        plan_next_cognitive_probe(
            analysis(
                first=None,
                semantic="NOT_EVALUATED",
                execution="FAIL",
                first_execution="route",
            )
        )
    )

    assert (
        result["status"]
        == "EXECUTION_FAILURE_FIRST"
    )

    assert (
        result["next_probe"]
        is None
    )

    assert (
        result["target_layer"]
        == "route"
    )
