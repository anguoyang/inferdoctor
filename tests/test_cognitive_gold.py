from inferdoctor.core.cognitive_gold import (
    interpret_gold_context_probe,
    render_cognitive_gold_context,
    run_cognitive_gold_context_probe,
)


def baseline(
    first,
):
    return {
        "first_broken_layer": first,
        "semantic_status": "FAIL",
    }


def gold_result(
    *,
    overall,
    transport="pass",
    evaluation="pass",
    deterministic=True,
):
    return {
        "schema_version": (
            "inferdoctor.rag.gold_context_probe.v1"
        ),
        "transport_status": transport,
        "evaluation_status": evaluation,
        "overall_status": overall,
        "deterministic_checks_available": (
            deterministic
        ),
        "status": overall.upper(),
    }


def test_gold_context_pass_supports_upstream_evidence_path():
    result = interpret_gold_context_probe(
        baseline("retrieval"),
        gold_result(
            overall="pass"
        ),
    )

    assert (
        result["verdict"]
        == "GOLD_CONTEXT_PASS"
    )

    assert (
        result["diagnostic_effect"]
        == "UPSTREAM_EVIDENCE_PATH_SUPPORTED"
    )

    assert (
        "not a strict one-variable replay"
        in result["causal_boundary"]
    )


def test_gold_context_pass_refines_generation_attribution():
    result = interpret_gold_context_probe(
        baseline("generation"),
        gold_result(
            overall="pass"
        ),
    )

    assert (
        result["diagnostic_effect"]
        == "GENERATION_FAILURE_NOT_REPRODUCED"
    )


def test_gold_context_fail_keeps_downstream_limitation_plausible():
    result = interpret_gold_context_probe(
        baseline("retrieval"),
        gold_result(
            overall="fail",
            evaluation="fail",
        ),
    )

    assert (
        result["verdict"]
        == "GOLD_CONTEXT_FAIL"
    )

    assert (
        result["diagnostic_effect"]
        == "DOWNSTREAM_LIMITATION_PERSISTS"
    )


def test_gold_context_transport_failure_is_not_model_failure():
    result = interpret_gold_context_probe(
        baseline("retrieval"),
        gold_result(
            overall="request_failed",
            transport="fail",
            evaluation="skipped",
        ),
    )

    assert (
        result["verdict"]
        == "PROBE_FAILED"
    )

    assert (
        "remains unknown"
        in result[
            "model_capability_interpretation"
        ]
    )


def test_runner_reuses_existing_rag_gold_probe_interface():
    calls = []

    def fake_runner(
        case,
        **kwargs,
    ):
        calls.append(
            {
                "case": case,
                "kwargs": kwargs,
            }
        )

        return gold_result(
            overall="pass"
        )

    result = (
        run_cognitive_gold_context_probe(
            baseline("retrieval"),
            {
                "case_id": "rag-case"
            },
            context_text="gold evidence",
            endpoint=(
                "http://127.0.0.1:8000/v1"
            ),
            model="local-model",
            probe_runner=fake_runner,
        )
    )

    assert len(calls) == 1

    assert (
        calls[0]["kwargs"][
            "context_text"
        ]
        == "gold evidence"
    )

    assert (
        result["verdict"]
        == "GOLD_CONTEXT_PASS"
    )


def test_renderer_exposes_causal_boundary():
    result = interpret_gold_context_probe(
        baseline("context"),
        gold_result(
            overall="pass"
        ),
    )

    rendered = (
        render_cognitive_gold_context(
            result
        )
    )

    assert (
        "UPSTREAM_EVIDENCE_PATH_SUPPORTED"
        in rendered
    )

    assert "Boundary:" in rendered
