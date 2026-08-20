import json

from inferdoctor.cli import main
from inferdoctor.core.cognitive import (
    COGNITIVE_TRACE_SCHEMA_VERSION,
)
from inferdoctor.core.cognitive_semantics import (
    COGNITIVE_CASE_SCHEMA_VERSION,
    semantic_value_sha256,
)


def write_trace(
    path,
    *,
    route,
    wrapped=False,
):
    trace = {
        "schema_version": (
            COGNITIVE_TRACE_SCHEMA_VERSION
        ),
        "observations": [
            {
                "layer": "intent",
                "node_type": (
                    "question-classifier"
                ),
                "status": "succeeded",
                "decision_sha256": (
                    semantic_value_sha256(
                        "refund"
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
        ],
    }

    payload = (
        {
            "schema_version": (
                "inferdoctor.cognitive.capture.v1"
            ),
            "trace": trace,
            "analysis": {},
        }
        if wrapped
        else trace
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_cognitive_case_init_and_validate(
    tmp_path,
    capsys,
):
    path = tmp_path / "case.json"

    assert main([
        "cognitive",
        "case",
        "init",
        "--output",
        str(path),
    ]) == 0

    assert path.exists()

    assert main([
        "cognitive",
        "case",
        "validate",
        str(path),
    ]) == 0

    assert (
        "PASS"
        in capsys.readouterr().out
    )


def test_cognitive_diagnose_finds_route_failure(
    tmp_path,
    capsys,
):
    case_path = tmp_path / "case.json"
    trace_path = tmp_path / "trace.json"

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "case-1",
            "expected_intent": "refund",
            "expected_route": (
                "refund-route"
            ),
        }),
        encoding="utf-8",
    )

    write_trace(
        trace_path,
        route="wrong-route",
    )

    assert main([
        "cognitive",
        "diagnose",
        "--case",
        str(case_path),
        "--trace",
        str(trace_path),
    ]) == 1

    output = capsys.readouterr().out

    assert (
        "Semantic first broken layer: route"
        in output
    )


def test_cognitive_diagnose_accepts_dify_capture_envelope(
    tmp_path,
):
    case_path = tmp_path / "case.json"
    trace_path = tmp_path / "capture.json"

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "case-2",
            "expected_intent": "refund",
            "expected_route": (
                "refund-route"
            ),
        }),
        encoding="utf-8",
    )

    write_trace(
        trace_path,
        route="refund-route",
        wrapped=True,
    )

    assert main([
        "cognitive",
        "diagnose",
        "--case",
        str(case_path),
        "--trace",
        str(trace_path),
    ]) == 0


def test_cognitive_probe_next_recommends_gold_route(
    tmp_path,
    capsys,
):
    case_path = tmp_path / "case.json"
    trace_path = tmp_path / "trace.json"

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "probe-case",
            "expected_intent": "refund",
            "expected_route": (
                "refund-route"
            ),
        }),
        encoding="utf-8",
    )

    write_trace(
        trace_path,
        route="wrong-route",
    )

    assert main([
        "cognitive",
        "probe",
        "next",
        "--case",
        str(case_path),
        "--trace",
        str(trace_path),
    ]) == 0

    output = (
        capsys.readouterr().out
    )

    assert (
        "next_probe: gold_route"
        in output
    )

    assert (
        "target_layer: route"
        in output
    )


def test_cognitive_probe_next_json_output(
    tmp_path,
):
    case_path = tmp_path / "case.json"
    trace_path = tmp_path / "trace.json"
    output_path = tmp_path / "probe.json"

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "probe-json",
            "expected_intent": "refund",
            "expected_route": (
                "refund-route"
            ),
        }),
        encoding="utf-8",
    )

    write_trace(
        trace_path,
        route="wrong-route",
    )

    assert main([
        "cognitive",
        "probe",
        "next",
        "--case",
        str(case_path),
        "--trace",
        str(trace_path),
        "--format",
        "json",
        "--output",
        str(output_path),
    ]) == 0

    result = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        result["next_probe"]
        == "gold_route"
    )

    assert (
        result["target_layer"]
        == "route"
    )
