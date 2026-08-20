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


def _write_replay_trace(
    path,
    *,
    route,
    source,
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
            {
                "layer": "retrieval",
                "node_type": (
                    "retriever_resources"
                ),
                "status": "succeeded",
                "source_ids": [
                    source
                ],
            },
        ],
    }

    path.write_text(
        json.dumps(trace),
        encoding="utf-8",
    )


def test_cognitive_replay_cli_validates_downstream_movement(
    tmp_path,
    capsys,
):
    case_path = tmp_path / "case.json"
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "replay-cli",
            "expected_intent": "refund",
            "expected_route": (
                "refund-route"
            ),
            "expected_sources": [
                "doc-policy"
            ],
        }),
        encoding="utf-8",
    )

    _write_replay_trace(
        before_path,
        route="wrong-route",
        source="doc-wrong",
    )

    _write_replay_trace(
        after_path,
        route="refund-route",
        source="doc-wrong",
    )

    code = main([
        "cognitive",
        "replay",
        "compare",
        "--case",
        str(case_path),
        "--before",
        str(before_path),
        "--after",
        str(after_path),
        "--target-layer",
        "route",
        "--probe-name",
        "gold_route",
    ])

    assert code == 0

    output = (
        capsys.readouterr().out
    )

    assert (
        "VALIDATED_UPSTREAM_BOTTLENECK"
        in output
    )

    assert (
        "first_broken_layer: retrieval"
        in output
    )


def test_cognitive_replay_cli_json_output(
    tmp_path,
):
    case_path = tmp_path / "case.json"
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    output_path = tmp_path / "result.json"

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "replay-json",
            "expected_intent": "refund",
            "expected_route": (
                "refund-route"
            ),
            "expected_sources": [
                "doc-policy"
            ],
        }),
        encoding="utf-8",
    )

    _write_replay_trace(
        before_path,
        route="wrong-route",
        source="doc-wrong",
    )

    _write_replay_trace(
        after_path,
        route="refund-route",
        source="doc-wrong",
    )

    code = main([
        "cognitive",
        "replay",
        "compare",
        "--case",
        str(case_path),
        "--before",
        str(before_path),
        "--after",
        str(after_path),
        "--target-layer",
        "route",
        "--probe-name",
        "gold_route",
        "--format",
        "json",
        "--output",
        str(output_path),
    ])

    assert code == 0

    result = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
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



def test_cognitive_gold_context_cli_wiring(
    tmp_path,
    monkeypatch,
    capsys,
):
    case_path = (
        tmp_path
        / "cognitive-case.json"
    )

    trace_path = (
        tmp_path
        / "cognitive-trace.json"
    )

    rag_case_path = (
        tmp_path
        / "rag-case.json"
    )

    context_path = (
        tmp_path
        / "gold.txt"
    )

    case_path.write_text(
        json.dumps({
            "schema_version": (
                COGNITIVE_CASE_SCHEMA_VERSION
            ),
            "case_id": "gold-cli",
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

    rag_case_path.write_text(
        "{}",
        encoding="utf-8",
    )

    context_path.write_text(
        "synthetic gold evidence",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "inferdoctor.cli.load_case",
        lambda path: {
            "case_id": "rag-case"
        },
    )

    calls = []

    def fake_probe(
        analysis,
        rag_case,
        **kwargs,
    ):
        calls.append({
            "analysis": analysis,
            "rag_case": rag_case,
            "kwargs": kwargs,
        })

        return {
            "verdict": "DRY_RUN",
            "confidence": "none",
            "baseline_first_broken_layer": (
                "route"
            ),
            "diagnostic_effect": (
                "NO_ATTRIBUTION_UPDATE"
            ),
            "conclusion": (
                "Dry run only."
            ),
            "model_capability_interpretation": (
                "No capability conclusion."
            ),
            "attribution_interpretation": (
                "No attribution update."
            ),
            "causal_boundary": (
                "Capability isolation only."
            ),
        }

    monkeypatch.setattr(
        "inferdoctor.cli.run_cognitive_gold_context_probe",
        fake_probe,
    )

    code = main([
        "cognitive",
        "probe",
        "gold-context",
        "--cognitive-case",
        str(case_path),
        "--cognitive-trace",
        str(trace_path),
        "--rag-case",
        str(rag_case_path),
        "--context-file",
        str(context_path),
        "--endpoint",
        "http://127.0.0.1:8000/v1",
        "--model",
        "local-model",
        "--dry-run",
    ])

    assert code == 0
    assert len(calls) == 1

    assert (
        calls[0]["kwargs"][
            "context_text"
        ]
        == "synthetic gold evidence"
    )

    assert (
        "Cognitive Gold Context Probe"
        in capsys.readouterr().out
    )
