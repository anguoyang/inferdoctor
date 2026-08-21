import copy
import json
from pathlib import Path

import pytest

from inferdoctor.cli import main
from inferdoctor.core.rag import RAG_CASE_SCHEMA_VERSION, RAG_TRACE_SCHEMA_VERSION
from inferdoctor.core.rag_gate import (
    rag_gate_exit_code,
    render_rag_gate,
    run_rag_gate,
)


def _case(case_id="fictional-handbook-case"):
    return {
        "schema_version": RAG_CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "question": "What is the fictional handbook code?",
        "language": "en",
        "category": "grounding",
        "why_bad": "The answer must include the supported fictional code.",
        "expected_sources": [
            {
                "source_id": "fictional-handbook",
                "title": "Fictional Handbook",
                "required": True,
            }
        ],
        "required_facts": [
            {
                "fact_id": "handbook-code",
                "description": "The fictional handbook code",
                "match_terms": ["HANDBOOK-7"],
                "match_mode": "exact_phrase",
            }
        ],
        "forbidden_claims": [
            {
                "claim_id": "invented-prohibition",
                "description": "An invented prohibition",
                "match_terms": ["never permitted"],
                "match_mode": "any_term",
            }
        ],
    }


def _trace(
    case_id="fictional-handbook-case",
    *,
    answer=True,
    retrieved=True,
    forbidden=False,
    conversation=False,
    redacted=False,
):
    context = "The fictional handbook code is HANDBOOK-7."
    final_answer = "The applicable code is HANDBOOK-7."
    if not answer:
        final_answer = "No supported code is available."
    if forbidden:
        final_answer = "HANDBOOK-7 applies, but it is never permitted."
    candidates = (
        [
            {
                "chunk_id": "fictional-chunk-1",
                "source_id": "fictional-handbook",
                "title": "Fictional Handbook",
                "rank": 1,
                "score": 0.9,
                "text": context,
                "text_sha256": "synthetic-hash",
                "text_length": len(context),
                "selected_for_context": True,
            }
        ]
        if retrieved
        else []
    )
    selected_ids = ["fictional-chunk-1"] if retrieved else []
    trace = {
        "schema_version": RAG_TRACE_SCHEMA_VERSION,
        "trace_id": "trace-{0}".format(case_id),
        "timestamp": "2026-01-01T00:00:00+00:00",
        "case_id": case_id,
        "system": {"name": "synthetic"},
        "pipeline": {"name": "fictional-rag", "version": "1"},
        "input": {
            "original_question": "What is the fictional handbook code?",
            "language": "en",
        },
        "retrieval": {
            "query": "fictional handbook code",
            "top_k_requested": 3,
            "candidates": candidates,
            "latency_ms": 10,
            "backend": "fixture",
            "status": "ok",
        },
        "context_selection": {
            "selected_chunk_ids": selected_ids,
            "dropped_chunk_ids": [],
            "drop_reasons": {},
            "context_text": context if retrieved else "",
            "context_length_chars": len(context) if retrieved else 0,
            "truncated": False,
        },
        "prompt": {
            "grounding_instruction_present": True,
            "citation_instruction_present": False,
            "refusal_instruction_present": True,
        },
        "generation": {
            "provider": "fixture",
            "model": "fixture-model",
            "raw_answer": final_answer,
            "streaming": False,
            "total_ms": 100,
            "status": "ok",
        },
        "postprocessing": {
            "final_answer": final_answer,
            "transformations": [],
            "status": "ok",
        },
        "timings": {"total_ms": 120},
        "privacy": {
            "content_included": True,
            "redaction_applied": False,
            "private_data_present": False,
            "export_mode": "synthetic",
        },
    }
    if conversation:
        trace["conversation"] = {
            "history_included": True,
            "possible_contamination_signals": ["synthetic prior-turn overlap"],
        }
    if redacted:
        trace["context_selection"]["context_text"] = None
        trace["context_selection"]["context_length_chars"] = 0
        trace["generation"]["raw_answer"] = None
        trace["postprocessing"]["final_answer"] = None
        trace["privacy"] = {
            "content_included": False,
            "redaction_applied": True,
            "private_data_present": False,
            "export_mode": "redacted",
        }
    return trace


def _write_dataset(root: Path, cases, before_traces, after_traces):
    root.mkdir(parents=True, exist_ok=True)
    cases_path = root / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(item) + "\n" for item in cases),
        encoding="utf-8",
    )
    before_dir = root / "baseline-artifacts"
    after_dir = root / "candidate-artifacts"
    before_dir.mkdir()
    after_dir.mkdir()
    for index, trace in enumerate(before_traces):
        (before_dir / "baseline-{0}-unrelated-name.json".format(index)).write_text(
            json.dumps(trace),
            encoding="utf-8",
        )
    for index, trace in enumerate(after_traces):
        (after_dir / "candidate-{0}-different-name.json".format(index)).write_text(
            json.dumps(trace),
            encoding="utf-8",
        )
    return cases_path, before_dir, after_dir


def _run_dataset(root, cases, before_traces, after_traces):
    paths = _write_dataset(root, cases, before_traces, after_traces)
    return run_rag_gate(*paths), paths


def test_all_unchanged_is_pass_and_matches_by_case_id(tmp_path):
    item_case = _case()
    before = _trace()
    after = copy.deepcopy(before)

    result, _paths = _run_dataset(tmp_path, [item_case], [before], [after])

    assert result["status"] == "PASS"
    assert rag_gate_exit_code(result) == 0
    assert result["summary"] == {
        "total_cases": 1,
        "improved": 0,
        "unchanged": 1,
        "regressed": 0,
        "inconclusive": 0,
        "incompatible": 0,
        "unresolved": 0,
        "input_issues": 0,
    }
    assert result["cases"][0]["verdict"] == "unchanged"


def test_improved_and_unchanged_is_pass(tmp_path):
    improved_case = _case("case-improved")
    unchanged_case = _case("case-unchanged")
    result, _paths = _run_dataset(
        tmp_path,
        [improved_case, unchanged_case],
        [_trace("case-improved", answer=False), _trace("case-unchanged")],
        [_trace("case-improved"), _trace("case-unchanged")],
    )

    assert result["status"] == "PASS"
    assert rag_gate_exit_code(result) == 0
    assert result["summary"]["improved"] == 1
    assert result["summary"]["unchanged"] == 1


def test_deterministic_regression_blocks_with_existing_attribution(tmp_path):
    result, _paths = _run_dataset(
        tmp_path,
        [_case()],
        [_trace()],
        [_trace(answer=False, retrieved=False)],
    )
    item = result["cases"][0]

    assert result["status"] == "BLOCKED"
    assert rag_gate_exit_code(result) == 1
    assert item["verdict"] == "regressed"
    assert item["after_first_broken_layer"] == "retrieval"
    assert item["first_broken_layer_changed"] is True
    assert item["after_evidence_sufficiency"] == {
        "status": "SUFFICIENT",
        "supports_first_broken_layer": True,
    }


def test_regression_projects_existing_minimal_next_probe(tmp_path):
    result, _paths = _run_dataset(
        tmp_path,
        [_case()],
        [_trace()],
        [_trace(answer=False, conversation=True)],
    )
    item = result["cases"][0]

    assert result["status"] == "BLOCKED"
    assert item["verdict"] == "regressed"
    assert item["after_first_broken_layer"] is None
    assert item["after_evidence_sufficiency"]["status"] == "PARTIAL"
    assert item["minimal_next_probe"]["probe_type"] == "CONTROLLED_REPLAY"
    assert item["minimal_next_probe"]["target_layer"] == "conversation"


@pytest.mark.parametrize(
    ("missing_side", "issue_kind"),
    (("before", "missing_before_trace"), ("after", "missing_after_trace")),
)
def test_missing_trace_is_inconclusive_exit_two(tmp_path, missing_side, issue_kind):
    before = [] if missing_side == "before" else [_trace()]
    after = [] if missing_side == "after" else [_trace()]
    result, _paths = _run_dataset(tmp_path, [_case()], before, after)

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    assert result["cases"][0]["verdict"] == "inconclusive"
    assert any(item["kind"] == issue_kind for item in result["input_issues"])


def test_duplicate_case_id_is_inconclusive(tmp_path):
    duplicate = _case()
    result, _paths = _run_dataset(
        tmp_path,
        [duplicate, copy.deepcopy(duplicate)],
        [_trace()],
        [_trace()],
    )

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    assert result["summary"]["inconclusive"] == 2
    assert any(
        item["kind"] == "duplicate_case_id" for item in result["input_issues"]
    )


def test_duplicate_trace_case_id_is_inconclusive(tmp_path):
    result, _paths = _run_dataset(
        tmp_path,
        [_case()],
        [_trace(), copy.deepcopy(_trace())],
        [_trace()],
    )

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    assert any(
        item["kind"] == "duplicate_trace_case_id"
        for item in result["input_issues"]
    )


def test_incompatible_traces_are_inconclusive(tmp_path):
    after = _trace()
    after["pipeline"] = {"name": "fictional-candidate-rag", "version": "2"}
    result, _paths = _run_dataset(tmp_path, [_case()], [_trace()], [after])

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    assert result["cases"][0]["verdict"] == "incompatible"
    assert "pipeline differs" in result["cases"][0]["compatibility_warnings"]


def test_unmatched_trace_case_id_is_reported(tmp_path):
    result, _paths = _run_dataset(
        tmp_path,
        [_case()],
        [_trace(), _trace("unmatched-fictional-case")],
        [_trace()],
    )

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    assert result["cases"][0]["verdict"] == "unchanged"
    assert any(
        item["kind"] == "mismatched_trace_case_id"
        for item in result["input_issues"]
    )


def test_regression_wins_over_inconclusive_exit_code(tmp_path):
    regressed_case = _case("case-regressed")
    missing_case = _case("case-missing-after")
    result, _paths = _run_dataset(
        tmp_path,
        [regressed_case, missing_case],
        [_trace("case-regressed"), _trace("case-missing-after")],
        [_trace("case-regressed", answer=False, retrieved=False)],
    )

    assert result["status"] == "BLOCKED"
    assert rag_gate_exit_code(result) == 1
    assert result["summary"]["regressed"] == 1
    assert result["summary"]["inconclusive"] == 1


def test_redacted_evidence_is_never_silently_passed(tmp_path):
    result, _paths = _run_dataset(
        tmp_path,
        [_case()],
        [_trace()],
        [_trace(redacted=True)],
    )

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    assert result["cases"][0]["verdict"] == "inconclusive"
    assert result["summary"]["unchanged"] == 0


def test_invalid_case_and_trace_are_input_evidence_problems(tmp_path):
    invalid_case = _case()
    invalid_case["question"] = ""
    invalid_trace = _trace()
    invalid_trace.pop("system")
    result, _paths = _run_dataset(
        tmp_path,
        [invalid_case],
        [invalid_trace],
        [copy.deepcopy(invalid_trace)],
    )

    assert result["status"] == "INCONCLUSIVE"
    assert rag_gate_exit_code(result) == 2
    kinds = {item["kind"] for item in result["input_issues"]}
    assert "invalid_case" in kinds
    assert "invalid_trace" in kinds
    assert result["cases"][0]["verdict"] == "inconclusive"


def test_json_projection_does_not_retain_raw_case_or_trace_content(tmp_path):
    item_case = _case()
    item_case["question"] = "PRIVATE-PROMPT-SENTINEL"
    before = _trace()
    after = _trace()
    for trace in (before, after):
        trace["input"]["original_question"] = "PRIVATE-PROMPT-SENTINEL"
        trace["context_selection"]["context_text"] = (
            "PRIVATE-CONTEXT-SENTINEL HANDBOOK-7"
        )
        trace["generation"]["raw_answer"] = "PRIVATE-RAW-SENTINEL HANDBOOK-7"
        trace["postprocessing"]["final_answer"] = (
            "PRIVATE-FINAL-SENTINEL HANDBOOK-7"
        )
    result, _paths = _run_dataset(tmp_path, [item_case], [before], [after])
    rendered = render_rag_gate(result, "json")

    assert result["status"] == "PASS"
    for sentinel in (
        "PRIVATE-PROMPT-SENTINEL",
        "PRIVATE-CONTEXT-SENTINEL",
        "PRIVATE-RAW-SENTINEL",
        "PRIVATE-FINAL-SENTINEL",
    ):
        assert sentinel not in rendered


def test_console_is_concise_and_deterministic(tmp_path):
    result, _paths = _run_dataset(tmp_path, [_case()], [_trace()], [_trace()])
    first = render_rag_gate(result, "console")
    second = render_rag_gate(result, "console")

    assert first == second
    assert "InferDoctor RAG Quality Gate" in first
    assert "QUALITY GATE: PASS" in first
    assert "Safe to proceed from the available evaluated Cases and evidence." in first
    assert len(first.splitlines()) < 15


def test_markdown_emphasizes_regression_attribution_and_probe(tmp_path):
    result, _paths = _run_dataset(
        tmp_path,
        [_case()],
        [_trace()],
        [_trace(answer=False, conversation=True)],
    )
    rendered = render_rag_gate(result, "markdown")

    assert "**QUALITY GATE: BLOCKED**" in rendered
    assert "## Regressions" in rendered
    assert "First broken layer: `not established`" in rendered
    assert "Minimal next probe: `CONTROLLED_REPLAY`" in rendered


def test_cli_exit_codes_and_output_file(tmp_path, capsys):
    pass_result, pass_paths = _run_dataset(
        tmp_path / "pass",
        [_case("case-pass")],
        [_trace("case-pass")],
        [_trace("case-pass")],
    )
    assert pass_result["status"] == "PASS"
    output_path = tmp_path / "gate.json"
    assert main(
        [
            "rag",
            "gate",
            "--cases",
            str(pass_paths[0]),
            "--before",
            str(pass_paths[1]),
            "--after",
            str(pass_paths[2]),
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    ) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "PASS"

    _blocked_result, blocked_paths = _run_dataset(
        tmp_path / "blocked",
        [_case("case-blocked")],
        [_trace("case-blocked")],
        [_trace("case-blocked", answer=False, retrieved=False)],
    )
    assert main(
        [
            "rag",
            "gate",
            "--cases",
            str(blocked_paths[0]),
            "--before",
            str(blocked_paths[1]),
            "--after",
            str(blocked_paths[2]),
        ]
    ) == 1
    assert "QUALITY GATE: BLOCKED" in capsys.readouterr().out

    _unknown_result, unknown_paths = _run_dataset(
        tmp_path / "inconclusive",
        [_case("case-inconclusive")],
        [_trace("case-inconclusive")],
        [],
    )
    assert main(
        [
            "rag",
            "gate",
            "--cases",
            str(unknown_paths[0]),
            "--before",
            str(unknown_paths[1]),
            "--after",
            str(unknown_paths[2]),
        ]
    ) == 2
    assert "QUALITY GATE: INCONCLUSIVE" in capsys.readouterr().out
