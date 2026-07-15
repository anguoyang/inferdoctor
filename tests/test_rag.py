import json
from pathlib import Path

import pytest

from inferdoctor.core import rag as rag_module

from inferdoctor.cli import main
from inferdoctor.core.rag import (
    RAG_CASE_SCHEMA_VERSION,
    RAG_TRACE_SCHEMA_VERSION,
    compare_rag,
    diagnose_rag,
    init_case_template,
    run_gold_context_probe,
    validate_case_file,
    validate_trace_file,
)


def case():
    return {
        "schema_version": RAG_CASE_SCHEMA_VERSION,
        "case_id": "synthetic-case-1",
        "question": "What is the fictional return window?",
        "language": "en",
        "category": "grounding",
        "why_bad": "The answer omitted the required fact.",
        "expected_sources": [{"source_id": "policy", "title": "Policy", "required": True}],
        "required_facts": [{"fact_id": "window", "description": "30-day return window", "match_terms": ["30 days", "return"], "match_mode": "all_terms"}],
        "forbidden_claims": [{"claim_id": "no-refund", "description": "No refund claim", "match_terms": ["no refunds"], "match_mode": "any_term"}],
    }


def trace(*, retrieved=True, selected=True, context=True, answer=True, truncated=False):
    candidates = []
    selected_ids = []
    if retrieved:
        candidates.append({"chunk_id": "c1", "source_id": "policy", "title": "Policy", "rank": 1, "score": 0.9, "text": "Returns are allowed for 30 days.", "text_sha256": "x", "text_length": 33, "selected_for_context": selected})
        if selected:
            selected_ids.append("c1")
    return {
        "schema_version": RAG_TRACE_SCHEMA_VERSION,
        "trace_id": "trace-1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "case_id": "synthetic-case-1",
        "system": {"name": "synthetic"},
        "pipeline": {"name": "synthetic-rag", "version": "1"},
        "input": {"original_question": "What is the fictional return window?", "language": "en"},
        "retrieval": {"query": "return window", "top_k_requested": 5, "candidates": candidates, "latency_ms": 10, "backend": "fixture", "status": "ok"},
        "context_selection": {"selected_chunk_ids": selected_ids, "dropped_chunk_ids": [], "drop_reasons": {}, "context_text": "Returns are allowed for 30 days." if context and selected else "", "context_length_chars": 33 if context and selected else 0, "truncated": truncated},
        "prompt": {"grounding_instruction_present": True, "citation_instruction_present": False, "refusal_instruction_present": True},
        "generation": {"provider": "fixture", "model": "fixture-model", "raw_answer": "You can return it within 30 days." if answer else "I do not know.", "streaming": False, "total_ms": 100, "status": "ok"},
        "postprocessing": {"final_answer": "You can return it within 30 days." if answer else "I do not know.", "transformations": [], "status": "ok"},
        "timings": {"total_ms": 120},
        "privacy": {"content_included": True, "redaction_applied": False, "private_data_present": False, "export_mode": "synthetic"},
    }


def write(path: Path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_case_init_and_validate(tmp_path):
    path = tmp_path / "cases.jsonl"
    init_case_template(path)
    result = validate_case_file(path)
    assert result["status"] == "PASS"
    assert result["case_count"] == 1


def test_case_validation_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    bad = dict(case(), case_id="", required_facts=[{"fact_id": "x", "match_mode": "bad", "match_terms": ["x"]}])
    path.write_text(json.dumps(bad), encoding="utf-8")
    result = validate_case_file(path)
    assert result["status"] == "FAIL"
    assert any(item["field"] == "case_id" for item in result["findings"])


def test_trace_validate(tmp_path):
    path = write(tmp_path / "trace.json", trace())
    assert validate_trace_file(path)["status"] == "PASS"


def test_trace_validate_pretty_json(tmp_path):
    path = tmp_path / "trace-pretty.json"
    path.write_text(json.dumps(trace(), indent=2), encoding="utf-8")
    assert validate_trace_file(path)["status"] == "PASS"


def test_trace_validate_allows_explicit_private_content_export(tmp_path):
    item = trace()
    item["privacy"] = {
        "content_included": True,
        "redaction_applied": False,
        "private_data_present": True,
        "export_mode": "include_content",
    }
    path = write(tmp_path / "trace-private.json", item)
    assert validate_trace_file(path)["status"] == "PASS"


def test_trace_validate_rejects_unredacted_private_content_without_explicit_export(tmp_path):
    item = trace()
    item["privacy"] = {
        "content_included": False,
        "redaction_applied": False,
        "private_data_present": True,
        "export_mode": "redacted",
    }
    path = write(tmp_path / "trace-private-bad.json", item)
    assert validate_trace_file(path)["status"] == "FAIL"


def test_trace_validate_allows_redacted_question_hash(tmp_path):
    item = trace()
    item["input"] = {"original_question": None, "original_question_sha256": "a" * 64, "language": "unknown"}
    item["privacy"] = {"content_included": False, "redaction_applied": True, "private_data_present": False, "export_mode": "redacted"}
    path = write(tmp_path / "trace-redacted-question.json", item)
    assert validate_trace_file(path)["status"] == "PASS"


def test_diagnose_reports_evidence_completeness():
    result = diagnose_rag(case(), trace())
    assert "evidence_completeness" in result
    assert "retrieval_candidates" in result["evidence_completeness"]["available"]


def test_diagnose_retrieval_failure():
    result = diagnose_rag(case(), trace(retrieved=False))
    assert result["status"] == "FAIL"
    assert result["diagnoses"][0]["category"] == "retrieval_failure"


def test_diagnose_context_selection_failure():
    result = diagnose_rag(case(), trace(retrieved=True, selected=False))
    assert any(item["category"] == "context_selection_failure" for item in result["diagnoses"])


def test_diagnose_postprocessing_failure():
    tr = trace(answer=True)
    tr["postprocessing"]["final_answer"] = "I do not know."
    result = diagnose_rag(case(), tr)
    assert any(item["category"] == "answer_postprocessing_failure" for item in result["diagnoses"])


def test_compare_improved_and_incompatible():
    before = trace(retrieved=False, answer=False)
    after = trace(retrieved=True, selected=True, answer=True)
    assert compare_rag(case(), before, after)["verdict"] == "improved"
    changed = dict(after)
    changed["pipeline"] = {"name": "other"}
    assert compare_rag(case(), before, changed)["verdict"] == "incompatible"


def test_gold_context_probe_dry_run():
    result = run_gold_context_probe(case(), context_text="Returns are allowed for 30 days.", endpoint="http://127.0.0.1:8000/v1", model="fixture", dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["request_sent"] is False
    assert result["required_fact_checks"]["matched"] == 1

class _FakeOpenAIResponse:
    def __init__(self, answer: str):
        self.answer = answer

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _limit: int) -> bytes:
        return json.dumps({"choices": [{"message": {"content": self.answer}}]}).encode("utf-8")


def _gold_case(required_facts, forbidden_claims=None):
    item = case()
    item["required_facts"] = required_facts
    item["forbidden_claims"] = forbidden_claims or []
    return item


def _mock_gold_probe(monkeypatch, answer: str, probe_case=None):
    monkeypatch.setattr(rag_module.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeOpenAIResponse(answer))
    return run_gold_context_probe(
        probe_case or case(),
        context_text="Synthetic gold context.",
        endpoint="http://127.0.0.1:8000/v1",
        model="fixture",
    )


def test_gold_context_probe_transport_pass_evaluation_pass(monkeypatch):
    probe_case = _gold_case([
        {"fact_id": "return-code", "description": "Return policy code", "match_terms": ["RETURN-14"], "match_mode": "exact_phrase"},
        {"fact_id": "defect-code", "description": "Defect policy code", "match_terms": ["DEFECT-30"], "match_mode": "exact_phrase"},
    ])
    result = _mock_gold_probe(monkeypatch, "Use RETURN-14 and DEFECT-30.", probe_case)
    assert result["transport_status"] == "pass"
    assert result["evaluation_status"] == "pass"
    assert result["overall_status"] == "pass"
    assert result["status"] == "PASS"
    assert result["required_facts_total"] == 2
    assert result["required_facts_matched"] == 2
    assert result["answer_retained"] is False
    assert result["answer_evaluated_in_memory"] is True
    assert result["answer_preview"] is None


def test_gold_context_probe_transport_pass_evaluation_fail(monkeypatch):
    probe_case = _gold_case([
        {"fact_id": "return-code", "description": "Return policy code", "match_terms": ["RETURN-14"], "match_mode": "exact_phrase"},
        {"fact_id": "defect-code", "description": "Defect policy code", "match_terms": ["DEFECT-30"], "match_mode": "exact_phrase"},
    ])
    result = _mock_gold_probe(monkeypatch, "Use RETURN-14 only.", probe_case)
    assert result["transport_status"] == "pass"
    assert result["evaluation_status"] == "fail"
    assert result["overall_status"] == "fail"
    assert result["status"] == "FAIL"
    assert result["required_facts_matched"] == 1
    assert result["required_fact_checks"]["results"][1]["missing_terms"] == ["DEFECT-30"]


def test_gold_context_probe_transport_fail(monkeypatch):
    def fail(*_args, **_kwargs):
        raise rag_module.urllib.error.URLError("connection refused")

    monkeypatch.setattr(rag_module.urllib.request, "urlopen", fail)
    result = run_gold_context_probe(case(), context_text="Synthetic context.", endpoint="http://127.0.0.1:8000/v1", model="fixture")
    assert result["transport_status"] == "fail"
    assert result["evaluation_status"] == "skipped"
    assert result["overall_status"] == "request_failed"
    assert result["status"] == "FAIL"


def test_gold_context_probe_human_review_only_is_inconclusive(monkeypatch):
    probe_case = _gold_case([
        {"fact_id": "semantic", "description": "Needs semantic review", "match_terms": [], "match_mode": "human_review"},
    ])
    result = _mock_gold_probe(monkeypatch, "A plausible answer.", probe_case)
    assert result["evaluation_status"] == "inconclusive"
    assert result["overall_status"] == "inconclusive"
    assert result["status"] == "INCONCLUSIVE"
    assert result["human_review_required"] is True


def test_gold_context_probe_mixed_human_review_requires_review(monkeypatch):
    probe_case = _gold_case([
        {"fact_id": "code", "description": "Policy code", "match_terms": ["RETURN-14"], "match_mode": "exact_phrase"},
        {"fact_id": "semantic", "description": "Needs semantic review", "match_terms": [], "match_mode": "human_review"},
    ])
    result = _mock_gold_probe(monkeypatch, "Use RETURN-14.", probe_case)
    assert result["evaluation_status"] == "pass"
    assert result["overall_status"] == "inconclusive"
    assert result["review_status"] == "required"


def test_gold_context_probe_forbidden_claim_present(monkeypatch):
    probe_case = _gold_case(
        [{"fact_id": "code", "description": "Policy code", "match_terms": ["RETURN-14"], "match_mode": "exact_phrase"}],
        [{"claim_id": "wrong", "description": "Wrong no-refund claim", "match_terms": ["no refunds"], "match_mode": "any_term"}],
    )
    result = _mock_gold_probe(monkeypatch, "Use RETURN-14. There are no refunds.", probe_case)
    assert result["evaluation_status"] == "fail"
    assert result["overall_status"] == "fail"
    assert result["forbidden_claims_matched"] == 1


def test_gold_context_probe_normalizes_unicode_punctuation_and_whitespace(monkeypatch):
    probe_case = _gold_case([
        {"fact_id": "return-code", "description": "Return policy code", "match_terms": ["RETURN-14"], "match_mode": "exact_phrase"},
        {"fact_id": "defect-terms", "description": "Defect terms", "match_terms": ["defective", "30 days"], "match_mode": "all_terms"},
        {"fact_id": "any-term", "description": "Any term", "match_terms": ["missing", "opened"], "match_mode": "any_term"},
    ])
    result = _mock_gold_probe(monkeypatch, "Policy RETURN－14 applies. Defective items: 30   days when opened.", probe_case)
    assert result["overall_status"] == "pass"
    assert result["required_facts_matched"] == 3


def test_rag_cli_smoke(tmp_path, capsys):
    case_path = write(tmp_path / "case.json", case())
    before = write(tmp_path / "before.json", trace(retrieved=False, answer=False))
    after = write(tmp_path / "after.json", trace())
    assert main(["rag", "case", "validate", str(case_path)]) == 0
    assert "RAG Case Validation" in capsys.readouterr().out
    assert main(["rag", "trace", "validate", str(after)]) == 0
    assert main(["rag", "diagnose", "--case", str(case_path), "--trace", str(before)]) == 1
    assert "retrieval_failure" in capsys.readouterr().out
    assert main(["rag", "compare", "--case", str(case_path), "--before", str(before), "--after", str(after)]) == 0
    assert "improved" in capsys.readouterr().out
    context = tmp_path / "gold.md"
    context.write_text("Returns are allowed for 30 days.", encoding="utf-8")
    assert main(["rag", "probe", "gold-context", "--case", str(case_path), "--context-file", str(context), "--endpoint", "http://127.0.0.1:8000/v1", "--model", "fixture", "--dry-run"]) == 0


def test_rag_cli_help(capsys):
    for args in (["rag", "--help"], ["rag", "case", "--help"], ["rag", "probe", "gold-context", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out
