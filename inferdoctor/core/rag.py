from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from inferdoctor import __version__
from inferdoctor.core.endpoint_safety import classify_endpoint, redact_endpoint

RAG_CASE_SCHEMA_VERSION = "inferdoctor.rag.case.v1"
RAG_TRACE_SCHEMA_VERSION = "inferdoctor.rag.trace.v1"
RAG_DIAGNOSIS_SCHEMA_VERSION = "inferdoctor.rag.diagnosis.v1"
RAG_COMPARISON_SCHEMA_VERSION = "inferdoctor.rag.comparison.v1"
RAG_GOLD_PROBE_SCHEMA_VERSION = "inferdoctor.rag.gold_context_probe.v1"

MATCH_MODES = {"all_terms", "any_term", "exact_phrase", "human_review"}
MAX_FIELD_CHARS = 20000


class RagError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: str | Path, *, limit: int = 2 * 1024 * 1024) -> str:
    data = Path(path).read_bytes()
    if len(data) > limit:
        raise RagError("file is too large for a bounded RAG diagnostic input")
    return data.decode("utf-8")


def _load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    text = _read_text(path)
    stripped = text.strip()
    if not stripped:
        raise RagError("input file is empty")
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        if stripped.startswith("["):
            raise RagError("JSON array input must contain objects")
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RagError(f"line {line_no} is not a JSON object")
        rows.append(parsed)
    if not rows:
        raise RagError("JSONL input contained no objects")
    return rows


def _single_json(path: str | Path) -> Dict[str, Any]:
    items = _load_json_or_jsonl(path)
    if len(items) != 1:
        raise RagError("expected exactly one JSON object")
    return items[0]


def _finding(status: str, field: str, message: str) -> Dict[str, str]:
    return {"status": status, "field": field, "message": message}


def init_case_template(output: str | Path) -> Path:
    example = {
        "schema_version": RAG_CASE_SCHEMA_VERSION,
        "case_id": "fictional-return-policy-001",
        "question": "What is the return window for the fictional blue widget?",
        "language": "en",
        "category": "retrieval-grounding",
        "why_bad": "The current answer omits the return window even though the policy exists.",
        "current_answer": "",
        "expected_answer": "The fictional blue widget can be returned within 30 days when the receipt is available.",
        "expected_sources": [
            {
                "source_id": "fictional-policy",
                "title": "Fictional Support Policy",
                "section": "Returns",
                "locator": "sample://fictional-support-policy#returns",
                "required": True,
                "notes": "Synthetic example only.",
            }
        ],
        "required_facts": [
            {
                "fact_id": "return-window",
                "description": "The answer should mention the 30-day return window.",
                "match_terms": ["30 days", "return"],
                "match_mode": "all_terms",
                "notes": "",
            }
        ],
        "forbidden_claims": [
            {
                "claim_id": "no-refund",
                "description": "The answer must not claim refunds are impossible.",
                "match_terms": ["no refunds", "cannot be returned"],
                "match_mode": "any_term",
                "notes": "",
            }
        ],
        "expected_behavior": "Answer only from the provided policy and say when evidence is missing.",
        "metadata": {"fixture": True},
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def validate_case_object(case: Dict[str, Any], seen_ids: Optional[set[str]] = None) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    if case.get("schema_version") != RAG_CASE_SCHEMA_VERSION:
        findings.append(_finding("FAIL", "schema_version", "unknown or missing RAG Case schema version"))
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        findings.append(_finding("FAIL", "case_id", "missing case_id"))
    elif seen_ids is not None:
        if case_id in seen_ids:
            findings.append(_finding("FAIL", "case_id", f"duplicate case_id: {case_id}"))
        seen_ids.add(case_id)
    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        findings.append(_finding("FAIL", "question", "question must be a non-empty string"))
    for field in ("question", "current_answer", "expected_answer", "why_bad"):
        value = case.get(field)
        if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
            findings.append(_finding("FAIL", field, "field is oversized"))
    if not isinstance(case.get("language"), str) or not case.get("language"):
        findings.append(_finding("FAIL", "language", "language is required"))
    if not isinstance(case.get("category"), str) or not case.get("category"):
        findings.append(_finding("FAIL", "category", "category is required"))
    if not isinstance(case.get("why_bad"), str) or not case.get("why_bad"):
        findings.append(_finding("FAIL", "why_bad", "why_bad is required"))
    _validate_sources(case.get("expected_sources", []), findings)
    _validate_claims(case.get("required_facts", []), "required_facts", findings)
    _validate_claims(case.get("forbidden_claims", []), "forbidden_claims", findings)
    _validate_conflicts(case, findings)
    return findings


def _validate_sources(sources: Any, findings: List[Dict[str, str]]) -> None:
    if sources is None:
        return
    if not isinstance(sources, list):
        findings.append(_finding("FAIL", "expected_sources", "expected_sources must be a list"))
        return
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            findings.append(_finding("FAIL", f"expected_sources[{index}]", "source must be an object"))
            continue
        if not isinstance(source.get("source_id"), str) or not source.get("source_id"):
            findings.append(_finding("FAIL", f"expected_sources[{index}].source_id", "source_id is required"))
        if source.get("required") is not None and not isinstance(source.get("required"), bool):
            findings.append(_finding("FAIL", f"expected_sources[{index}].required", "required must be boolean"))


def _validate_claims(claims: Any, field: str, findings: List[Dict[str, str]]) -> None:
    if claims is None:
        return
    if not isinstance(claims, list):
        findings.append(_finding("FAIL", field, f"{field} must be a list"))
        return
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            findings.append(_finding("FAIL", f"{field}[{index}]", "claim/fact must be an object"))
            continue
        mode = claim.get("match_mode")
        if mode not in MATCH_MODES:
            findings.append(_finding("FAIL", f"{field}[{index}].match_mode", "invalid match mode"))
        terms = claim.get("match_terms", [])
        if mode != "human_review" and (not isinstance(terms, list) or not all(isinstance(term, str) and term for term in terms)):
            findings.append(_finding("FAIL", f"{field}[{index}].match_terms", "match_terms are required for deterministic match modes"))


def _validate_conflicts(case: Dict[str, Any], findings: List[Dict[str, str]]) -> None:
    required_terms = {term.lower() for fact in case.get("required_facts", []) if isinstance(fact, dict) for term in fact.get("match_terms", []) if isinstance(term, str)}
    forbidden_terms = {term.lower() for fact in case.get("forbidden_claims", []) if isinstance(fact, dict) for term in fact.get("match_terms", []) if isinstance(term, str)}
    overlap = required_terms & forbidden_terms
    if overlap:
        findings.append(_finding("FAIL", "required_facts/forbidden_claims", "conflicting required and forbidden terms: " + ", ".join(sorted(overlap))))


def validate_case_file(path: str | Path) -> Dict[str, Any]:
    cases = _load_json_or_jsonl(path)
    seen: set[str] = set()
    findings: List[Dict[str, str]] = []
    for case in cases:
        findings.extend(validate_case_object(case, seen))
    status = "PASS" if not any(item["status"] == "FAIL" for item in findings) else "FAIL"
    return {"schema_version": "inferdoctor.rag.case.validation.v1", "status": status, "case_count": len(cases), "findings": findings}


def validate_trace_object(trace: Dict[str, Any]) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    if trace.get("schema_version") != RAG_TRACE_SCHEMA_VERSION:
        findings.append(_finding("FAIL", "schema_version", "unknown or missing RAG Trace schema version"))
    for field in ("trace_id", "timestamp", "system", "pipeline", "input", "retrieval", "context_selection", "generation", "postprocessing", "timings", "privacy"):
        if field not in trace:
            findings.append(_finding("FAIL", field, f"{field} is required"))
    input_data = trace.get("input") if isinstance(trace.get("input"), dict) else {}
    has_question_text = isinstance(input_data.get("original_question"), str) and bool(input_data.get("original_question"))
    has_question_hash = isinstance(input_data.get("original_question_sha256"), str) and len(str(input_data.get("original_question_sha256"))) >= 16
    if not has_question_text and not has_question_hash:
        findings.append(_finding("FAIL", "input.original_question", "original_question or original_question_sha256 is required"))
    retrieval = trace.get("retrieval")
    if isinstance(retrieval, dict):
        candidates = retrieval.get("candidates", [])
        if not isinstance(candidates, list):
            findings.append(_finding("FAIL", "retrieval.candidates", "candidates must be a list"))
        else:
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    findings.append(_finding("FAIL", f"retrieval.candidates[{index}]", "candidate must be an object"))
                    continue
                if "chunk_id" not in candidate or "rank" not in candidate:
                    findings.append(_finding("FAIL", f"retrieval.candidates[{index}]", "candidate requires chunk_id and rank"))
    privacy = trace.get("privacy")
    if isinstance(privacy, dict):
        explicit_content_export = privacy.get("content_included") is True and str(privacy.get("export_mode") or "").lower() in {"include_content", "explicit_content", "synthetic"}
        if privacy.get("private_data_present") is True and privacy.get("redaction_applied") is not True and not explicit_content_export:
            findings.append(_finding("FAIL", "privacy", "private_data_present requires redaction_applied unless explicitly exporting private content"))
    return findings


def validate_trace_file(path: str | Path) -> Dict[str, Any]:
    trace = _single_json(path)
    findings = validate_trace_object(trace)
    status = "PASS" if not any(item["status"] == "FAIL" for item in findings) else "FAIL"
    return {"schema_version": "inferdoctor.rag.trace.validation.v1", "status": status, "findings": findings}


def _terms_match(text: str, terms: Sequence[str], mode: str) -> Optional[bool]:
    lower = text.lower()
    if mode == "human_review":
        return None
    if mode == "exact_phrase":
        return all(term.lower() in lower for term in terms)
    if mode == "all_terms":
        return all(term.lower() in lower for term in terms)
    if mode == "any_term":
        return any(term.lower() in lower for term in terms)
    return False


def _candidate_source_ids(trace: Dict[str, Any]) -> set[str]:
    return {str(candidate.get("source_id")) for candidate in trace.get("retrieval", {}).get("candidates", []) if isinstance(candidate, dict) and candidate.get("source_id")}


def _selected_candidate_ids(trace: Dict[str, Any]) -> set[str]:
    selected = trace.get("context_selection", {}).get("selected_chunk_ids", [])
    return {str(item) for item in selected if item is not None}


def _candidate_by_chunk(trace: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(candidate.get("chunk_id")): candidate for candidate in trace.get("retrieval", {}).get("candidates", []) if isinstance(candidate, dict) and candidate.get("chunk_id")}


def _context_text(trace: Dict[str, Any]) -> str:
    return str(trace.get("context_selection", {}).get("context_text") or "")


def _raw_answer(trace: Dict[str, Any]) -> str:
    return str(trace.get("generation", {}).get("raw_answer") or "")


def _final_answer(trace: Dict[str, Any]) -> str:
    return str(trace.get("postprocessing", {}).get("final_answer") or "")


def _required_fact_coverage(case: Dict[str, Any], text: str) -> Dict[str, Any]:
    results = []
    deterministic = 0
    matched = 0
    human_review = 0
    for fact in case.get("required_facts", []) or []:
        if not isinstance(fact, dict):
            continue
        mode = fact.get("match_mode", "human_review")
        terms = fact.get("match_terms", [])
        outcome = _terms_match(text, terms, mode)
        if outcome is None:
            human_review += 1
        else:
            deterministic += 1
            matched += 1 if outcome else 0
        results.append({"fact_id": fact.get("fact_id"), "match_mode": mode, "matched": outcome})
    return {"matched": matched, "deterministic": deterministic, "human_review": human_review, "results": results}


def _forbidden_claims(case: Dict[str, Any], text: str) -> Dict[str, Any]:
    hits = []
    human_review = 0
    for claim in case.get("forbidden_claims", []) or []:
        if not isinstance(claim, dict):
            continue
        outcome = _terms_match(text, claim.get("match_terms", []), claim.get("match_mode", "human_review"))
        if outcome is None:
            human_review += 1
        elif outcome:
            hits.append(claim.get("claim_id"))
    return {"hits": hits, "human_review": human_review}



def _trace_evidence_summary(trace: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "question_text_or_hash": bool(trace.get("input", {}).get("original_question") or trace.get("input", {}).get("original_question_sha256")),
        "retrieval_candidates": bool(trace.get("retrieval", {}).get("candidates")),
        "retrieval_latency": trace.get("retrieval", {}).get("latency_ms") is not None,
        "rerank_status": bool(trace.get("rerank", {}).get("status")) if isinstance(trace.get("rerank"), dict) else False,
        "context_text_or_hash": bool(trace.get("context_selection", {}).get("context_text") or trace.get("context_selection", {}).get("context_sha256")),
        "context_budget": trace.get("context_selection", {}).get("context_budget") is not None,
        "prompt_hash": bool(trace.get("prompt", {}).get("prompt_sha256")),
        "grounding_signal": trace.get("prompt", {}).get("grounding_instruction_present") is not None,
        "raw_answer_hash_or_text": bool(trace.get("generation", {}).get("raw_answer") or trace.get("generation", {}).get("raw_answer_sha256")),
        "final_answer_hash_or_text": bool(trace.get("postprocessing", {}).get("final_answer") or trace.get("postprocessing", {}).get("final_answer_sha256")),
        "ttft": trace.get("generation", {}).get("ttft_ms") is not None,
        "token_usage": bool(trace.get("generation", {}).get("token_usage")),
        "conversation_metadata": trace.get("conversation", {}).get("history_included") is not None,
        "stage_events": bool(trace.get("stage_events")),
    }
    return {
        "available": [key for key, value in checks.items() if value],
        "missing_or_redacted": [key for key, value in checks.items() if not value],
    }


def diagnose_rag(case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    case_findings = validate_case_object(case)
    trace_findings = validate_trace_object(trace)
    diagnoses: List[Dict[str, Any]] = []

    def add(category: str, status: str, evidence: Sequence[str], strength: str, confidence: str, next_experiment: str, known: Sequence[str] = (), unknown: Sequence[str] = ()) -> None:
        diagnoses.append({
            "category": category,
            "status": status,
            "evidence": list(evidence),
            "evidence_strength": strength,
            "confidence": confidence,
            "what_is_known": list(known),
            "what_is_not_known": list(unknown),
            "downstream_effects": [],
            "next_experiment": next_experiment,
            "unsafe_conclusions_to_avoid": ["Do not call the model stupid before isolating retrieval, context, prompt, and post-processing evidence."],
        })

    if any(item["status"] == "FAIL" for item in case_findings + trace_findings):
        add("insufficient_evidence", "warn", [item["message"] for item in case_findings + trace_findings if item["status"] == "FAIL"], "observed", "high", "Fix the Case/Trace schema errors and rerun diagnosis.")
    source_ids = _candidate_source_ids(trace)
    selected_ids = _selected_candidate_ids(trace)
    by_chunk = _candidate_by_chunk(trace)
    selected_sources = {str(by_chunk[chunk].get("source_id")) for chunk in selected_ids if chunk in by_chunk}
    required_sources = [source for source in case.get("expected_sources", []) or [] if isinstance(source, dict) and source.get("required", True)]
    missing_sources = [str(source.get("source_id")) for source in required_sources if source.get("source_id") not in source_ids]
    if missing_sources:
        add("retrieval_failure", "fail", [f"Required source not retrieved: {source_id}" for source_id in missing_sources], "observed", "high", "Probe retrieval with the exact expected source terms and inspect indexing/source availability.", known=["Expected source was absent from retrieval candidates."])
    not_selected = [str(source.get("source_id")) for source in required_sources if source.get("source_id") in source_ids and source.get("source_id") not in selected_sources]
    if not_selected:
        add("context_selection_failure", "fail", [f"Required source retrieved but not selected into final context: {source_id}" for source_id in not_selected], "observed", "high", "Inspect ranking, rerank, context cutoff, and selected_chunk_ids.")
    context = _context_text(trace)
    context_coverage = _required_fact_coverage(case, context)
    final_coverage = _required_fact_coverage(case, _final_answer(trace) or _raw_answer(trace))
    if trace.get("context_selection", {}).get("truncated") and context_coverage["deterministic"] and context_coverage["matched"] < context_coverage["deterministic"]:
        add("context_truncation", "fail", ["Context was marked truncated and required facts were absent from final context."], "observed", "medium", "Run the same case with larger context budget or inspect truncation_detail.", unknown=["Whether the missing fact was present before truncation unless dropped chunks retain content or hashes."])
    prompt = trace.get("prompt", {}) if isinstance(trace.get("prompt"), dict) else {}
    if context and context_coverage["matched"] == context_coverage["deterministic"] and final_coverage["deterministic"] and final_coverage["matched"] < final_coverage["deterministic"]:
        if prompt.get("grounding_instruction_present") is False:
            add("prompt_grounding_failure", "warn", ["Correct evidence appears in context but grounding instruction is absent."], "strongly_indicated", "medium", "Run a Gold Context Probe with an explicit grounding prompt.")
        else:
            add("insufficient_evidence", "warn", ["Correct evidence appears in context but answer omits required facts."], "possible", "medium", "Run Gold Context Probe before blaming model reasoning.")
    raw = _raw_answer(trace)
    final = _final_answer(trace)
    raw_cov = _required_fact_coverage(case, raw)
    final_cov = _required_fact_coverage(case, final)
    if raw and final and raw_cov["matched"] > final_cov["matched"]:
        add("answer_postprocessing_failure", "fail", ["Raw answer covered more required facts than final answer."], "observed", "high", "Inspect postprocessing transformations and answer assembly.")
    conversation = trace.get("conversation", {}) if isinstance(trace.get("conversation"), dict) else {}
    if conversation.get("history_included") and conversation.get("possible_contamination_signals"):
        add("conversation_memory_contamination", "warn", ["Conversation history was included and contamination signals were reported."], "possible", "medium", "Replay the same question as a single-turn trace.")
    if not diagnoses:
        add("no_clear_failure", "pass", ["No deterministic failure was identified from available evidence."], "observed", "medium", "Add more trace fields or run Gold Context Probe if the answer is still unacceptable.")
    evidence_summary = _trace_evidence_summary(trace)
    missing_count = len(evidence_summary["missing_or_redacted"])
    evidence_score = max(0, 100 - 12 * len([d for d in diagnoses if d["category"] == "insufficient_evidence"]) - min(40, missing_count * 3))
    return {"schema_version": RAG_DIAGNOSIS_SCHEMA_VERSION, "timestamp": utc_now(), "inferdoctor_version": __version__, "case_id": case.get("case_id"), "trace_id": trace.get("trace_id"), "status": "FAIL" if any(d["status"] == "fail" for d in diagnoses) else "WARN" if any(d["status"] == "warn" for d in diagnoses) else "PASS", "evidence_completeness_score": evidence_score, "evidence_completeness": evidence_summary, "diagnoses": diagnoses, "required_fact_coverage": {"context": context_coverage, "final_answer": final_coverage}, "forbidden_claims": _forbidden_claims(case, final or raw)}


def compare_rag(case: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    compatibility: List[str] = []
    if before.get("case_id") and after.get("case_id") and before.get("case_id") != after.get("case_id"):
        compatibility.append("case IDs differ")
    before_q = before.get("input", {}).get("original_question")
    after_q = after.get("input", {}).get("original_question")
    if before_q and after_q and before_q != after_q:
        compatibility.append("questions differ")
    if before.get("pipeline") != after.get("pipeline"):
        compatibility.append("pipeline differs")
    if before.get("generation", {}).get("model") != after.get("generation", {}).get("model"):
        compatibility.append("model differs")
    before_diag = diagnose_rag(case, before)
    after_diag = diagnose_rag(case, after)
    before_context = before_diag["required_fact_coverage"]["context"]
    after_context = after_diag["required_fact_coverage"]["context"]
    before_final = before_diag["required_fact_coverage"]["final_answer"]
    after_final = after_diag["required_fact_coverage"]["final_answer"]
    changes = {
        "context_required_fact_delta": after_context["matched"] - before_context["matched"],
        "final_required_fact_delta": after_final["matched"] - before_final["matched"],
        "forbidden_claim_delta": len(after_diag["forbidden_claims"]["hits"]) - len(before_diag["forbidden_claims"]["hits"]),
        "retrieval_latency_ms_delta": _num(after, "retrieval", "latency_ms") - _num(before, "retrieval", "latency_ms"),
        "generation_total_ms_delta": _num(after, "generation", "total_ms") - _num(before, "generation", "total_ms"),
    }
    if compatibility:
        verdict = "incompatible"
    elif changes["final_required_fact_delta"] > 0 and changes["forbidden_claim_delta"] <= 0:
        verdict = "improved"
    elif changes["final_required_fact_delta"] < 0 or changes["forbidden_claim_delta"] > 0:
        verdict = "regressed"
    elif all(value == 0 for value in changes.values()):
        verdict = "unchanged"
    else:
        verdict = "inconclusive"
    return {"schema_version": RAG_COMPARISON_SCHEMA_VERSION, "timestamp": utc_now(), "inferdoctor_version": __version__, "case_id": case.get("case_id"), "verdict": verdict, "compatibility_warnings": compatibility, "changes": changes, "before_status": before_diag["status"], "after_status": after_diag["status"], "confidence": "low" if compatibility else "medium"}


def _num(data: Dict[str, Any], section: str, key: str) -> float:
    value = data.get(section, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _chat_payload(case: Dict[str, Any], context: str, *, retain_answer: bool) -> Dict[str, Any]:
    prompt = (
        "Use only the provided gold context. If the answer is missing, say the evidence is missing.\n\n"
        f"Question:\n{case.get('question')}\n\nGold context:\n{context}\n"
    )
    return {
        "prompt": prompt,
        "prompt_hash": sha256_text(prompt),
        "messages": [
            {"role": "system", "content": "Answer only from the supplied context and avoid unsupported claims."},
            {"role": "user", "content": prompt},
        ],
        "retain_answer": retain_answer,
    }


def run_gold_context_probe(case: Dict[str, Any], *, context_text: str, endpoint: str, model: str, timeout: float = 30.0, dry_run: bool = False, allow_non_local: bool = False, allow_public: bool = False, api_key: Optional[str] = None, retain_answer: bool = False) -> Dict[str, Any]:
    safety = classify_endpoint(endpoint)
    if safety.category == "invalid":
        raise RagError("invalid endpoint URL")
    if urllib.parse.urlsplit(endpoint).username or urllib.parse.urlsplit(endpoint).password:
        raise RagError("endpoint URL credentials are not allowed")
    if safety.category == "private" and not allow_non_local:
        raise RagError("LAN/private endpoint requires --allow-non-local")
    if safety.category == "public" and not allow_public:
        raise RagError("public endpoint requires --allow-public")
    payload_info = _chat_payload(case, context_text, retain_answer=retain_answer)
    result: Dict[str, Any] = {
        "schema_version": RAG_GOLD_PROBE_SCHEMA_VERSION,
        "timestamp": utc_now(),
        "inferdoctor_version": __version__,
        "case_id": case.get("case_id"),
        "endpoint": redact_endpoint(endpoint),
        "endpoint_category": safety.category,
        "model": model,
        "context_hash": sha256_text(context_text),
        "context_length_chars": len(context_text),
        "prompt_hash": payload_info["prompt_hash"],
        "request_sent": False,
        "status": "DRY_RUN" if dry_run else "UNKNOWN",
        "ttft_ms": None,
        "total_latency_ms": None,
        "generated_answer_sha256": None,
        "answer_preview": None,
        "required_fact_checks": _required_fact_coverage(case, context_text),
        "forbidden_claim_checks": _forbidden_claims(case, context_text),
        "diagnostic_interpretation": "Dry run only; no model request was sent." if dry_run else "",
    }
    if dry_run:
        return result
    body = json.dumps({"model": model, "messages": payload_info["messages"], "stream": False, "max_tokens": 256}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": f"InferDoctor/{__version__}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    try:
        request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body, method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise RagError("gold-context probe response exceeded 1 MiB")
            parsed = json.loads(data.decode("utf-8"))
        elapsed = int((time.perf_counter() - started) * 1000)
        answer = _extract_chat_answer(parsed)
        result.update({
            "request_sent": True,
            "status": "PASS",
            "total_latency_ms": elapsed,
            "generated_answer_sha256": sha256_text(answer) if answer else None,
            "answer_preview": answer[:240] if retain_answer else None,
            "required_fact_checks": _required_fact_coverage(case, answer),
            "forbidden_claim_checks": _forbidden_claims(case, answer),
            "diagnostic_interpretation": "Gold context probe completed. Use deterministic fact checks and human review where required.",
        })
        return result
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RagError) as exc:
        result.update({"request_sent": True, "status": "FAIL", "total_latency_ms": int((time.perf_counter() - started) * 1000), "error": str(exc)[:300], "diagnostic_interpretation": "Gold context probe failed before a usable answer was produced."})
        return result


def _extract_chat_answer(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    return ""


def render_rag_result(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if output_format == "markdown":
        return _render_markdown(result)
    return _render_console(result)


def _render_console(result: Dict[str, Any]) -> str:
    schema = str(result.get("schema_version", ""))
    title = "RAG Intelligence Report"
    if "case.validation" in schema:
        title = "RAG Case Validation"
    elif "trace.validation" in schema:
        title = "RAG Trace Validation"
    elif "diagnosis" in schema:
        title = "RAG Failure Diagnosis"
    elif "comparison" in schema:
        title = "RAG Before/After Comparison"
    elif "gold_context_probe" in schema:
        title = "RAG Gold Context Probe"
    lines = [title, "=" * len(title)]
    for key in ("status", "verdict", "case_count", "case_id", "trace_id"):
        if result.get(key) is not None:
            lines.append(f"{key}: {result.get(key)}")
    if result.get("findings"):
        lines.append("")
        for item in result["findings"][:20]:
            lines.append(f"- {item.get('status')} {item.get('field')}: {item.get('message')}")
    if result.get("diagnoses"):
        lines.append("")
        for item in result["diagnoses"]:
            lines.append(f"- {item.get('status').upper()} {item.get('category')}: {item.get('evidence')[0] if item.get('evidence') else ''}")
            lines.append(f"  Next: {item.get('next_experiment')}")
    if result.get("changes"):
        lines.append("")
        for key, value in result["changes"].items():
            lines.append(f"- {key}: {value}")
    if result.get("diagnostic_interpretation"):
        lines.append("")
        lines.append(str(result["diagnostic_interpretation"]))
    return "\n".join(lines)


def _render_markdown(result: Dict[str, Any]) -> str:
    return "# " + _render_console(result).replace("\n", "\n\n")


def load_case(path: str | Path) -> Dict[str, Any]:
    return _single_json(path)


def load_trace(path: str | Path) -> Dict[str, Any]:
    return _single_json(path)
