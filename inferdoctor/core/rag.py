from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from inferdoctor import __version__
from inferdoctor.core.endpoint_safety import classify_endpoint, redact_endpoint
from inferdoctor.core.openai_compatible import (
    OpenAICompatibleTransportError,
    create_chat_completion,
    extract_chat_text,
)

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


def _normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"[\u2010-\u2015\u2212]", "-", normalized)
    normalized = re.sub(r"[\s\u3000]+", " ", normalized)
    normalized = re.sub(r"\s*([:;,.!?()\[\]{}<>/\\|+*=])\s*", r"\1", normalized)
    return normalized.strip()


def _match_term_details(text: str, terms: Sequence[str], mode: str) -> tuple[Optional[bool], list[str], list[str]]:
    if mode == "human_review":
        return None, [], [str(term) for term in terms if isinstance(term, str)]
    normalized_text = _normalize_match_text(text)
    pairs = [(str(term), _normalize_match_text(str(term))) for term in terms if isinstance(term, str) and term]
    matched = [raw for raw, normalized in pairs if normalized and normalized in normalized_text]
    missing = [raw for raw, normalized in pairs if normalized and normalized not in normalized_text]
    if mode in {"exact_phrase", "all_terms"}:
        outcome = bool(pairs) and not missing
    elif mode == "any_term":
        outcome = bool(matched)
    else:
        outcome = False
    return outcome, matched, missing


def _terms_match(text: str, terms: Sequence[str], mode: str) -> Optional[bool]:
    return _match_term_details(text, terms, mode)[0]


def _section_dict(
    trace: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    value = trace.get(name)
    return value if isinstance(value, dict) else {}


def _list_field_known(
    trace: Dict[str, Any],
    section: str,
    key: str,
) -> bool:
    return isinstance(
        _section_dict(trace, section).get(key),
        list,
    )


def _text_field_state(
    trace: Dict[str, Any],
    section: str,
    text_key: str,
    hash_key: str,
) -> str:
    data = _section_dict(trace, section)

    if (
        text_key in data
        and isinstance(data.get(text_key), str)
    ):
        return "available"

    if (
        isinstance(data.get(hash_key), str)
        and bool(data.get(hash_key))
    ):
        return "redacted"

    return "missing"


def _candidate_source_ids(
    trace: Dict[str, Any],
) -> set[str]:
    candidates = _section_dict(
        trace,
        "retrieval",
    ).get("candidates", [])

    if not isinstance(candidates, list):
        return set()

    return {
        str(candidate.get("source_id"))
        for candidate in candidates
        if (
            isinstance(candidate, dict)
            and candidate.get("source_id")
        )
    }


def _selected_candidate_ids(
    trace: Dict[str, Any],
) -> set[str]:
    selected = _section_dict(
        trace,
        "context_selection",
    ).get("selected_chunk_ids", [])

    if not isinstance(selected, list):
        return set()

    return {
        str(item)
        for item in selected
        if item is not None
    }


def _candidate_by_chunk(
    trace: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    candidates = _section_dict(
        trace,
        "retrieval",
    ).get("candidates", [])

    if not isinstance(candidates, list):
        return {}

    return {
        str(candidate.get("chunk_id")): candidate
        for candidate in candidates
        if (
            isinstance(candidate, dict)
            and candidate.get("chunk_id")
        )
    }


def _context_text(trace: Dict[str, Any]) -> str:
    return str(
        _section_dict(
            trace,
            "context_selection",
        ).get("context_text")
        or ""
    )


def _raw_answer(trace: Dict[str, Any]) -> str:
    return str(
        _section_dict(
            trace,
            "generation",
        ).get("raw_answer")
        or ""
    )


def _final_answer(trace: Dict[str, Any]) -> str:
    return str(
        _section_dict(
            trace,
            "postprocessing",
        ).get("final_answer")
        or ""
    )


def _required_fact_coverage(
    case: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:
    results = []
    deterministic = 0
    matched = 0
    human_review = 0

    for fact in case.get("required_facts", []) or []:
        if not isinstance(fact, dict):
            continue

        mode = fact.get(
            "match_mode",
            "human_review",
        )
        terms = fact.get("match_terms", [])

        outcome, matched_terms, missing_terms = (
            _match_term_details(
                text,
                terms,
                mode,
            )
        )

        if outcome is None:
            human_review += 1
            evaluation_state = "human_review"
        else:
            deterministic += 1
            matched += 1 if outcome else 0
            evaluation_state = "evaluated"

        results.append({
            "fact_id": fact.get("fact_id"),
            "match_mode": mode,
            "matched": outcome,
            "matched_terms": matched_terms,
            "missing_terms": missing_terms,
            "evaluation_state": evaluation_state,
        })

    total = deterministic + human_review

    return {
        "matched": matched,
        "deterministic": deterministic,
        "human_review": human_review,
        "total": total,
        "deterministic_matched": matched,
        "deterministic_failed": max(
            0,
            deterministic - matched,
        ),
        "human_review_required": (
            human_review > 0
        ),
        "evaluable": True,
        "evidence_state": "available",
        "results": results,
    }


def _unevaluable_required_fact_coverage(
    case: Dict[str, Any],
    evidence_state: str,
) -> Dict[str, Any]:
    results = []
    deterministic = 0
    human_review = 0

    for fact in case.get("required_facts", []) or []:
        if not isinstance(fact, dict):
            continue

        mode = fact.get(
            "match_mode",
            "human_review",
        )

        if mode == "human_review":
            human_review += 1
            state = "human_review"
        else:
            deterministic += 1
            state = evidence_state

        results.append({
            "fact_id": fact.get("fact_id"),
            "match_mode": mode,
            "matched": None,
            "matched_terms": [],
            "missing_terms": [],
            "evaluation_state": state,
        })

    return {
        "matched": 0,
        "deterministic": deterministic,
        "human_review": human_review,
        "total": deterministic + human_review,
        "deterministic_matched": 0,
        "deterministic_failed": 0,
        "human_review_required": (
            human_review > 0
        ),
        "evaluable": False,
        "evidence_state": evidence_state,
        "results": results,
    }


def _forbidden_claims(
    case: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:
    hits = []
    results = []
    deterministic = 0
    human_review = 0

    for claim in case.get(
        "forbidden_claims",
        [],
    ) or []:
        if not isinstance(claim, dict):
            continue

        mode = claim.get(
            "match_mode",
            "human_review",
        )

        outcome, matched_terms, missing_terms = (
            _match_term_details(
                text,
                claim.get("match_terms", []),
                mode,
            )
        )

        if outcome is None:
            human_review += 1
            evaluation_state = "human_review"
        else:
            deterministic += 1
            evaluation_state = "evaluated"

            if outcome:
                hits.append(
                    claim.get("claim_id")
                )

        results.append({
            "claim_id": claim.get("claim_id"),
            "match_mode": mode,
            "matched": outcome,
            "matched_terms": matched_terms,
            "missing_terms": missing_terms,
            "evaluation_state": evaluation_state,
        })

    return {
        "hits": hits,
        "matched": len(hits),
        "deterministic": deterministic,
        "human_review": human_review,
        "total": deterministic + human_review,
        "human_review_required": (
            human_review > 0
        ),
        "evaluable": True,
        "evidence_state": "available",
        "results": results,
    }


def _unevaluable_forbidden_claims(
    case: Dict[str, Any],
    evidence_state: str,
) -> Dict[str, Any]:
    results = []
    deterministic = 0
    human_review = 0

    for claim in case.get(
        "forbidden_claims",
        [],
    ) or []:
        if not isinstance(claim, dict):
            continue

        mode = claim.get(
            "match_mode",
            "human_review",
        )

        if mode == "human_review":
            human_review += 1
            state = "human_review"
        else:
            deterministic += 1
            state = evidence_state

        results.append({
            "claim_id": claim.get("claim_id"),
            "match_mode": mode,
            "matched": None,
            "matched_terms": [],
            "missing_terms": [],
            "evaluation_state": state,
        })

    return {
        "hits": [],
        "matched": 0,
        "deterministic": deterministic,
        "human_review": human_review,
        "total": deterministic + human_review,
        "human_review_required": (
            human_review > 0
        ),
        "evaluable": False,
        "evidence_state": evidence_state,
        "results": results,
    }


def _trace_evidence_summary(
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    input_data = _section_dict(
        trace,
        "input",
    )
    retrieval = _section_dict(
        trace,
        "retrieval",
    )
    rerank = _section_dict(
        trace,
        "rerank",
    )
    context = _section_dict(
        trace,
        "context_selection",
    )
    prompt = _section_dict(
        trace,
        "prompt",
    )
    generation = _section_dict(
        trace,
        "generation",
    )
    postprocessing = _section_dict(
        trace,
        "postprocessing",
    )
    conversation = _section_dict(
        trace,
        "conversation",
    )

    checks = {
        "question_text_or_hash": bool(
            input_data.get("original_question")
            or input_data.get(
                "original_question_sha256"
            )
        ),
        "retrieval_candidates": isinstance(
            retrieval.get("candidates"),
            list,
        ),
        "retrieval_latency": (
            retrieval.get("latency_ms")
            is not None
        ),
        "rerank_status": bool(
            rerank.get("status")
        ),
        "context_text_or_hash": (
            _text_field_state(
                trace,
                "context_selection",
                "context_text",
                "context_sha256",
            )
            != "missing"
        ),
        "context_budget": (
            context.get("context_budget")
            is not None
        ),
        "prompt_hash": bool(
            prompt.get("prompt_sha256")
        ),
        "grounding_signal": (
            prompt.get(
                "grounding_instruction_present"
            )
            is not None
        ),
        "raw_answer_hash_or_text": (
            _text_field_state(
                trace,
                "generation",
                "raw_answer",
                "raw_answer_sha256",
            )
            != "missing"
        ),
        "final_answer_hash_or_text": (
            _text_field_state(
                trace,
                "postprocessing",
                "final_answer",
                "final_answer_sha256",
            )
            != "missing"
        ),
        "ttft": (
            generation.get("ttft_ms")
            is not None
        ),
        "token_usage": bool(
            generation.get("token_usage")
        ),
        "conversation_metadata": (
            conversation.get(
                "history_included"
            )
            is not None
        ),
        "stage_events": bool(
            trace.get("stage_events")
        ),
    }

    return {
        "available": [
            key
            for key, value in checks.items()
            if value
        ],
        "missing_or_redacted": [
            key
            for key, value in checks.items()
            if not value
        ],
    }



def _candidate_rank(
    candidate: Dict[str, Any],
) -> Optional[int]:
    value = candidate.get("rank")

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, float):
        integer = int(value)
        return (
            integer
            if value == integer and integer > 0
            else None
        )

    return None


def _drop_reason_text(
    value: Any,
) -> str:
    if isinstance(value, str):
        return value.casefold()

    if isinstance(value, list):
        return " ".join(
            str(item)
            for item in value
        ).casefold()

    if isinstance(value, dict):
        return " ".join(
            "{0} {1}".format(key, item)
            for key, item in value.items()
        ).casefold()

    return str(value or "").casefold()


def _ranking_failure_evidence(
    source_id: str,
    candidates: Sequence[Dict[str, Any]],
    selected_ids: set[str],
    context_selection: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    source_candidates = [
        candidate
        for candidate in candidates
        if (
            isinstance(candidate, dict)
            and str(candidate.get("source_id"))
            == source_id
        )
    ]

    if not source_candidates:
        return None

    source_chunk_ids = {
        str(candidate.get("chunk_id"))
        for candidate in source_candidates
        if candidate.get("chunk_id")
    }

    if source_chunk_ids & selected_ids:
        return None

    ranked_source_candidates = [
        (
            str(candidate.get("chunk_id")),
            _candidate_rank(candidate),
        )
        for candidate in source_candidates
        if (
            candidate.get("chunk_id")
            and _candidate_rank(candidate)
            is not None
        )
    ]

    if not ranked_source_candidates:
        return None

    best_source_rank = min(
        rank
        for _, rank in ranked_source_candidates
        if rank is not None
    )

    drop_reasons = context_selection.get(
        "drop_reasons"
    )

    if isinstance(drop_reasons, dict):
        for chunk_id, rank in ranked_source_candidates:
            reason = _drop_reason_text(
                drop_reasons.get(chunk_id)
            )

            ranking_tokens = (
                "rank",
                "top_k",
                "top-k",
                "cutoff",
                "score threshold",
                "score_threshold",
            )

            if any(
                token in reason
                for token in ranking_tokens
            ):
                return {
                    "evidence": [
                        "Required source {0} was retrieved at rank {1} but its chunk {2} was explicitly dropped for a ranking/cutoff reason: {3}".format(
                            source_id,
                            rank,
                            chunk_id,
                            reason,
                        )
                    ],
                    "evidence_strength": "observed",
                    "confidence": "high",
                    "known": [
                        "The expected source was retrieved.",
                        "The expected source was not selected into final context.",
                        "The trace explicitly attributes the drop to ranking or cutoff behavior.",
                    ],
                    "best_source_rank": (
                        best_source_rank
                    ),
                    "effective_cutoff": None,
                    "method": (
                        "explicit_drop_reason"
                    ),
                }

    selected_candidates = [
        candidate
        for candidate in candidates
        if (
            isinstance(candidate, dict)
            and candidate.get("chunk_id")
            and str(
                candidate.get("chunk_id")
            )
            in selected_ids
        )
    ]

    if not selected_candidates:
        return None

    selected_ranks = [
        _candidate_rank(candidate)
        for candidate in selected_candidates
    ]

    if any(
        rank is None
        for rank in selected_ranks
    ):
        return None

    normalized_ranks = sorted(
        {
            int(rank)
            for rank in selected_ranks
            if rank is not None
        }
    )

    if not normalized_ranks:
        return None

    expected_prefix = list(
        range(
            1,
            max(normalized_ranks) + 1,
        )
    )

    if normalized_ranks != expected_prefix:
        return None

    cutoff = max(normalized_ranks)

    if best_source_rank <= cutoff:
        return None

    return {
        "evidence": [
            "Required source {0} was retrieved at best rank {1}, while selected context chunks form an observed top-rank prefix ending at rank {2}.".format(
                source_id,
                best_source_rank,
                cutoff,
            )
        ],
        "evidence_strength": (
            "strongly_indicated"
        ),
        "confidence": "medium",
        "known": [
            "The expected source was retrieved.",
            "The selected chunks correspond to the observed top-ranked prefix.",
            "The best expected-source rank falls below that observed selection cutoff.",
        ],
        "best_source_rank": (
            best_source_rank
        ),
        "effective_cutoff": cutoff,
        "method": (
            "observed_top_rank_prefix"
        ),
    }

def diagnose_rag(
    case: Dict[str, Any],
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    case_findings = validate_case_object(case)
    trace_findings = validate_trace_object(trace)

    diagnoses: List[Dict[str, Any]] = []
    missing_evidence: List[str] = []

    def add(
        category: str,
        status: str,
        evidence: Sequence[str],
        strength: str,
        confidence: str,
        next_experiment: str,
        known: Sequence[str] = (),
        unknown: Sequence[str] = (),
    ) -> None:
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
            "unsafe_conclusions_to_avoid": [
                "Do not blame the model before retrieval, ranking, context, prompt, and post-processing evidence are isolated."
            ],
        })

    def need(message: str) -> None:
        if message not in missing_evidence:
            missing_evidence.append(message)

    schema_failures = [
        item["message"]
        for item in case_findings + trace_findings
        if item["status"] == "FAIL"
    ]

    if schema_failures:
        add(
            "insufficient_evidence",
            "warn",
            schema_failures,
            "observed",
            "high",
            "Fix the Case/Trace schema errors and rerun diagnosis.",
        )

    retrieval = _section_dict(
        trace,
        "retrieval",
    )
    context_selection = _section_dict(
        trace,
        "context_selection",
    )
    prompt = _section_dict(
        trace,
        "prompt",
    )
    conversation = _section_dict(
        trace,
        "conversation",
    )

    retrieval_known = _list_field_known(
        trace,
        "retrieval",
        "candidates",
    )
    selection_known = _list_field_known(
        trace,
        "context_selection",
        "selected_chunk_ids",
    )

    candidates = (
        retrieval.get("candidates", [])
        if retrieval_known
        else []
    )

    selected_ids = _selected_candidate_ids(trace)
    source_ids = _candidate_source_ids(trace)

    required_sources = [
        source
        for source in case.get(
            "expected_sources",
            [],
        ) or []
        if (
            isinstance(source, dict)
            and source.get("required", True)
            and source.get("source_id")
        )
    ]

    required_source_ids = [
        str(source["source_id"])
        for source in required_sources
    ]

    if not retrieval_known:
        need(
            "Retrieval candidates were not captured, so the retrieval and ranking layers cannot be evaluated."
        )

    elif required_source_ids:
        if not candidates:
            add(
                "retrieval_failure",
                "fail",
                [
                    "Required source not retrieved: {0}".format(
                        source_id
                    )
                    for source_id in required_source_ids
                ],
                "observed",
                "high",
                "Probe retrieval with the expected source terms and inspect indexing or source availability.",
                known=[
                    "The captured retrieval candidate list is explicitly empty."
                ],
            )

        else:
            candidates_without_source_id = [
                candidate
                for candidate in candidates
                if (
                    isinstance(candidate, dict)
                    and not candidate.get("source_id")
                )
            ]

            missing_sources = [
                source_id
                for source_id in required_source_ids
                if source_id not in source_ids
            ]

            if missing_sources:
                if candidates_without_source_id:
                    need(
                        "Some retrieval candidates do not contain source_id, so absence of the expected source cannot be proven."
                    )
                else:
                    add(
                        "retrieval_failure",
                        "fail",
                        [
                            "Required source not retrieved: {0}".format(
                                source_id
                            )
                            for source_id in missing_sources
                        ],
                        "observed",
                        "high",
                        "Probe retrieval with the expected source terms and inspect indexing or source availability.",
                        known=[
                            "The expected source was absent from the captured retrieval candidates."
                        ],
                    )

    retrieved_required_source_ids = [
        source_id
        for source_id in required_source_ids
        if source_id in source_ids
    ]

    if retrieved_required_source_ids:
        if not selection_known:
            need(
                "Selected context chunk IDs were not captured, so ranking-vs-context-selection attribution cannot be proven."
            )

        else:
            for source_id in (
                retrieved_required_source_ids
            ):
                source_candidates = [
                    candidate
                    for candidate in candidates
                    if (
                        isinstance(
                            candidate,
                            dict,
                        )
                        and str(
                            candidate.get(
                                "source_id"
                            )
                        )
                        == source_id
                    )
                ]

                source_chunk_ids = {
                    str(
                        candidate.get(
                            "chunk_id"
                        )
                    )
                    for candidate
                    in source_candidates
                    if candidate.get(
                        "chunk_id"
                    )
                }

                if not source_chunk_ids:
                    need(
                        "A required source was retrieved but its candidate chunk_id was not captured, so ranking and context selection cannot be evaluated."
                    )
                    continue

                if (
                    source_chunk_ids
                    & selected_ids
                ):
                    continue

                ranking_evidence = (
                    _ranking_failure_evidence(
                        source_id,
                        candidates,
                        selected_ids,
                        context_selection,
                    )
                )

                if ranking_evidence:
                    add(
                        "ranking_failure",
                        "fail",
                        ranking_evidence[
                            "evidence"
                        ],
                        ranking_evidence[
                            "evidence_strength"
                        ],
                        ranking_evidence[
                            "confidence"
                        ],
                        "Keep retrieval fixed and test ranking or selection cutoff changes before changing the model or prompt.",
                        known=(
                            ranking_evidence[
                                "known"
                            ]
                        ),
                    )

                    diagnoses[-1][
                        "ranking_evidence"
                    ] = {
                        "method": (
                            ranking_evidence[
                                "method"
                            ]
                        ),
                        "best_source_rank": (
                            ranking_evidence[
                                "best_source_rank"
                            ]
                        ),
                        "effective_cutoff": (
                            ranking_evidence[
                                "effective_cutoff"
                            ]
                        ),
                    }

                    continue

                add(
                    "context_selection_failure",
                    "fail",
                    [
                        "Required source retrieved but not selected into final context: {0}".format(
                            source_id
                        )
                    ],
                    "observed",
                    "high",
                    "Keep retrieval fixed and inspect context-selection rules, rerank output, filters, context budget, and selected_chunk_ids.",
                    known=[
                        "The required source exists in retrieval candidates.",
                        "None of its captured chunks appear in selected_chunk_ids.",
                        "Available evidence does not prove that rank or top-k cutoff caused the exclusion.",
                    ],
                )

    elif required_source_ids and retrieval_known:
        if not any(
            diagnosis["category"]
            == "retrieval_failure"
            for diagnosis in diagnoses
        ):
            need(
                "No required source could be positively matched to a captured retrieval candidate."
            )

    if not selection_known:
        need(
            "selected_chunk_ids were not captured, so the context-selection layer is not fully observable."
        )

    context_state = _text_field_state(
        trace,
        "context_selection",
        "context_text",
        "context_sha256",
    )
    raw_state = _text_field_state(
        trace,
        "generation",
        "raw_answer",
        "raw_answer_sha256",
    )
    final_state = _text_field_state(
        trace,
        "postprocessing",
        "final_answer",
        "final_answer_sha256",
    )

    context = _context_text(trace)
    raw = _raw_answer(trace)
    final = _final_answer(trace)

    if context_state == "available":
        context_coverage = _required_fact_coverage(
            case,
            context,
        )
    else:
        context_coverage = (
            _unevaluable_required_fact_coverage(
                case,
                context_state,
            )
        )

    if raw_state == "available":
        raw_coverage = _required_fact_coverage(
            case,
            raw,
        )
    else:
        raw_coverage = (
            _unevaluable_required_fact_coverage(
                case,
                raw_state,
            )
        )

    if final_state == "available":
        final_coverage = _required_fact_coverage(
            case,
            final,
        )
        forbidden = _forbidden_claims(
            case,
            final,
        )
    else:
        final_coverage = (
            _unevaluable_required_fact_coverage(
                case,
                final_state,
            )
        )
        forbidden = (
            _unevaluable_forbidden_claims(
                case,
                final_state,
            )
        )

    if case.get("required_facts"):
        if context_state != "available":
            need(
                "Final context text is {0}; required-fact coverage in context cannot be evaluated.".format(
                    context_state
                )
            )

        if raw_state != "available":
            need(
                "Raw model answer is {0}; generation-layer fact coverage cannot be evaluated.".format(
                    raw_state
                )
            )

        if final_state != "available":
            need(
                "Final answer text is {0}; final-answer fact coverage cannot be evaluated.".format(
                    final_state
                )
            )

    if (
        context_selection.get("truncated")
        and context_coverage["evaluable"]
        and context_coverage["deterministic"]
        and (
            context_coverage["matched"]
            < context_coverage["deterministic"]
        )
    ):
        add(
            "context_truncation",
            "fail",
            [
                "Context was marked truncated and required facts were absent from final context."
            ],
            "observed",
            "medium",
            "Run the same case with a larger context budget or inspect truncation_detail.",
            unknown=[
                "Whether the missing fact existed before truncation unless dropped chunks retain safe evidence."
            ],
        )

    if (
        context_coverage["evaluable"]
        and final_coverage["evaluable"]
        and context_coverage["deterministic"]
        and final_coverage["deterministic"]
        and (
            context_coverage["matched"]
            == context_coverage["deterministic"]
        )
        and (
            final_coverage["matched"]
            < final_coverage["deterministic"]
        )
    ):
        grounding_signal = prompt.get(
            "grounding_instruction_present"
        )

        if grounding_signal is False:
            add(
                "prompt_grounding_failure",
                "warn",
                [
                    "Correct evidence appears in context but the grounding instruction is absent."
                ],
                "strongly_indicated",
                "medium",
                "Run a Gold Context Probe with an explicit grounding instruction.",
                known=[
                    "Required facts are present in the final context."
                ],
            )

        elif grounding_signal is True:
            add(
                "insufficient_evidence",
                "warn",
                [
                    "Correct evidence appears in context but the final answer omits required facts."
                ],
                "possible",
                "medium",
                "Run Gold Context Probe before attributing the problem to model capability.",
                known=[
                    "Required facts are present in context.",
                    "Grounding instruction was reported as present.",
                ],
                unknown=[
                    "Whether the model can use the same evidence when retrieval and context construction are removed from the path."
                ],
            )

        else:
            need(
                "Prompt grounding metadata was not captured, so prompt-vs-model responsibility cannot be isolated."
            )

    if (
        raw_coverage["evaluable"]
        and final_coverage["evaluable"]
        and (
            raw_coverage["matched"]
            > final_coverage["matched"]
        )
    ):
        add(
            "answer_postprocessing_failure",
            "fail",
            [
                "Raw answer covered more required facts than the final answer."
            ],
            "observed",
            "high",
            "Inspect postprocessing transformations and final answer assembly.",
            known=[
                "Required-fact coverage decreased after generation."
            ],
        )

    if (
        conversation.get("history_included")
        and conversation.get(
            "possible_contamination_signals"
        )
    ):
        add(
            "conversation_memory_contamination",
            "warn",
            [
                "Conversation history was included and contamination signals were reported."
            ],
            "possible",
            "medium",
            "Replay the same question as a clean single-turn trace.",
        )

    if case.get("forbidden_claims"):
        if not forbidden["evaluable"]:
            need(
                "Final answer text is {0}; forbidden-claim checks cannot be evaluated.".format(
                    final_state
                )
            )

        if forbidden["human_review_required"]:
            need(
                "At least one forbidden-claim rule requires human review."
            )

    if final_coverage["human_review_required"]:
        need(
            "At least one required-fact rule requires human review."
        )

    if missing_evidence:
        add(
            "insufficient_evidence",
            "warn",
            missing_evidence,
            "observed",
            "high",
            "Capture the missing trace fields, or use hashes and explicit safe metadata when content cannot be retained.",
            unknown=missing_evidence,
        )

    if not diagnoses:
        add(
            "no_clear_failure",
            "pass",
            [
                "No deterministic failure was identified from the available evidence."
            ],
            "observed",
            "medium",
            "Add more trace fields or run Gold Context Probe if the answer is still unacceptable.",
        )

    evidence_summary = _trace_evidence_summary(
        trace
    )

    missing_count = len(
        evidence_summary[
            "missing_or_redacted"
        ]
    )

    evidence_score = max(
        0,
        100
        - 12 * len([
            diagnosis
            for diagnosis in diagnoses
            if (
                diagnosis["category"]
                == "insufficient_evidence"
            )
        ])
        - min(
            40,
            missing_count * 3,
        ),
    )

    evidence_states = {
        "retrieval_candidates": (
            "available"
            if retrieval_known
            else "missing"
        ),
        "selected_chunk_ids": (
            "available"
            if selection_known
            else "missing"
        ),
        "context_text": context_state,
        "raw_answer": raw_state,
        "final_answer": final_state,
    }

    return _apply_rag_attribution({
        "schema_version": (
            RAG_DIAGNOSIS_SCHEMA_VERSION
        ),
        "timestamp": utc_now(),
        "inferdoctor_version": __version__,
        "case_id": case.get("case_id"),
        "trace_id": trace.get("trace_id"),
        "status": (
            "FAIL"
            if any(
                diagnosis["status"] == "fail"
                for diagnosis in diagnoses
            )
            else "WARN"
            if any(
                diagnosis["status"] == "warn"
                for diagnosis in diagnoses
            )
            else "PASS"
        ),
        "evidence_completeness_score": (
            evidence_score
        ),
        "evidence_completeness": (
            evidence_summary
        ),
        "evidence_states": evidence_states,
        "diagnoses": diagnoses,
        "required_fact_coverage": {
            "context": context_coverage,
            "raw_answer": raw_coverage,
            "final_answer": final_coverage,
        },
        "forbidden_claims": forbidden,
    })


RAG_LAYER_ORDER = {
    "conversation_memory_contamination": 10,
    "retrieval_failure": 20,
    "ranking_failure": 30,
    "context_selection_failure": 40,
    "context_truncation": 50,
    "prompt_grounding_failure": 60,
    "model_reasoning_limitation": 70,
    "answer_postprocessing_failure": 80,
}

RAG_LAYER_NAMES = {
    "conversation_memory_contamination": "conversation",
    "retrieval_failure": "retrieval",
    "ranking_failure": "ranking",
    "context_selection_failure": "context_selection",
    "context_truncation": "context",
    "prompt_grounding_failure": "prompt",
    "model_reasoning_limitation": "generation",
    "answer_postprocessing_failure": "postprocessing",
}



RAG_ATTRIBUTION_LAYER_SEQUENCE = (
    (
        "conversation",
        "conversation_memory_contamination",
    ),
    (
        "retrieval",
        "retrieval_failure",
    ),
    (
        "ranking",
        "ranking_failure",
    ),
    (
        "context_selection",
        "context_selection_failure",
    ),
    (
        "context",
        "context_truncation",
    ),
    (
        "prompt",
        "prompt_grounding_failure",
    ),
    (
        "generation",
        "model_reasoning_limitation",
    ),
    (
        "postprocessing",
        "answer_postprocessing_failure",
    ),
)


def _build_layer_chain(
    diagnoses: Sequence[Dict[str, Any]],
    first_category: Optional[str],
) -> List[Dict[str, Any]]:
    first_order = (
        RAG_LAYER_ORDER.get(first_category)
        if first_category
        else None
    )

    by_category: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for diagnosis in diagnoses:
        category = str(
            diagnosis.get("category") or ""
        )

        if category not in RAG_LAYER_ORDER:
            continue

        if diagnosis.get("status") not in {
            "fail",
            "warn",
        }:
            continue

        if category not in by_category:
            by_category[category] = diagnosis

    established_upstream: Dict[
        str,
        str,
    ] = {}

    if first_category == "ranking_failure":
        established_upstream["retrieval"] = (
            "Required source was present in "
            "retrieval candidates."
        )

    elif (
        first_category
        == "context_selection_failure"
    ):
        established_upstream["retrieval"] = (
            "Required source was present in "
            "retrieval candidates."
        )

    elif (
        first_category
        == "prompt_grounding_failure"
    ):
        established_upstream["context"] = (
            "Required facts were observed in "
            "the final context."
        )

    elif (
        first_category
        == "model_reasoning_limitation"
    ):
        established_upstream["context"] = (
            "Required facts were observed in "
            "the supplied context."
        )
        established_upstream["prompt"] = (
            "Prompt grounding was established "
            "before generation attribution."
        )

    elif (
        first_category
        == "answer_postprocessing_failure"
    ):
        established_upstream["generation"] = (
            "The raw model answer contained "
            "more required facts than the final "
            "postprocessed answer."
        )

    chain: List[Dict[str, Any]] = []

    for layer, category in (
        RAG_ATTRIBUTION_LAYER_SEQUENCE
    ):
        diagnosis = by_category.get(
            category
        )

        if diagnosis is not None:
            order = RAG_LAYER_ORDER[
                category
            ]

            if category == first_category:
                role = "FIRST_BROKEN"

            elif (
                first_order is not None
                and order > first_order
            ):
                role = (
                    "DOWNSTREAM_OBSERVATION"
                )

            else:
                role = (
                    "INDEPENDENT_OBSERVATION"
                )

            evidence = (
                diagnosis.get("evidence")
                or []
            )

            chain.append({
                "layer": layer,
                "status": str(
                    diagnosis.get(
                        "status",
                        "warn",
                    )
                ).upper(),
                "role": role,
                "category": category,
                "confidence": (
                    diagnosis.get(
                        "confidence",
                        "unknown",
                    )
                ),
                "evidence": (
                    evidence[0]
                    if evidence
                    else ""
                ),
            })

            continue

        if layer in established_upstream:
            chain.append({
                "layer": layer,
                "status": "PASS",
                "role": (
                    "ESTABLISHED_UPSTREAM"
                ),
                "category": None,
                "confidence": "high",
                "evidence": (
                    established_upstream[
                        layer
                    ]
                ),
            })

            continue

        chain.append({
            "layer": layer,
            "status": "UNKNOWN",
            "role": "NOT_ATTRIBUTED",
            "category": None,
            "confidence": "unknown",
            "evidence": (
                "No supported conclusion for "
                "this layer from the available "
                "evidence."
            ),
        })

    return chain


def _build_rag_attribution(
    diagnoses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = []

    for index, diagnosis in enumerate(
        diagnoses
    ):
        category = str(
            diagnosis.get("category") or ""
        )
        order = RAG_LAYER_ORDER.get(
            category
        )

        if order is None:
            diagnosis[
                "attribution_role"
            ] = "evidence_or_summary"
            continue

        if diagnosis.get("status") not in {
            "fail",
            "warn",
        }:
            diagnosis[
                "attribution_role"
            ] = "not_broken"
            continue

        candidates.append(
            (
                order,
                0
                if diagnosis.get("status")
                == "fail"
                else 1,
                index,
                diagnosis,
            )
        )

    if not candidates:
        return {
            "first_broken_layer": None,
            "first_broken_category": None,
            "first_broken_status": None,
            "confidence": "unknown",
            "downstream_observations": [],
            "layer_chain": (
                _build_layer_chain(
                    diagnoses,
                    None,
                )
            ),
            "causal_claim": (
                "No broken layer was "
                "established from the "
                "available evidence."
            ),
        }

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    first_order, _, _, first = (
        candidates[0]
    )

    first_category = str(
        first["category"]
    )

    downstream = []

    for (
        order,
        _,
        _,
        diagnosis,
    ) in candidates:
        category = str(
            diagnosis.get("category") or ""
        )

        if diagnosis is first:
            diagnosis[
                "attribution_role"
            ] = "first_broken_layer"
            continue

        if order > first_order:
            diagnosis[
                "attribution_role"
            ] = "downstream_observation"

            downstream.append(
                category
            )

        else:
            diagnosis[
                "attribution_role"
            ] = (
                "same_or_independent_layer"
            )

    first["downstream_effects"] = list(
        dict.fromkeys(downstream)
    )

    return {
        "first_broken_layer": (
            RAG_LAYER_NAMES.get(
                first_category,
                first_category,
            )
        ),
        "first_broken_category": (
            first_category
        ),
        "first_broken_status": (
            first.get("status")
        ),
        "confidence": first.get(
            "confidence",
            "unknown",
        ),
        "downstream_observations": (
            list(
                dict.fromkeys(
                    downstream
                )
            )
        ),
        "layer_chain": (
            _build_layer_chain(
                diagnoses,
                first_category,
            )
        ),
        "causal_claim": (
            "This is the earliest supported "
            "broken layer in pipeline order. "
            "Later findings are downstream "
            "observations; causation is not "
            "assumed unless separately proven."
        ),
    }

def _apply_rag_attribution(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    diagnoses = result.get("diagnoses")

    if not isinstance(diagnoses, list):
        result["attribution"] = (
            _build_rag_attribution([])
        )
        return result

    result["attribution"] = (
        _build_rag_attribution(diagnoses)
    )

    return result


def compare_rag(
    case: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    compatibility: List[str] = []
    limitations: List[str] = []

    if (
        before.get("case_id")
        and after.get("case_id")
        and before.get("case_id")
        != after.get("case_id")
    ):
        compatibility.append(
            "case IDs differ"
        )

    before_q = _section_dict(
        before,
        "input",
    ).get("original_question")

    after_q = _section_dict(
        after,
        "input",
    ).get("original_question")

    if (
        before_q
        and after_q
        and before_q != after_q
    ):
        compatibility.append(
            "questions differ"
        )

    if (
        before.get("pipeline")
        != after.get("pipeline")
    ):
        compatibility.append(
            "pipeline differs"
        )

    before_model = _section_dict(
        before,
        "generation",
    ).get("model")

    after_model = _section_dict(
        after,
        "generation",
    ).get("model")

    if before_model != after_model:
        compatibility.append(
            "model differs"
        )

    before_diag = diagnose_rag(
        case,
        before,
    )
    after_diag = diagnose_rag(
        case,
        after,
    )

    before_context = (
        before_diag[
            "required_fact_coverage"
        ]["context"]
    )
    after_context = (
        after_diag[
            "required_fact_coverage"
        ]["context"]
    )

    before_final = (
        before_diag[
            "required_fact_coverage"
        ]["final_answer"]
    )
    after_final = (
        after_diag[
            "required_fact_coverage"
        ]["final_answer"]
    )

    quality_evaluable = (
        before_final.get("evaluable") is True
        and after_final.get("evaluable") is True
        and before_context.get("evaluable") is True
        and after_context.get("evaluable") is True
        and before_diag[
            "forbidden_claims"
        ].get("evaluable") is True
        and after_diag[
            "forbidden_claims"
        ].get("evaluable") is True
    )

    if not quality_evaluable:
        limitations.append(
            "Answer-quality comparison is "
            "incomplete because context or answer "
            "evidence is missing or redacted."
        )

    human_review_required = bool(
        before_final.get(
            "human_review_required"
        )
        or after_final.get(
            "human_review_required"
        )
        or before_diag[
            "forbidden_claims"
        ].get(
            "human_review_required"
        )
        or after_diag[
            "forbidden_claims"
        ].get(
            "human_review_required"
        )
    )

    if human_review_required:
        limitations.append(
            "At least one quality rule requires "
            "human review."
        )

    retrieval_delta = _delta_num(
        after,
        before,
        "retrieval",
        "latency_ms",
    )

    generation_delta = _delta_num(
        after,
        before,
        "generation",
        "total_ms",
    )

    if retrieval_delta is None:
        limitations.append(
            "Retrieval latency delta is unknown."
        )

    if generation_delta is None:
        limitations.append(
            "Generation latency delta is unknown."
        )

    changes = {
        "context_required_fact_delta": (
            after_context["matched"]
            - before_context["matched"]
            if quality_evaluable
            else None
        ),
        "final_required_fact_delta": (
            after_final["matched"]
            - before_final["matched"]
            if quality_evaluable
            else None
        ),
        "forbidden_claim_delta": (
            len(
                after_diag[
                    "forbidden_claims"
                ]["hits"]
            )
            - len(
                before_diag[
                    "forbidden_claims"
                ]["hits"]
            )
            if quality_evaluable
            else None
        ),
        "retrieval_latency_ms_delta": (
            retrieval_delta
        ),
        "generation_total_ms_delta": (
            generation_delta
        ),
    }

    final_delta = changes[
        "final_required_fact_delta"
    ]
    forbidden_delta = changes[
        "forbidden_claim_delta"
    ]

    if compatibility:
        verdict = "incompatible"

    elif (
        not quality_evaluable
        or human_review_required
    ):
        verdict = "inconclusive"

    elif (
        isinstance(final_delta, (int, float))
        and final_delta > 0
        and (
            forbidden_delta is None
            or forbidden_delta <= 0
        )
    ):
        verdict = "improved"

    elif (
        (
            isinstance(
                final_delta,
                (int, float),
            )
            and final_delta < 0
        )
        or (
            isinstance(
                forbidden_delta,
                (int, float),
            )
            and forbidden_delta > 0
        )
    ):
        verdict = "regressed"

    else:
        quality_changes = [
            changes[
                "context_required_fact_delta"
            ],
            changes[
                "final_required_fact_delta"
            ],
            changes[
                "forbidden_claim_delta"
            ],
        ]

        performance_changes = [
            retrieval_delta,
            generation_delta,
        ]

        if (
            all(
                value == 0
                for value in quality_changes
            )
            and all(
                value == 0
                for value in performance_changes
                if value is not None
            )
            and all(
                value is not None
                for value in performance_changes
            )
        ):
            verdict = "unchanged"
        else:
            verdict = "inconclusive"

    before_first = (
        before_diag.get(
            "attribution",
            {},
        ).get("first_broken_layer")
    )

    after_first = (
        after_diag.get(
            "attribution",
            {},
        ).get("first_broken_layer")
    )

    return {
        "schema_version": (
            RAG_COMPARISON_SCHEMA_VERSION
        ),
        "timestamp": utc_now(),
        "inferdoctor_version": __version__,
        "case_id": case.get("case_id"),
        "verdict": verdict,
        "compatibility_warnings": (
            compatibility
        ),
        "comparison_limitations": (
            list(dict.fromkeys(limitations))
        ),
        "changes": changes,
        "before_status": (
            before_diag["status"]
        ),
        "after_status": (
            after_diag["status"]
        ),
        "before_first_broken_layer": (
            before_first
        ),
        "after_first_broken_layer": (
            after_first
        ),
        "first_broken_layer_changed": (
            before_first != after_first
        ),
        "confidence": (
            "low"
            if (
                compatibility
                or not quality_evaluable
                or human_review_required
            )
            else "medium"
        ),
    }


def _num(
    data: Dict[str, Any],
    section: str,
    key: str,
) -> Optional[float]:
    value = _section_dict(
        data,
        section,
    ).get(key)

    if not isinstance(
        value,
        (int, float),
    ):
        return None

    return float(value)


def _delta_num(
    after: Dict[str, Any],
    before: Dict[str, Any],
    section: str,
    key: str,
) -> Optional[float]:
    after_value = _num(
        after,
        section,
        key,
    )
    before_value = _num(
        before,
        section,
        key,
    )

    if (
        after_value is None
        or before_value is None
    ):
        return None

    return (
        after_value
        - before_value
    )

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


def evaluate_deterministic_answer(answer: str, case: Dict[str, Any]) -> Dict[str, Any]:
    required = _required_fact_coverage(case, answer)
    forbidden = _forbidden_claims(case, answer)
    required_total = int(required.get("total") or 0)
    required_deterministic = int(required.get("deterministic") or 0)
    required_matched = int(required.get("matched") or 0)
    forbidden_total = int(forbidden.get("total") or 0)
    forbidden_hits = int(forbidden.get("matched") or 0)
    human_review_required = bool(required.get("human_review_required") or forbidden.get("human_review_required"))
    deterministic_available = required_deterministic > 0 or int(forbidden.get("deterministic") or 0) > 0

    if forbidden_hits:
        evaluation_status = "fail"
        overall_status = "fail"
        interpretation = "The model response contained a forbidden deterministic claim."
    elif required_deterministic and required_matched < required_deterministic:
        evaluation_status = "fail"
        overall_status = "fail"
        interpretation = "The model response did not satisfy all deterministic required facts."
    elif not deterministic_available:
        evaluation_status = "inconclusive"
        overall_status = "inconclusive"
        interpretation = "No deterministic checks were available; human review is required."
    elif human_review_required:
        evaluation_status = "pass"
        overall_status = "inconclusive"
        interpretation = "Deterministic checks passed, but at least one claim still requires human review."
    else:
        evaluation_status = "pass"
        overall_status = "pass"
        interpretation = "Transport succeeded and deterministic fact checks passed."

    return {
        "required_fact_checks": required,
        "forbidden_claim_checks": forbidden,
        "required_facts_total": required_total,
        "required_facts_matched": required_matched,
        "forbidden_claims_total": forbidden_total,
        "forbidden_claims_matched": forbidden_hits,
        "deterministic_checks_available": deterministic_available,
        "human_review_required": human_review_required,
        "evaluation_status": evaluation_status,
        "review_status": "required" if human_review_required else "not_required",
        "overall_status": overall_status,
        "status": overall_status.upper(),
        "diagnostic_interpretation": interpretation,
    }


def _gold_probe_evaluation(answer: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for the shared deterministic evaluator."""
    return evaluate_deterministic_answer(answer, case)


def unavailable_deterministic_answer(
    case: Dict[str, Any],
    *,
    evidence_state: str,
) -> Dict[str, Any]:
    required = _unevaluable_required_fact_coverage(case, evidence_state)
    forbidden = _unevaluable_forbidden_claims(case, evidence_state)
    return {
        "required_fact_checks": required,
        "forbidden_claim_checks": forbidden,
        "required_facts_total": int(required.get("total") or 0),
        "required_facts_matched": 0,
        "forbidden_claims_total": int(forbidden.get("total") or 0),
        "forbidden_claims_matched": 0,
        "deterministic_checks_available": bool(
            required.get("deterministic") or forbidden.get("deterministic")
        ),
        "human_review_required": bool(
            required.get("human_review_required")
            or forbidden.get("human_review_required")
        ),
        "evaluation_status": "unknown",
        "review_status": "unavailable",
        "overall_status": "unknown",
        "status": "UNKNOWN",
        "diagnostic_interpretation": (
            "No usable response evidence was available for deterministic answer verification."
        ),
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
    context_required = _required_fact_coverage(case, context_text)
    context_forbidden = _forbidden_claims(case, context_text)
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
        "request_status": "skipped" if dry_run else "pending",
        "transport_status": "skipped" if dry_run else "unknown",
        "evaluation_status": "skipped" if dry_run else "unknown",
        "review_status": "unavailable" if dry_run else "unknown",
        "overall_status": "dry_run" if dry_run else "unknown",
        "status": "DRY_RUN" if dry_run else "UNKNOWN",
        "ttft_ms": None,
        "total_latency_ms": None,
        "generated_answer_sha256": None,
        "answer_preview": None,
        "answer_retained": bool(retain_answer),
        "answer_evaluated_in_memory": False,
        "required_fact_checks": context_required,
        "forbidden_claim_checks": context_forbidden,
        "required_facts_total": int(context_required.get("total") or 0),
        "required_facts_matched": int(context_required.get("matched") or 0),
        "forbidden_claims_total": int(context_forbidden.get("total") or 0),
        "forbidden_claims_matched": int(context_forbidden.get("matched") or 0),
        "deterministic_checks_available": bool(context_required.get("deterministic") or context_forbidden.get("deterministic")),
        "human_review_required": bool(context_required.get("human_review_required") or context_forbidden.get("human_review_required")),
        "diagnostic_interpretation": "Dry run only; no model request was sent." if dry_run else "",
    }
    if dry_run:
        return result
    payload = {
        "model": model,
        "messages": payload_info["messages"],
        "stream": False,
        "max_tokens": 256,
    }
    started = time.perf_counter()
    try:
        response = create_chat_completion(
            endpoint,
            payload=payload,
            timeout=timeout,
            api_key=api_key,
        )
        if not 200 <= response.status < 300:
            raise RagError(
                "gold-context probe returned HTTP {0}".format(response.status)
            )
        if not response.json_valid:
            raise RagError("gold-context probe returned invalid JSON")
        parsed = response.json_data
        elapsed = response.elapsed_ms
        answer = _extract_chat_answer(parsed)
        evaluation = evaluate_deterministic_answer(answer, case)
        result.update(evaluation)
        result.update({
            "request_sent": True,
            "request_status": "sent",
            "transport_status": "pass",
            "total_latency_ms": elapsed,
            "generated_answer_sha256": sha256_text(answer) if answer else None,
            "answer_preview": answer[:240] if retain_answer else None,
            "answer_retained": bool(retain_answer),
            "answer_evaluated_in_memory": True,
        })
        return result
    except (OpenAICompatibleTransportError, json.JSONDecodeError, RagError) as exc:
        result.update({
            "request_sent": True,
            "request_status": "sent",
            "transport_status": "fail",
            "evaluation_status": "skipped",
            "review_status": "unavailable",
            "overall_status": "request_failed",
            "status": "FAIL",
            "total_latency_ms": int((time.perf_counter() - started) * 1000),
            "error": str(exc)[:300],
            "diagnostic_interpretation": "Gold context probe failed before a usable answer was produced.",
        })
        return result


def _extract_chat_answer(parsed: Any) -> str:
    return extract_chat_text(parsed)


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
    for key in ("status", "overall_status", "transport_status", "evaluation_status", "review_status", "required_facts_matched", "required_facts_total", "forbidden_claims_matched", "forbidden_claims_total", "verdict", "case_count", "case_id", "trace_id"):
        if result.get(key) is not None:
            lines.append(f"{key}: {result.get(key)}")
    attribution = result.get(
        "attribution"
    )

    if isinstance(
        attribution,
        dict,
    ):
        lines.append("")
        lines.append("Attribution:")

        first_layer = attribution.get(
            "first_broken_layer"
        )

        lines.append(
            "First broken layer: {0}".format(
                first_layer
                or "not established"
            )
        )

        chain = attribution.get(
            "layer_chain"
        )

        if isinstance(chain, list):
            for item in chain:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                role = str(
                    item.get("role")
                    or ""
                )

                marker = (
                    "  <-- FIRST BROKEN"
                    if role
                    == "FIRST_BROKEN"
                    else ""
                )

                role_note = ""

                if (
                    role
                    == "DOWNSTREAM_OBSERVATION"
                ):
                    role_note = (
                        " (downstream observation)"
                    )

                elif (
                    role
                    == "ESTABLISHED_UPSTREAM"
                ):
                    role_note = (
                        " (established upstream)"
                    )

                lines.append(
                    "- {0}: {1}{2}{3}".format(
                        item.get("layer"),
                        item.get("status"),
                        role_note,
                        marker,
                    )
                )

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
