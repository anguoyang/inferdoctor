from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from inferdoctor import __version__
from inferdoctor.core.rag import (
    RagError,
    RAG_COMPARISON_POLICY_QUALITY,
    compare_rag,
    diagnose_rag,
    load_cases,
    load_trace,
    utc_now,
    validate_case_object,
    validate_trace_object,
)


RAG_GATE_SCHEMA_VERSION = "inferdoctor.rag.gate.v1"
_TRACE_SUFFIXES = {".json", ".jsonl"}


def _issue(
    kind: str,
    message: str,
    *,
    case_id: Optional[str] = None,
    side: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "kind": kind,
        "message": message,
    }
    if case_id is not None:
        result["case_id"] = case_id
    if side is not None:
        result["side"] = side
    return result


def _case_id(item: Mapping[str, Any]) -> Optional[str]:
    value = item.get("case_id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _validation_reason(label: str, findings: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted(
        {
            str(item.get("field") or "unknown")
            for item in findings
            if item.get("status") == "FAIL"
        }
    )
    return "{0} failed validation for fields: {1}.".format(
        label,
        ", ".join(fields) if fields else "unknown",
    )


def _load_case_entries(
    cases_path: str | Path,
) -> Tuple[List[Dict[str, Any]], List[List[str]], List[Dict[str, Any]]]:
    try:
        cases = load_cases(cases_path)
    except OSError:
        return [], [], [
            _issue(
                "invalid_cases_source",
                "The Cases source could not be read.",
            )
        ]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [], [], [
            _issue(
                "invalid_cases_source",
                "The Cases source is not valid UTF-8 JSON or JSONL.",
            )
        ]
    except RagError as exc:
        return [], [], [
            _issue(
                "invalid_cases_source",
                "The Cases source is invalid: {0}".format(str(exc)),
            )
        ]

    if not cases:
        return [], [], [
            _issue(
                "invalid_cases_source",
                "The Cases source contains no Case objects.",
            )
        ]

    identifiers = [_case_id(case) for case in cases]
    counts = Counter(identifier for identifier in identifiers if identifier is not None)
    duplicate_ids = {identifier for identifier, count in counts.items() if count > 1}
    issues = [
        _issue(
            "duplicate_case_id",
            "Duplicate Case case_id prevents unambiguous trace matching.",
            case_id=identifier,
        )
        for identifier in sorted(duplicate_ids)
    ]
    reasons: List[List[str]] = []
    for case, identifier in zip(cases, identifiers):
        item_reasons: List[str] = []
        failures = [
            item
            for item in validate_case_object(case)
            if item.get("status") == "FAIL"
        ]
        if failures:
            reason = _validation_reason("Case", failures)
            item_reasons.append(reason)
            issues.append(
                _issue(
                    "invalid_case",
                    reason,
                    case_id=identifier,
                )
            )
        if identifier in duplicate_ids:
            item_reasons.append(
                "Duplicate Case case_id prevents unambiguous trace matching."
            )
        reasons.append(item_reasons)
    return cases, reasons, issues


def _trace_paths(directory: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _TRACE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def _load_trace_directory(
    directory_value: str | Path,
    *,
    side: str,
    expected_case_ids: set[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], List[Dict[str, Any]]]:
    directory = Path(directory_value)
    issues: List[Dict[str, Any]] = []
    blocked: DefaultDict[str, List[str]] = defaultdict(list)
    valid: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    observed: DefaultDict[str, List[str]] = defaultdict(list)

    if not directory.is_dir():
        issues.append(
            _issue(
                "invalid_trace_directory",
                "The {0} trace directory is missing or is not a directory.".format(
                    side
                ),
                side=side,
            )
        )
        return {}, {}, issues

    try:
        paths = _trace_paths(directory)
    except OSError:
        issues.append(
            _issue(
                "invalid_trace_directory",
                "The {0} trace directory could not be read.".format(side),
                side=side,
            )
        )
        return {}, {}, issues

    for path in paths:
        try:
            trace = load_trace(path)
        except OSError:
            issues.append(
                _issue(
                    "invalid_trace",
                    "A {0} trace file could not be read.".format(side),
                    side=side,
                )
            )
            continue
        except (UnicodeDecodeError, json.JSONDecodeError, RagError):
            issues.append(
                _issue(
                    "invalid_trace",
                    "A {0} trace file is not one valid RAG Trace JSON object.".format(
                        side
                    ),
                    side=side,
                )
            )
            continue

        identifier = _case_id(trace)
        if identifier is None:
            issues.append(
                _issue(
                    "missing_trace_case_id",
                    "A {0} trace has no case_id required for gate matching.".format(
                        side
                    ),
                    side=side,
                )
            )
            continue

        observed[identifier].append(path.name)
        failures = [
            item
            for item in validate_trace_object(trace)
            if item.get("status") == "FAIL"
        ]
        if failures:
            reason = _validation_reason(
                "{0} trace".format(side.capitalize()),
                failures,
            )
            blocked[identifier].append(reason)
            issues.append(
                _issue(
                    "invalid_trace",
                    reason,
                    case_id=identifier,
                    side=side,
                )
            )
            continue

        if identifier not in expected_case_ids:
            issues.append(
                _issue(
                    "mismatched_trace_case_id",
                    "A {0} trace case_id does not match any supplied Case.".format(
                        side
                    ),
                    case_id=identifier,
                    side=side,
                )
            )
            continue
        valid[identifier].append(trace)

    duplicate_ids = {
        identifier
        for identifier, names in observed.items()
        if identifier in expected_case_ids and len(names) > 1
    }
    for identifier in sorted(duplicate_ids):
        reason = "Duplicate {0} traces share the same case_id.".format(side)
        blocked[identifier].append(reason)
        issues.append(
            _issue(
                "duplicate_trace_case_id",
                reason,
                case_id=identifier,
                side=side,
            )
        )

    index = {
        identifier: traces[0]
        for identifier, traces in valid.items()
        if len(traces) == 1 and identifier not in duplicate_ids and not blocked.get(identifier)
    }
    return index, dict(blocked), issues


def _established_first_layer(diagnosis: Mapping[str, Any]) -> Optional[str]:
    attribution = diagnosis.get("attribution")
    if not isinstance(attribution, Mapping):
        return None
    sufficiency = diagnosis.get("evidence_sufficiency")
    if (
        isinstance(sufficiency, Mapping)
        and sufficiency.get("supports_first_broken_layer") is False
    ):
        return None
    value = attribution.get("first_broken_layer")
    return str(value) if value else None


def _project_sufficiency(diagnosis: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    value = diagnosis.get("evidence_sufficiency")
    if not isinstance(value, Mapping):
        return None
    return {
        "status": value.get("status"),
        "supports_first_broken_layer": value.get("supports_first_broken_layer"),
    }


def _project_probe(diagnosis: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    value = diagnosis.get("minimal_next_probe")
    if not isinstance(value, Mapping):
        return None
    projected = {
        key: value.get(key)
        for key in (
            "probe_type",
            "target_layer",
            "required_evidence",
            "action",
            "reason",
            "expected_disambiguation",
        )
    }
    return projected


def _inconclusive_case(
    case_id: Optional[str],
    reasons: Sequence[str],
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "verdict": "inconclusive",
        "compatibility_warnings": [],
        "comparison_limitations": list(dict.fromkeys(reasons)),
        "implementation_changes": [],
        "observed_latency_deltas_ms": {
            "retrieval": None,
            "generation": None,
        },
        "before_first_broken_layer": None,
        "after_first_broken_layer": None,
        "first_broken_layer_changed": False,
        "after_evidence_sufficiency": None,
        "minimal_next_probe": None,
    }


def _matched_case(
    case: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    comparison = compare_rag(
        case,
        before,
        after,
        comparison_policy=RAG_COMPARISON_POLICY_QUALITY,
    )
    before_diagnosis = diagnose_rag(case, before)
    after_diagnosis = diagnose_rag(case, after)
    before_first = _established_first_layer(before_diagnosis)
    after_first = _established_first_layer(after_diagnosis)
    return {
        "case_id": comparison.get("case_id"),
        "verdict": comparison.get("verdict"),
        "compatibility_warnings": list(
            comparison.get("compatibility_warnings") or []
        ),
        "comparison_limitations": list(
            comparison.get("comparison_limitations") or []
        ),
        "implementation_changes": list(
            comparison.get("implementation_changes") or []
        ),
        "observed_latency_deltas_ms": {
            "retrieval": comparison.get("changes", {}).get(
                "retrieval_latency_ms_delta"
            ),
            "generation": comparison.get("changes", {}).get(
                "generation_total_ms_delta"
            ),
        },
        "before_first_broken_layer": before_first,
        "after_first_broken_layer": after_first,
        "first_broken_layer_changed": before_first != after_first,
        "after_evidence_sufficiency": _project_sufficiency(after_diagnosis),
        "minimal_next_probe": _project_probe(after_diagnosis),
    }


def _build_report(
    case_results: Sequence[Dict[str, Any]],
    input_issues: Sequence[Dict[str, Any]],
    *,
    total_cases: int,
) -> Dict[str, Any]:
    counts = Counter(str(item.get("verdict") or "inconclusive") for item in case_results)
    regressed = counts["regressed"]
    unresolved = counts["inconclusive"] + counts["incompatible"]
    if regressed:
        status = "BLOCKED"
        exit_code = 1
    elif unresolved or input_issues:
        status = "INCONCLUSIVE"
        exit_code = 2
    else:
        status = "PASS"
        exit_code = 0
    return {
        "schema_version": RAG_GATE_SCHEMA_VERSION,
        "timestamp": utc_now(),
        "inferdoctor_version": __version__,
        "status": status,
        "exit_code": exit_code,
        "summary": {
            "total_cases": total_cases,
            "improved": counts["improved"],
            "unchanged": counts["unchanged"],
            "regressed": regressed,
            "inconclusive": counts["inconclusive"],
            "incompatible": counts["incompatible"],
            "unresolved": unresolved,
            "input_issues": len(input_issues),
        },
        "cases": list(case_results),
        "input_issues": list(input_issues),
    }


def run_rag_gate(
    cases_path: str | Path,
    before_directory: str | Path,
    after_directory: str | Path,
) -> Dict[str, Any]:
    cases, case_reasons, issues = _load_case_entries(cases_path)
    if not cases:
        return _build_report([], issues, total_cases=0)

    expected_ids = {
        identifier
        for identifier in (_case_id(case) for case in cases)
        if identifier is not None
    }
    before, before_blocked, before_issues = _load_trace_directory(
        before_directory,
        side="before",
        expected_case_ids=expected_ids,
    )
    after, after_blocked, after_issues = _load_trace_directory(
        after_directory,
        side="after",
        expected_case_ids=expected_ids,
    )
    issues.extend(before_issues)
    issues.extend(after_issues)

    case_results: List[Dict[str, Any]] = []
    for case, initial_reasons in zip(cases, case_reasons):
        identifier = _case_id(case)
        reasons = list(initial_reasons)
        if identifier is not None and not reasons:
            reasons.extend(before_blocked.get(identifier, []))
            reasons.extend(after_blocked.get(identifier, []))
            if identifier not in before and identifier not in before_blocked:
                reason = "Missing before trace for this case_id."
                reasons.append(reason)
                issues.append(
                    _issue(
                        "missing_before_trace",
                        reason,
                        case_id=identifier,
                        side="before",
                    )
                )
            if identifier not in after and identifier not in after_blocked:
                reason = "Missing after trace for this case_id."
                reasons.append(reason)
                issues.append(
                    _issue(
                        "missing_after_trace",
                        reason,
                        case_id=identifier,
                        side="after",
                    )
                )
        if identifier is None and not reasons:
            reasons.append("Case has no usable case_id for trace matching.")
        if reasons:
            case_results.append(_inconclusive_case(identifier, reasons))
            continue
        case_results.append(
            _matched_case(
                case,
                before[identifier],
                after[identifier],
            )
        )

    return _build_report(case_results, issues, total_cases=len(cases))


def rag_gate_exit_code(result: Mapping[str, Any]) -> int:
    value = result.get("exit_code")
    if isinstance(value, int) and value in {0, 1, 2}:
        return value
    return {"PASS": 0, "BLOCKED": 1}.get(str(result.get("status")), 2)


def _one_line(value: Any, *, limit: int = 300) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:limit]


def _case_reasons(item: Mapping[str, Any]) -> List[str]:
    values = list(item.get("compatibility_warnings") or [])
    values.extend(item.get("comparison_limitations") or [])
    return list(dict.fromkeys(_one_line(value) for value in values if value))


def _render_console(result: Mapping[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    lines = [
        "InferDoctor RAG Quality Gate",
        "=" * 76,
        "{0:<18}{1:>6}".format("Cases:", summary.get("total_cases", 0)),
        "{0:<18}{1:>6}".format("Improved:", summary.get("improved", 0)),
        "{0:<18}{1:>6}".format("Unchanged:", summary.get("unchanged", 0)),
        "{0:<18}{1:>6}".format("Regressed:", summary.get("regressed", 0)),
        "{0:<18}{1:>6}".format("Inconclusive:", summary.get("unresolved", 0)),
    ]
    if summary.get("input_issues"):
        lines.append(
            "{0:<18}{1:>6}".format(
                "Input issues:",
                summary.get("input_issues", 0),
            )
        )
    lines.extend(["", "QUALITY GATE: {0}".format(result.get("status", "INCONCLUSIVE"))])
    if result.get("status") == "PASS":
        lines.append("Safe to proceed from the available evaluated Cases and evidence.")
    elif result.get("status") == "BLOCKED":
        lines.append("At least one established quality regression blocks this change.")
    else:
        lines.append("The change cannot be cleared from the available evidence.")

    regressions = [
        item for item in result.get("cases", []) if item.get("verdict") == "regressed"
    ]
    if regressions:
        lines.extend(["", "Regressions", "-" * 76])
        for item in regressions:
            lines.append(_one_line(item.get("case_id") or "(missing case_id)"))
            lines.append(
                "  First broken layer: {0}".format(
                    _one_line(item.get("after_first_broken_layer") or "not established")
                )
            )
            sufficiency = item.get("after_evidence_sufficiency")
            status = (
                sufficiency.get("status")
                if isinstance(sufficiency, Mapping)
                else "UNKNOWN"
            )
            lines.append("  Evidence sufficiency: {0}".format(_one_line(status)))
            probe = item.get("minimal_next_probe")
            if isinstance(probe, Mapping):
                lines.append("  Next probe: {0}".format(_one_line(probe.get("probe_type"))))
                lines.append("  Action: {0}".format(_one_line(probe.get("action"))))

    inconclusive = [
        item
        for item in result.get("cases", [])
        if item.get("verdict") in {"inconclusive", "incompatible"}
    ]
    if inconclusive:
        lines.extend(["", "Inconclusive", "-" * 76])
        for item in inconclusive:
            lines.append(_one_line(item.get("case_id") or "(missing case_id)"))
            reasons = _case_reasons(item)
            lines.append(
                "  Reason: {0}".format(
                    "; ".join(reasons) if reasons else "Comparison evidence is incomplete."
                )
            )

    input_issues = list(result.get("input_issues") or [])
    if input_issues:
        lines.extend(["", "Input issues", "-" * 76])
        for item in input_issues[:20]:
            case_suffix = (
                " ({0})".format(_one_line(item.get("case_id")))
                if item.get("case_id")
                else ""
            )
            lines.append("- {0}{1}".format(_one_line(item.get("message")), case_suffix))
        if len(input_issues) > 20:
            lines.append(
                "- {0} additional input issues omitted from console output.".format(
                    len(input_issues) - 20
                )
            )
    return "\n".join(lines)


def _render_markdown(result: Mapping[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    lines = [
        "# InferDoctor RAG Quality Gate",
        "",
        "**QUALITY GATE: {0}**".format(result.get("status", "INCONCLUSIVE")),
        "",
        "| Result | Count |",
        "| --- | ---: |",
        "| Cases | {0} |".format(summary.get("total_cases", 0)),
        "| Improved | {0} |".format(summary.get("improved", 0)),
        "| Unchanged | {0} |".format(summary.get("unchanged", 0)),
        "| Regressed | {0} |".format(summary.get("regressed", 0)),
        "| Inconclusive | {0} |".format(summary.get("unresolved", 0)),
    ]
    regressions = [
        item for item in result.get("cases", []) if item.get("verdict") == "regressed"
    ]
    if regressions:
        lines.extend(["", "## Regressions"])
        for item in regressions:
            lines.extend(
                [
                    "",
                    "### {0}".format(_one_line(item.get("case_id") or "(missing case_id)")),
                    "- First broken layer: `{0}`".format(
                        _one_line(
                            item.get("after_first_broken_layer")
                            or "not established"
                        )
                    ),
                ]
            )
            sufficiency = item.get("after_evidence_sufficiency")
            if isinstance(sufficiency, Mapping):
                lines.append(
                    "- Evidence sufficiency: `{0}`".format(
                        _one_line(sufficiency.get("status"))
                    )
                )
            probe = item.get("minimal_next_probe")
            if isinstance(probe, Mapping):
                lines.append(
                    "- Minimal next probe: `{0}`".format(
                        _one_line(probe.get("probe_type"))
                    )
                )
                lines.append("- Action: {0}".format(_one_line(probe.get("action"))))
    inconclusive = [
        item
        for item in result.get("cases", [])
        if item.get("verdict") in {"inconclusive", "incompatible"}
    ]
    if inconclusive:
        lines.extend(["", "## Inconclusive"])
        for item in inconclusive:
            reasons = _case_reasons(item)
            lines.append(
                "- **{0}:** {1}".format(
                    _one_line(item.get("case_id") or "(missing case_id)"),
                    "; ".join(reasons) if reasons else "Comparison evidence is incomplete.",
                )
            )
    input_issues = list(result.get("input_issues") or [])
    if input_issues:
        lines.extend(["", "## Input issues"])
        lines.extend(
            "- {0}".format(_one_line(item.get("message")))
            for item in input_issues
        )
    return "\n".join(lines)


def render_rag_gate(result: Mapping[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if output_format == "markdown":
        return _render_markdown(result)
    return _render_console(result)
