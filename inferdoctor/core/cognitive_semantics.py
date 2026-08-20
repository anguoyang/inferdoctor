from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from inferdoctor.core.cognitive import (
    COGNITIVE_LAYER_ORDER,
    analyze_cognitive_trace,
)


COGNITIVE_CASE_SCHEMA_VERSION = (
    "inferdoctor.cognitive.case.v1"
)


def semantic_value_sha256(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _expected_hash(
    value: Any,
) -> Optional[str]:
    if isinstance(value, str) and value:
        return semantic_value_sha256(
            value
        )

    if not isinstance(value, dict):
        return None

    hashed = value.get("sha256")

    if isinstance(hashed, str) and hashed:
        return hashed

    plain = value.get("value")

    if isinstance(plain, str) and plain:
        return semantic_value_sha256(
            plain
        )

    return None


def _expected_node_id(
    value: Any,
) -> Optional[str]:
    if not isinstance(value, dict):
        return None

    node_id = value.get("node_id")

    if (
        isinstance(node_id, str)
        and node_id
    ):
        return node_id

    return None


def validate_cognitive_case(
    case: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []

    if (
        case.get("schema_version")
        != COGNITIVE_CASE_SCHEMA_VERSION
    ):
        errors.append(
            "schema_version must be "
            + COGNITIVE_CASE_SCHEMA_VERSION
        )

    if not isinstance(
        case.get("case_id"),
        str,
    ) or not case.get("case_id"):
        errors.append(
            "case_id is required"
        )

    expected_fields = (
        "expected_intent",
        "expected_route",
        "expected_tool",
        "expected_sources",
    )

    if not any(
        field in case
        for field in expected_fields
    ):
        errors.append(
            "at least one semantic expectation is required"
        )

    for field in (
        "expected_intent",
        "expected_route",
    ):
        if field not in case:
            continue

        if _expected_hash(
            case.get(field)
        ) is None:
            errors.append(
                field
                + " must be a string or an object "
                + "containing value or sha256"
            )

    if "expected_tool" in case:
        tool = case.get(
            "expected_tool"
        )

        if not isinstance(
            tool,
            str,
        ) or not tool:
            errors.append(
                "expected_tool must be a non-empty string"
            )

    if "expected_sources" in case:
        sources = case.get(
            "expected_sources"
        )

        if not isinstance(
            sources,
            list,
        ) or any(
            not isinstance(
                item,
                str,
            )
            or not item
            for item in (
                sources
                if isinstance(
                    sources,
                    list,
                )
                else []
            )
        ):
            errors.append(
                "expected_sources must be a list of non-empty strings"
            )

    return errors


def _layer_observations(
    trace: Dict[str, Any],
    layer: str,
) -> List[Dict[str, Any]]:
    observations = trace.get(
        "observations"
    )

    if not isinstance(
        observations,
        list,
    ):
        return []

    return [
        item
        for item in observations
        if (
            isinstance(item, dict)
            and item.get("layer")
            == layer
        )
    ]


def _decision_result(
    trace: Dict[str, Any],
    layer: str,
    expectation: Any,
) -> Tuple[str, str]:
    expected = _expected_hash(
        expectation
    )

    if expected is None:
        return (
            "UNKNOWN",
            "expected decision is not evaluable",
        )

    node_id = _expected_node_id(
        expectation
    )

    observations = (
        _layer_observations(
            trace,
            layer,
        )
    )

    if node_id:
        observations = [
            item
            for item in observations
            if item.get("node_id")
            == node_id
        ]

    hashes = {
        item.get(
            "decision_sha256"
        )
        for item in observations
        if isinstance(
            item.get(
                "decision_sha256"
            ),
            str,
        )
    }

    if not hashes:
        return (
            "UNKNOWN",
            "decision hash was not captured",
        )

    if expected in hashes:
        return (
            "PASS",
            "observed decision matches expected decision hash",
        )

    return (
        "FAIL",
        "observed decision does not match expected decision hash",
    )


def _tool_result(
    trace: Dict[str, Any],
    expected_tool: str,
) -> Tuple[str, str]:
    observations = (
        _layer_observations(
            trace,
            "action",
        )
    )

    observed = {
        item.get("tool_name")
        for item in observations
        if isinstance(
            item.get("tool_name"),
            str,
        )
    }

    if not observed:
        return (
            "UNKNOWN",
            "tool selection was not captured",
        )

    if expected_tool in observed:
        return (
            "PASS",
            "expected tool was selected",
        )

    return (
        "FAIL",
        "expected tool was not selected",
    )


def _source_result(
    trace: Dict[str, Any],
    expected_sources: Sequence[str],
) -> Tuple[str, str]:
    observations = (
        _layer_observations(
            trace,
            "retrieval",
        )
    )

    source_observations = [
        item
        for item in observations
        if isinstance(
            item.get("source_ids"),
            list,
        )
    ]

    if not source_observations:
        return (
            "UNKNOWN",
            "retrieved source IDs were not captured",
        )

    observed = {
        str(source_id)
        for item in source_observations
        for source_id
        in item.get(
            "source_ids",
            [],
        )
    }

    missing = [
        source_id
        for source_id
        in expected_sources
        if source_id
        not in observed
    ]

    if not missing:
        return (
            "PASS",
            "all expected sources were retrieved",
        )

    return (
        "FAIL",
        "one or more expected sources were not retrieved",
    )


def evaluate_cognitive_case(
    case: Dict[str, Any],
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    errors = validate_cognitive_case(
        case
    )

    if errors:
        raise ValueError(
            "; ".join(errors)
        )

    base = analyze_cognitive_trace(
        trace
    )

    semantic: Dict[
        str,
        Dict[str, Any],
    ] = {}

    if "expected_intent" in case:
        status, evidence = (
            _decision_result(
                trace,
                "intent",
                case[
                    "expected_intent"
                ],
            )
        )

        semantic["intent"] = {
            "status": status,
            "evidence": evidence,
        }

    if "expected_route" in case:
        status, evidence = (
            _decision_result(
                trace,
                "route",
                case[
                    "expected_route"
                ],
            )
        )

        semantic["route"] = {
            "status": status,
            "evidence": evidence,
        }

    if "expected_tool" in case:
        status, evidence = (
            _tool_result(
                trace,
                case[
                    "expected_tool"
                ],
            )
        )

        semantic["action"] = {
            "status": status,
            "evidence": evidence,
        }

    if "expected_sources" in case:
        status, evidence = (
            _source_result(
                trace,
                case[
                    "expected_sources"
                ],
            )
        )

        semantic["retrieval"] = {
            "status": status,
            "evidence": evidence,
        }

    order = {
        layer: index
        for index, layer
        in enumerate(
            COGNITIVE_LAYER_ORDER
        )
    }

    failed_layers = [
        layer
        for layer, result
        in semantic.items()
        if result["status"]
        == "FAIL"
    ]

    first_broken = (
        min(
            failed_layers,
            key=lambda layer: (
                order.get(
                    layer,
                    999,
                )
            ),
        )
        if failed_layers
        else None
    )

    first_order = (
        order.get(
            first_broken
        )
        if first_broken
        else None
    )

    for layer in base["layers"]:
        name = layer["layer"]

        result = semantic.get(
            name
        )

        if result is None:
            layer[
                "semantic_status"
            ] = "NOT_EVALUATED"

            layer[
                "semantic_role"
            ] = "NOT_EVALUATED"

            continue

        layer[
            "semantic_status"
        ] = result["status"]

        layer[
            "semantic_evidence"
        ] = result["evidence"]

        if name == first_broken:
            role = "FIRST_BROKEN"

        elif (
            first_order is not None
            and order.get(
                name,
                999,
            )
            > first_order
        ):
            role = (
                "DOWNSTREAM_OBSERVATION"
            )

        elif (
            result["status"]
            == "PASS"
        ):
            role = (
                "ESTABLISHED_UPSTREAM"
            )

        else:
            role = "OBSERVATION"

        layer[
            "semantic_role"
        ] = role

    statuses = [
        result["status"]
        for result
        in semantic.values()
    ]

    if not statuses:
        semantic_status = (
            "NOT_EVALUATED"
        )

    elif "FAIL" in statuses:
        semantic_status = "FAIL"

    elif "UNKNOWN" in statuses:
        semantic_status = (
            "INCOMPLETE"
        )

    else:
        semantic_status = "PASS"

    base["case_id"] = case.get(
        "case_id"
    )

    base[
        "semantic_status"
    ] = semantic_status

    base[
        "first_broken_layer"
    ] = first_broken

    base[
        "first_broken_layer_status"
    ] = (
        "ESTABLISHED"
        if first_broken
        else "NOT_ESTABLISHED"
    )

    base[
        "semantic_expectations"
    ] = {
        layer: {
            "status": result[
                "status"
            ],
            "evidence": result[
                "evidence"
            ],
        }
        for layer, result
        in semantic.items()
    }

    return base


def init_cognitive_case_template(
    output: str,
) -> Path:
    path = Path(output)

    template = {
        "schema_version": (
            COGNITIVE_CASE_SCHEMA_VERSION
        ),
        "case_id": "synthetic-cognitive-case",
        "expected_intent": {
            "value": "refund_request",
        },
        "expected_route": {
            "value": "refund_route",
        },
        "expected_tool": "crm_lookup",
        "expected_sources": [
            "policy-document"
        ],
    }

    path.write_text(
        json.dumps(
            template,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def load_cognitive_case(
    path: str,
) -> Dict[str, Any]:
    data = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Cognitive Case must be a JSON object"
        )

    errors = validate_cognitive_case(
        data
    )

    if errors:
        raise ValueError(
            "; ".join(errors)
        )

    return data


def load_cognitive_trace(
    path: str,
) -> Dict[str, Any]:
    data = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Cognitive Trace must be a JSON object"
        )

    if (
        data.get("schema_version")
        == "inferdoctor.cognitive.capture.v1"
    ):
        trace = data.get("trace")

        if not isinstance(
            trace,
            dict,
        ):
            raise ValueError(
                "Cognitive capture does not contain a trace object"
            )

        data = trace

    if not isinstance(
        data.get("observations"),
        list,
    ):
        raise ValueError(
            "Cognitive Trace must contain observations"
        )

    return data


def validate_cognitive_case_file(
    path: str,
) -> Dict[str, Any]:
    try:
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "status": "FAIL",
            "errors": [
                str(exc)
            ],
        }

    if not isinstance(data, dict):
        return {
            "status": "FAIL",
            "errors": [
                "Cognitive Case must be a JSON object"
            ],
        }

    errors = validate_cognitive_case(
        data
    )

    return {
        "status": (
            "FAIL"
            if errors
            else "PASS"
        ),
        "errors": errors,
        "case_id": data.get(
            "case_id"
        ),
    }

