from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


COGNITIVE_TRACE_SCHEMA_VERSION = (
    "inferdoctor.cognitive.trace.v1"
)

COGNITIVE_ANALYSIS_SCHEMA_VERSION = (
    "inferdoctor.cognitive.analysis.v1"
)

COGNITIVE_LAYER_ORDER = (
    "intent",
    "route",
    "plan",
    "action",
    "retrieval",
    "context",
    "generation",
    "postprocessing",
)

_EXECUTION_FAIL = {
    "failed",
    "failure",
    "error",
    "exception",
    "stopped",
}

_EXECUTION_WARN = {
    "retry",
    "retrying",
    "paused",
}

_EXECUTION_PASS = {
    "succeeded",
    "success",
    "completed",
}


def _normalized_status(
    value: Any,
) -> str:
    return str(
        value or ""
    ).strip().casefold()


def _execution_state(
    observations: Sequence[
        Dict[str, Any]
    ],
) -> str:
    if not observations:
        return "UNKNOWN"

    statuses = {
        _normalized_status(
            item.get("status")
        )
        for item in observations
    }

    if statuses & _EXECUTION_FAIL:
        return "FAIL"

    if statuses & _EXECUTION_WARN:
        return "WARN"

    if statuses & _EXECUTION_PASS:
        return "PASS"

    return "OBSERVED"


def analyze_cognitive_trace(
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    raw_observations = trace.get(
        "observations"
    )

    observations = (
        raw_observations
        if isinstance(
            raw_observations,
            list,
        )
        else []
    )

    layer_results: List[
        Dict[str, Any]
    ] = []

    first_execution_failure: Optional[
        str
    ] = None

    first_execution_warning: Optional[
        str
    ] = None

    for layer in COGNITIVE_LAYER_ORDER:
        layer_observations = [
            item
            for item in observations
            if (
                isinstance(
                    item,
                    dict,
                )
                and item.get("layer")
                == layer
            )
        ]

        execution_status = (
            _execution_state(
                layer_observations
            )
        )

        if (
            execution_status == "FAIL"
            and first_execution_failure
            is None
        ):
            first_execution_failure = (
                layer
            )

        if (
            execution_status == "WARN"
            and first_execution_warning
            is None
        ):
            first_execution_warning = (
                layer
            )

        node_types = sorted({
            str(
                item.get(
                    "node_type"
                )
            )
            for item
            in layer_observations
            if item.get("node_type")
        })

        layer_results.append({
            "layer": layer,
            "execution_status": (
                execution_status
            ),
            "semantic_status": (
                "UNKNOWN"
            ),
            "observation_count": len(
                layer_observations
            ),
            "node_types": node_types,
            "observations": (
                layer_observations
            ),
        })

    if first_execution_failure:
        execution_status = "FAIL"

    elif first_execution_warning:
        execution_status = "WARN"

    elif any(
        item["execution_status"]
        in {
            "PASS",
            "OBSERVED",
        }
        for item in layer_results
    ):
        execution_status = "PASS"

    else:
        execution_status = "UNKNOWN"

    return {
        "schema_version": (
            COGNITIVE_ANALYSIS_SCHEMA_VERSION
        ),
        "source_trace_schema_version": (
            trace.get("schema_version")
        ),
        "execution_status": (
            execution_status
        ),
        "semantic_status": (
            "NOT_EVALUATED"
        ),
        "first_execution_failure": (
            first_execution_failure
        ),
        "first_execution_warning": (
            first_execution_warning
        ),
        "first_broken_layer": None,
        "first_broken_layer_status": (
            "NOT_ESTABLISHED"
        ),
        "layers": layer_results,
        "unmapped_node_types": (
            trace.get(
                "unmapped_node_types",
                [],
            )
        ),
        "limitations": [
            (
                "Execution success does not prove semantic correctness."
            ),
            (
                "Intent, routing, planning, and tool-choice correctness require expected outcomes or controlled probes."
            ),
            (
                "first_broken_layer remains unset until semantic evidence is available."
            ),
        ],
    }


def render_cognitive_analysis(
    analysis: Dict[str, Any],
) -> str:
    lines = [
        "Cognitive Path Analysis",
        "=======================",
        "execution_status: {0}".format(
            analysis.get(
                "execution_status"
            )
        ),
        "semantic_status: {0}".format(
            analysis.get(
                "semantic_status"
            )
        ),
        "",
        "Layers:",
    ]

    first_execution_failure = (
        analysis.get(
            "first_execution_failure"
        )
    )

    for layer in analysis.get(
        "layers",
        [],
    ):
        if not isinstance(
            layer,
            dict,
        ):
            continue

        name = str(
            layer.get("layer")
        )

        execution = str(
            layer.get(
                "execution_status"
            )
        )

        semantic = str(
            layer.get(
                "semantic_status"
            )
        )

        markers = []

        if (
            name
            == first_execution_failure
        ):
            markers.append(
                "FIRST EXECUTION FAILURE"
            )

        if (
            layer.get(
                "semantic_role"
            )
            == "FIRST_BROKEN"
        ):
            markers.append(
                "FIRST SEMANTIC BROKEN"
            )

        marker = (
            "  <-- "
            + ", ".join(markers)
            if markers
            else ""
        )

        lines.append(
            "- {0}: execution={1}, semantic={2}{3}".format(
                name,
                execution,
                semantic,
                marker,
            )
        )

    first_broken = analysis.get(
        "first_broken_layer"
    )

    lines.extend([
        "",
        "Semantic first broken layer: {0}".format(
            first_broken
            or "not established"
        ),
        (
            "Note: a node that executed "
            "successfully may still have made "
            "the wrong semantic decision."
        ),
    ])

    return "\n".join(lines)
