from __future__ import annotations

from typing import Any, Dict, Optional

from inferdoctor.core.cognitive import (
    COGNITIVE_LAYER_ORDER,
)


COGNITIVE_PROBE_PLAN_SCHEMA_VERSION = (
    "inferdoctor.cognitive.probe_plan.v1"
)


PROBE_BY_LAYER = {
    "intent": "gold_intent",
    "route": "gold_route",
    "plan": "gold_plan",
    "action": "gold_tool_result",
    "retrieval": "gold_context",
    "context": "gold_context",
    "generation": "model_capability",
    "postprocessing": "raw_vs_final",
}


def _layer_result(
    analysis: Dict[str, Any],
    layer: str,
) -> Optional[Dict[str, Any]]:
    layers = analysis.get("layers")

    if not isinstance(layers, list):
        return None

    for item in layers:
        if (
            isinstance(item, dict)
            and item.get("layer") == layer
        ):
            return item

    return None


def _unknown_semantic_layers(
    analysis: Dict[str, Any],
) -> list[str]:
    unknown = []

    for layer in COGNITIVE_LAYER_ORDER:
        item = _layer_result(
            analysis,
            layer,
        )

        if not item:
            continue

        status = item.get(
            "semantic_status"
        )

        if status in {
            "UNKNOWN",
            "NOT_EVALUATED",
        }:
            unknown.append(layer)

    return unknown


def plan_next_cognitive_probe(
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    first_broken = analysis.get(
        "first_broken_layer"
    )

    semantic_status = analysis.get(
        "semantic_status"
    )

    execution_status = analysis.get(
        "execution_status"
    )

    if (
        isinstance(first_broken, str)
        and first_broken
    ):
        probe = PROBE_BY_LAYER.get(
            first_broken
        )

        if probe is None:
            return {
                "schema_version": (
                    COGNITIVE_PROBE_PLAN_SCHEMA_VERSION
                ),
                "status": "INCONCLUSIVE",
                "next_probe": None,
                "target_layer": (
                    first_broken
                ),
                "reason": (
                    "No controlled probe is defined "
                    "for the established first "
                    "broken layer."
                ),
                "goal": (
                    "Add a framework-neutral "
                    "controlled experiment for "
                    "this layer."
                ),
            }

        goals = {
            "gold_intent": (
                "Force or supply the known-good "
                "intent, then replay downstream "
                "processing without changing "
                "later layers."
            ),
            "gold_route": (
                "Force the known-good route, then "
                "replay downstream processing "
                "without changing later layers."
            ),
            "gold_plan": (
                "Supply a known-good plan while "
                "keeping tools, retrieval, and "
                "generation unchanged."
            ),
            "gold_tool_result": (
                "Supply the expected tool result "
                "or bypass tool selection while "
                "keeping downstream processing "
                "unchanged."
            ),
            "gold_context": (
                "Bypass retrieval/context assembly "
                "with known-good evidence and test "
                "whether downstream generation "
                "can answer correctly."
            ),
            "model_capability": (
                "Test the same task with all known "
                "upstream evidence fixed before "
                "attributing failure to model "
                "capability."
            ),
            "raw_vs_final": (
                "Compare raw generation with the "
                "final answer to isolate damage "
                "introduced after generation."
            ),
        }

        return {
            "schema_version": (
                COGNITIVE_PROBE_PLAN_SCHEMA_VERSION
            ),
            "status": "PROBE_RECOMMENDED",
            "next_probe": probe,
            "target_layer": (
                first_broken
            ),
            "reason": (
                "The earliest supported semantic "
                "failure is {0}.".format(
                    first_broken
                )
            ),
            "goal": goals[probe],
            "change_one_variable_only": True,
            "do_not_change": [
                layer
                for layer
                in COGNITIVE_LAYER_ORDER
                if layer != first_broken
            ],
            "unsafe_conclusion_to_avoid": (
                "Do not blame downstream layers "
                "or model capability until the "
                "controlled replay moves the "
                "failure downstream or clears it."
            ),
        }

    unknown = _unknown_semantic_layers(
        analysis
    )

    if (
        execution_status == "FAIL"
        and not first_broken
    ):
        return {
            "schema_version": (
                COGNITIVE_PROBE_PLAN_SCHEMA_VERSION
            ),
            "status": (
                "EXECUTION_FAILURE_FIRST"
            ),
            "next_probe": None,
            "target_layer": (
                analysis.get(
                    "first_execution_failure"
                )
            ),
            "reason": (
                "A runtime execution failure is "
                "established before a semantic "
                "failure can be evaluated."
            ),
            "goal": (
                "Fix the execution failure before "
                "running semantic Gold Probes."
            ),
        }

    if semantic_status in {
        "INCOMPLETE",
        "NOT_EVALUATED",
    }:
        return {
            "schema_version": (
                COGNITIVE_PROBE_PLAN_SCHEMA_VERSION
            ),
            "status": (
                "MORE_EVIDENCE_REQUIRED"
            ),
            "next_probe": (
                "capture_missing_evidence"
            ),
            "target_layer": (
                unknown[0]
                if unknown
                else None
            ),
            "reason": (
                "No semantic broken layer is "
                "established because required "
                "evidence is missing."
            ),
            "goal": (
                "Capture the smallest missing "
                "semantic evidence before running "
                "a controlled probe."
            ),
            "missing_or_unevaluated_layers": (
                unknown
            ),
        }

    if (
        semantic_status == "PASS"
        and execution_status
        in {
            "PASS",
            "UNKNOWN",
        }
    ):
        return {
            "schema_version": (
                COGNITIVE_PROBE_PLAN_SCHEMA_VERSION
            ),
            "status": "NO_PROBE_NEEDED",
            "next_probe": None,
            "target_layer": None,
            "reason": (
                "All currently defined semantic "
                "expectations passed."
            ),
            "goal": (
                "Add expectations for later layers "
                "only if the application output "
                "is still unacceptable."
            ),
        }

    return {
        "schema_version": (
            COGNITIVE_PROBE_PLAN_SCHEMA_VERSION
        ),
        "status": "INCONCLUSIVE",
        "next_probe": None,
        "target_layer": None,
        "reason": (
            "Available evidence does not identify "
            "a safe next controlled experiment."
        ),
        "goal": (
            "Capture additional evidence without "
            "changing multiple variables."
        ),
    }


def render_probe_plan(
    plan: Dict[str, Any],
) -> str:
    lines = [
        "Cognitive Probe Plan",
        "====================",
        "status: {0}".format(
            plan.get("status")
        ),
        "next_probe: {0}".format(
            plan.get("next_probe")
            or "none"
        ),
        "target_layer: {0}".format(
            plan.get("target_layer")
            or "none"
        ),
        "",
        "Reason:",
        str(
            plan.get("reason")
            or ""
        ),
        "",
        "Goal:",
        str(
            plan.get("goal")
            or ""
        ),
    ]

    if plan.get(
        "change_one_variable_only"
    ):
        lines.extend([
            "",
            (
                "Rule: change exactly one "
                "controlled variable."
            ),
        ])

    warning = plan.get(
        "unsafe_conclusion_to_avoid"
    )

    if warning:
        lines.extend([
            "",
            "Avoid:",
            str(warning),
        ])

    return "\n".join(lines)
