from __future__ import annotations

from typing import Any, Dict, Optional

from inferdoctor.core.cognitive import (
    COGNITIVE_LAYER_ORDER,
)
from inferdoctor.core.cognitive_semantics import (
    evaluate_cognitive_case,
)


COGNITIVE_REPLAY_SCHEMA_VERSION = (
    "inferdoctor.cognitive.replay_comparison.v1"
)


def _layer_index(
    layer: Optional[str],
) -> Optional[int]:
    if not layer:
        return None

    try:
        return list(
            COGNITIVE_LAYER_ORDER
        ).index(layer)
    except ValueError:
        return None


def _semantic_layer_status(
    analysis: Dict[str, Any],
    layer: str,
) -> str:
    layers = analysis.get(
        "layers"
    )

    if not isinstance(
        layers,
        list,
    ):
        return "UNKNOWN"

    for item in layers:
        if (
            isinstance(item, dict)
            and item.get("layer")
            == layer
        ):
            return str(
                item.get(
                    "semantic_status"
                )
                or "UNKNOWN"
            )

    return "UNKNOWN"


def compare_controlled_replay(
    case: Dict[str, Any],
    before_trace: Dict[str, Any],
    after_trace: Dict[str, Any],
    *,
    target_layer: str,
    probe_name: Optional[str] = None,
) -> Dict[str, Any]:
    if (
        target_layer
        not in COGNITIVE_LAYER_ORDER
    ):
        raise ValueError(
            "unknown cognitive target layer: {0}".format(
                target_layer
            )
        )

    before = evaluate_cognitive_case(
        case,
        before_trace,
    )

    after = evaluate_cognitive_case(
        case,
        after_trace,
    )

    before_first = before.get(
        "first_broken_layer"
    )

    after_first = after.get(
        "first_broken_layer"
    )

    before_target_status = (
        _semantic_layer_status(
            before,
            target_layer,
        )
    )

    after_target_status = (
        _semantic_layer_status(
            after,
            target_layer,
        )
    )

    before_index = _layer_index(
        before_first
    )

    after_index = _layer_index(
        after_first
    )

    target_index = _layer_index(
        target_layer
    )

    moved_downstream = bool(
        before_index is not None
        and after_index is not None
        and after_index > before_index
    )

    cleared = bool(
        after_first is None
        and after.get(
            "semantic_status"
        )
        == "PASS"
    )

    target_fixed = bool(
        before_target_status
        == "FAIL"
        and after_target_status
        == "PASS"
    )

    if before_first != target_layer:
        verdict = (
            "INVALID_BASELINE"
        )

        confidence = "high"

        conclusion = (
            "The replay target was not the "
            "baseline first broken layer, so "
            "this experiment cannot validate "
            "that layer as the upstream "
            "bottleneck."
        )

    elif not target_fixed:
        verdict = (
            "TARGET_NOT_ISOLATED"
        )

        confidence = "high"

        conclusion = (
            "The target layer did not change "
            "from semantic FAIL to PASS. "
            "The controlled replay therefore "
            "did not successfully isolate or "
            "correct the target."
        )

    elif moved_downstream:
        verdict = (
            "VALIDATED_UPSTREAM_BOTTLENECK"
        )

        confidence = "high"

        conclusion = (
            "The target layer changed from "
            "FAIL to PASS and the first broken "
            "layer moved downstream. This "
            "supports the target as a validated "
            "upstream bottleneck for the "
            "evaluated expectations."
        )

    elif cleared:
        verdict = (
            "VALIDATED_AND_CLEARED"
        )

        confidence = "high"

        conclusion = (
            "The target layer changed from "
            "FAIL to PASS and no evaluated "
            "semantic failure remains. This "
            "supports the target as the "
            "upstream bottleneck for the "
            "evaluated expectations."
        )

    elif (
        after_first is not None
        and target_index is not None
        and after_index is not None
        and after_index < target_index
    ):
        verdict = "REGRESSED"

        confidence = "medium"

        conclusion = (
            "The replay exposed an earlier "
            "semantic failure than the target. "
            "The experiment does not validate "
            "the original attribution."
        )

    else:
        verdict = "INCONCLUSIVE"

        confidence = "low"

        conclusion = (
            "The target appears corrected, but "
            "the available semantic evidence "
            "does not establish a clean "
            "downstream movement or complete "
            "clearance."
        )

    return {
        "schema_version": (
            COGNITIVE_REPLAY_SCHEMA_VERSION
        ),
        "case_id": case.get(
            "case_id"
        ),
        "probe_name": probe_name,
        "target_layer": target_layer,
        "verdict": verdict,
        "confidence": confidence,
        "before": {
            "semantic_status": (
                before.get(
                    "semantic_status"
                )
            ),
            "first_broken_layer": (
                before_first
            ),
            "target_status": (
                before_target_status
            ),
        },
        "after": {
            "semantic_status": (
                after.get(
                    "semantic_status"
                )
            ),
            "first_broken_layer": (
                after_first
            ),
            "target_status": (
                after_target_status
            ),
        },
        "target_fixed": target_fixed,
        "first_broken_moved_downstream": (
            moved_downstream
        ),
        "all_evaluated_failures_cleared": (
            cleared
        ),
        "conclusion": conclusion,
        "causal_boundary": (
            "This comparison validates only "
            "the evaluated semantic expectations. "
            "It assumes the replay changed the "
            "declared target variable only; "
            "InferDoctor does not infer that "
            "experimental control from the two "
            "traces themselves."
        ),
    }


def render_replay_comparison(
    result: Dict[str, Any],
) -> str:
    before = result.get(
        "before",
        {},
    )

    after = result.get(
        "after",
        {},
    )

    lines = [
        "Controlled Replay Comparison",
        "============================",
        "verdict: {0}".format(
            result.get("verdict")
        ),
        "confidence: {0}".format(
            result.get("confidence")
        ),
        "probe: {0}".format(
            result.get("probe_name")
            or "unspecified"
        ),
        "target_layer: {0}".format(
            result.get(
                "target_layer"
            )
        ),
        "",
        "Before:",
        "- first_broken_layer: {0}".format(
            before.get(
                "first_broken_layer"
            )
            or "none"
        ),
        "- target_status: {0}".format(
            before.get(
                "target_status"
            )
        ),
        "",
        "After:",
        "- first_broken_layer: {0}".format(
            after.get(
                "first_broken_layer"
            )
            or "none"
        ),
        "- target_status: {0}".format(
            after.get(
                "target_status"
            )
        ),
        "",
        "Conclusion:",
        str(
            result.get(
                "conclusion"
            )
            or ""
        ),
        "",
        "Boundary:",
        str(
            result.get(
                "causal_boundary"
            )
            or ""
        ),
    ]

    return "\n".join(lines)
