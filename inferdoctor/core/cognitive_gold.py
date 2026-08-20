from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from inferdoctor.core.rag import (
    run_gold_context_probe,
)


COGNITIVE_GOLD_CONTEXT_SCHEMA_VERSION = (
    "inferdoctor.cognitive.gold_context.v1"
)


def interpret_gold_context_probe(
    baseline_analysis: Dict[str, Any],
    probe_result: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_first = (
        baseline_analysis.get(
            "first_broken_layer"
        )
    )

    transport = str(
        probe_result.get(
            "transport_status"
        )
        or "unknown"
    ).casefold()

    overall = str(
        probe_result.get(
            "overall_status"
        )
        or "unknown"
    ).casefold()

    evaluation = str(
        probe_result.get(
            "evaluation_status"
        )
        or "unknown"
    ).casefold()

    deterministic = bool(
        probe_result.get(
            "deterministic_checks_available"
        )
    )

    if transport == "fail":
        verdict = "PROBE_FAILED"
        confidence = "low"

        conclusion = (
            "The Gold Context request failed "
            "before model capability could be "
            "evaluated."
        )

        model_interpretation = (
            "Model capability remains unknown."
        )

    elif overall == "pass":
        verdict = "GOLD_CONTEXT_PASS"
        confidence = "high"

        conclusion = (
            "The model satisfied the deterministic "
            "answer checks when supplied known-good "
            "context and explicit grounding."
        )

        model_interpretation = (
            "The model demonstrated that it can "
            "use known-good evidence for this case. "
            "Model size or raw reasoning capability "
            "should not be treated as the first "
            "suspect for this evaluated task."
        )

    elif overall == "fail":
        verdict = "GOLD_CONTEXT_FAIL"
        confidence = (
            "high"
            if deterministic
            else "low"
        )

        conclusion = (
            "The model still failed the evaluated "
            "answer checks even with known-good "
            "context and explicit grounding."
        )

        model_interpretation = (
            "A downstream generation, prompting, "
            "or model-capability limitation remains "
            "plausible."
        )

    elif overall == "inconclusive":
        verdict = "INCONCLUSIVE"
        confidence = "low"

        conclusion = (
            "The Gold Context probe did not provide "
            "enough deterministic evidence for a "
            "conclusive capability result."
        )

        model_interpretation = (
            "Model capability remains unresolved."
        )

    elif overall == "dry_run":
        verdict = "DRY_RUN"
        confidence = "none"

        conclusion = (
            "Dry run only; no model request was sent."
        )

        model_interpretation = (
            "No capability conclusion is available."
        )

    else:
        verdict = "INCONCLUSIVE"
        confidence = "low"

        conclusion = (
            "The Gold Context probe returned an "
            "unrecognized or incomplete result."
        )

        model_interpretation = (
            "Model capability remains unresolved."
        )

    if (
        verdict == "GOLD_CONTEXT_PASS"
        and baseline_first
        in {
            "retrieval",
            "context",
        }
    ):
        diagnostic_effect = (
            "UPSTREAM_EVIDENCE_PATH_SUPPORTED"
        )

        attribution_interpretation = (
            "The baseline failure was in the "
            "retrieval/context path, while the "
            "model passed with known-good evidence. "
            "This strengthens the hypothesis that "
            "the evidence-delivery path is upstream "
            "of the observed answer failure."
        )

    elif (
        verdict == "GOLD_CONTEXT_PASS"
        and baseline_first
        == "generation"
    ):
        diagnostic_effect = (
            "GENERATION_FAILURE_NOT_REPRODUCED"
        )

        attribution_interpretation = (
            "The baseline generation failure did "
            "not reproduce when known-good context "
            "and explicit grounding were supplied. "
            "The original generation attribution "
            "therefore needs refinement."
        )

    elif (
        verdict == "GOLD_CONTEXT_FAIL"
    ):
        diagnostic_effect = (
            "DOWNSTREAM_LIMITATION_PERSISTS"
        )

        attribution_interpretation = (
            "The failure persists after bypassing "
            "normal retrieval/context construction. "
            "A downstream limitation remains "
            "supported."
        )

    else:
        diagnostic_effect = (
            "NO_ATTRIBUTION_UPDATE"
        )

        attribution_interpretation = (
            "The probe does not justify changing "
            "the current root-cause attribution."
        )

    return {
        "schema_version": (
            COGNITIVE_GOLD_CONTEXT_SCHEMA_VERSION
        ),
        "probe_name": "gold_context",
        "verdict": verdict,
        "confidence": confidence,
        "baseline_first_broken_layer": (
            baseline_first
        ),
        "transport_status": transport,
        "evaluation_status": evaluation,
        "overall_status": overall,
        "diagnostic_effect": (
            diagnostic_effect
        ),
        "conclusion": conclusion,
        "model_capability_interpretation": (
            model_interpretation
        ),
        "attribution_interpretation": (
            attribution_interpretation
        ),
        "causal_boundary": (
            "Gold Context is a capability-isolation "
            "probe, not a strict one-variable replay. "
            "It supplies known-good context and an "
            "explicit grounding prompt, so it does "
            "not by itself isolate retrieval, context "
            "selection, prompting, and grounding as "
            "separate causal variables."
        ),
        "rag_probe": probe_result,
    }


def run_cognitive_gold_context_probe(
    baseline_analysis: Dict[str, Any],
    rag_case: Dict[str, Any],
    *,
    context_text: str,
    endpoint: str,
    model: str,
    timeout: float = 30.0,
    dry_run: bool = False,
    allow_non_local: bool = False,
    allow_public: bool = False,
    api_key: Optional[str] = None,
    retain_answer: bool = False,
    probe_runner: Callable[..., Dict[str, Any]] = (
        run_gold_context_probe
    ),
) -> Dict[str, Any]:
    probe_result = probe_runner(
        rag_case,
        context_text=context_text,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        dry_run=dry_run,
        allow_non_local=allow_non_local,
        allow_public=allow_public,
        api_key=api_key,
        retain_answer=retain_answer,
    )

    return interpret_gold_context_probe(
        baseline_analysis,
        probe_result,
    )


def render_cognitive_gold_context(
    result: Dict[str, Any],
) -> str:
    lines = [
        "Cognitive Gold Context Probe",
        "============================",
        "verdict: {0}".format(
            result.get("verdict")
        ),
        "confidence: {0}".format(
            result.get("confidence")
        ),
        "baseline_first_broken_layer: {0}".format(
            result.get(
                "baseline_first_broken_layer"
            )
            or "none"
        ),
        "diagnostic_effect: {0}".format(
            result.get(
                "diagnostic_effect"
            )
        ),
        "",
        "Conclusion:",
        str(
            result.get("conclusion")
            or ""
        ),
        "",
        "Model capability:",
        str(
            result.get(
                "model_capability_interpretation"
            )
            or ""
        ),
        "",
        "Attribution:",
        str(
            result.get(
                "attribution_interpretation"
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
