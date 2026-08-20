from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from inferdoctor.core.cognitive import (
    COGNITIVE_TRACE_SCHEMA_VERSION,
)


OPENINFERENCE_KIND_LAYER_MAP = {
    "AGENT": "plan",
    "TOOL": "action",
    "RETRIEVER": "retrieval",
    "RERANKER": "retrieval",
    "LLM": "generation",
}


def _sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _otlp_value(
    value: Any,
) -> Any:
    if not isinstance(value, dict):
        return value

    for key in (
        "stringValue",
        "intValue",
        "doubleValue",
        "boolValue",
    ):
        if key in value:
            return value[key]

    array = value.get("arrayValue")

    if isinstance(array, dict):
        values = array.get("values")

        if isinstance(values, list):
            return [
                _otlp_value(item)
                for item in values
            ]

    kvlist = value.get("kvlistValue")

    if isinstance(kvlist, dict):
        values = kvlist.get("values")

        if isinstance(values, list):
            result = {}

            for item in values:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                key = item.get("key")

                if not isinstance(
                    key,
                    str,
                ):
                    continue

                result[key] = (
                    _otlp_value(
                        item.get("value")
                    )
                )

            return result

    return value


def _attributes(
    span: Dict[str, Any],
) -> Dict[str, Any]:
    value = span.get(
        "attributes"
    )

    if isinstance(value, dict):
        return dict(value)

    if not isinstance(value, list):
        return {}

    result = {}

    for item in value:
        if not isinstance(
            item,
            dict,
        ):
            continue

        key = item.get("key")

        if not isinstance(
            key,
            str,
        ):
            continue

        result[key] = _otlp_value(
            item.get("value")
        )

    return result


def _iter_otlp_spans(
    payload: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    resources = payload.get(
        "resourceSpans"
    )

    if not isinstance(
        resources,
        list,
    ):
        return []

    spans: List[
        Dict[str, Any]
    ] = []

    for resource in resources:
        if not isinstance(
            resource,
            dict,
        ):
            continue

        scopes = (
            resource.get(
                "scopeSpans"
            )
            or resource.get(
                "instrumentationLibrarySpans"
            )
            or []
        )

        if not isinstance(
            scopes,
            list,
        ):
            continue

        for scope in scopes:
            if not isinstance(
                scope,
                dict,
            ):
                continue

            items = scope.get(
                "spans"
            )

            if not isinstance(
                items,
                list,
            ):
                continue

            spans.extend(
                item
                for item in items
                if isinstance(
                    item,
                    dict,
                )
            )

    return spans


def extract_openinference_spans(
    payload: Any,
) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(
                item,
                dict,
            )
        ]

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "OpenInference trace must be a JSON object or list"
        )

    spans = payload.get("spans")

    if isinstance(spans, list):
        return [
            item
            for item in spans
            if isinstance(
                item,
                dict,
            )
        ]

    otlp = list(
        _iter_otlp_spans(
            payload
        )
    )

    if otlp:
        return otlp

    if (
        isinstance(
            payload.get("attributes"),
            (dict, list),
        )
    ):
        return [payload]

    raise ValueError(
        "No OpenInference or OTLP spans found"
    )


def _context_value(
    span: Dict[str, Any],
    key: str,
) -> Optional[str]:
    context = span.get(
        "context"
    )

    if isinstance(
        context,
        dict,
    ):
        value = context.get(key)

        if isinstance(
            value,
            str,
        ) and value:
            return value

    aliases = {
        "trace_id": (
            "traceId",
            "trace_id",
        ),
        "span_id": (
            "spanId",
            "span_id",
        ),
    }

    for alias in aliases.get(
        key,
        (),
    ):
        value = span.get(alias)

        if isinstance(
            value,
            str,
        ) and value:
            return value

    return None


def _parent_span_id(
    span: Dict[str, Any],
) -> Optional[str]:
    for key in (
        "parent_id",
        "parentSpanId",
        "parent_span_id",
    ):
        value = span.get(key)

        if isinstance(
            value,
            str,
        ) and value:
            return value

    return None


def _span_kind(
    span: Dict[str, Any],
) -> Optional[str]:
    attributes = _attributes(
        span
    )

    value = attributes.get(
        "openinference.span.kind"
    )

    if (
        isinstance(value, str)
        and value
    ):
        return value.upper()

    return None


def _span_start_sort_key(
    span: Dict[str, Any],
) -> Optional[float]:
    for key in (
        "startTimeUnixNano",
        "start_time_unix_nano",
    ):
        value = span.get(key)

        if (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        ):
            return float(value)

        if (
            isinstance(value, str)
            and value.isdigit()
        ):
            return float(value)

    value = span.get(
        "start_time"
    )

    if not isinstance(
        value,
        str,
    ) or not value:
        return None

    try:
        normalized = (
            value[:-1] + "+00:00"
            if value.endswith("Z")
            else value
        )

        return datetime.fromisoformat(
            normalized
        ).timestamp()

    except ValueError:
        return None


def _nearest_agent_ancestor(
    span: Dict[str, Any],
    spans_by_id: Dict[
        str,
        Dict[str, Any],
    ],
    kinds_by_id: Dict[
        str,
        Optional[str],
    ],
) -> Optional[str]:
    parent = _parent_span_id(
        span
    )

    visited = set()

    while parent:
        if parent in visited:
            return None

        visited.add(parent)

        if (
            kinds_by_id.get(
                parent
            )
            == "AGENT"
        ):
            return parent

        ancestor = spans_by_id.get(
            parent
        )

        if not isinstance(
            ancestor,
            dict,
        ):
            return None

        parent = _parent_span_id(
            ancestor
        )

    return None


def _derived_agent_plan_observations(
    spans: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    spans_by_id: Dict[
        str,
        Dict[str, Any],
    ] = {}

    kinds_by_id: Dict[
        str,
        Optional[str],
    ] = {}

    for span in spans:
        span_id = _context_value(
            span,
            "span_id",
        )

        if not span_id:
            continue

        spans_by_id[
            span_id
        ] = span

        kinds_by_id[
            span_id
        ] = _span_kind(
            span
        )

    grouped: Dict[
        str,
        List[
            tuple[
                int,
                Dict[str, Any],
                str,
            ]
        ],
    ] = {}

    for index, span in enumerate(
        spans
    ):
        if (
            _span_kind(span)
            != "TOOL"
        ):
            continue

        attributes = _attributes(
            span
        )

        tool_name = attributes.get(
            "tool.name"
        )

        if not (
            isinstance(
                tool_name,
                str,
            )
            and tool_name
        ):
            continue

        agent_span_id = (
            _nearest_agent_ancestor(
                span,
                spans_by_id,
                kinds_by_id,
            )
        )

        if not agent_span_id:
            continue

        grouped.setdefault(
            agent_span_id,
            [],
        ).append(
            (
                index,
                span,
                tool_name,
            )
        )

    observations: List[
        Dict[str, Any]
    ] = []

    for (
        agent_span_id,
        tool_items,
    ) in grouped.items():
        if len(tool_items) == 1:
            ordered = tool_items
            order_source = (
                "single_tool"
            )

        else:
            timed = [
                (
                    _span_start_sort_key(
                        span
                    ),
                    original_index,
                    span,
                    tool_name,
                )
                for (
                    original_index,
                    span,
                    tool_name,
                )
                in tool_items
            ]

            if any(
                item[0] is None
                for item in timed
            ):
                continue

            timestamps = [
                item[0]
                for item in timed
            ]

            if (
                len(set(timestamps))
                != len(timestamps)
            ):
                continue

            timed.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            ordered = [
                (
                    original_index,
                    span,
                    tool_name,
                )
                for (
                    _,
                    original_index,
                    span,
                    tool_name,
                )
                in timed
            ]

            order_source = (
                "span_start_time"
            )

        for position, (
            _,
            tool_span,
            tool_name,
        ) in enumerate(
            ordered,
            start=1,
        ):
            observation: Dict[
                str,
                Any,
            ] = {
                "layer": "plan",
                "source": (
                    "openinference_agent_tool_sequence"
                ),
                "span_kind": (
                    "AGENT_TOOL_PLAN"
                ),
                "status": "observed",
                "semantic_correctness": (
                    "unknown"
                ),
                "agent_span_id": (
                    agent_span_id
                ),
                "planned_tool": (
                    tool_name
                ),
                "position": position,
                "order_source": (
                    order_source
                ),
            }

            tool_span_id = (
                _context_value(
                    tool_span,
                    "span_id",
                )
            )

            if tool_span_id:
                observation[
                    "tool_span_id"
                ] = tool_span_id

            trace_id = (
                _context_value(
                    tool_span,
                    "trace_id",
                )
            )

            if trace_id:
                observation[
                    "trace_id"
                ] = trace_id

            observations.append(
                observation
            )

    return observations


def _execution_status(
    span: Dict[str, Any],
) -> str:
    raw = span.get(
        "status_code"
    )

    status = span.get(
        "status"
    )

    if (
        raw is None
        and isinstance(
            status,
            dict,
        )
    ):
        raw = status.get("code")

    normalized = str(
        raw or ""
    ).upper()

    if (
        "ERROR" in normalized
        or "FAIL" in normalized
    ):
        return "failed"

    if (
        "OK" in normalized
        or "SUCCESS" in normalized
    ):
        return "succeeded"

    return "observed"


def _document_ids(
    attributes: Dict[str, Any],
) -> List[str]:
    result = set()

    for key, value in (
        attributes.items()
    ):
        if not isinstance(
            value,
            (str, int),
        ):
            continue

        if key == "document.id":
            result.add(str(value))
            continue

        if re.search(
            r"(?:retrieval|reranker)\.documents\.\d+\.document\.id$",
            key,
        ):
            result.add(str(value))

    return sorted(result)


def _safe_payload_metadata(
    attributes: Dict[str, Any],
) -> Dict[str, Any]:
    result = {}

    for prefix in (
        "input",
        "output",
    ):
        value = attributes.get(
            prefix + ".value"
        )

        if isinstance(value, str):
            result[
                prefix + "_length"
            ] = len(value)

            if value:
                result[
                    prefix + "_sha256"
                ] = _sha256_text(
                    value
                )

    return result


def _span_observation(
    span: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    attributes = _attributes(
        span
    )

    raw_kind = attributes.get(
        "openinference.span.kind"
    )

    if not isinstance(
        raw_kind,
        str,
    ):
        return None

    kind = raw_kind.upper()

    layer = (
        OPENINFERENCE_KIND_LAYER_MAP.get(
            kind
        )
    )

    if layer is None:
        return None

    observation: Dict[
        str,
        Any,
    ] = {
        "layer": layer,
        "source": "openinference_span",
        "span_kind": kind,
        "status": (
            _execution_status(
                span
            )
        ),
        "semantic_correctness": (
            "unknown"
        ),
    }

    trace_id = _context_value(
        span,
        "trace_id",
    )

    span_id = _context_value(
        span,
        "span_id",
    )

    parent_id = _parent_span_id(
        span
    )

    if trace_id:
        observation[
            "trace_id"
        ] = trace_id

    if span_id:
        observation[
            "span_id"
        ] = span_id

    if parent_id:
        observation[
            "parent_span_id"
        ] = parent_id

    name = span.get("name")

    if isinstance(
        name,
        str,
    ):
        observation[
            "span_name_length"
        ] = len(name)

        if name:
            observation[
                "span_name_sha256"
            ] = _sha256_text(
                name
            )

    observation.update(
        _safe_payload_metadata(
            attributes
        )
    )

    if kind == "TOOL":
        tool_name = attributes.get(
            "tool.name"
        )

        if isinstance(
            tool_name,
            str,
        ) and tool_name:
            observation[
                "tool_name"
            ] = tool_name

    if kind in {
        "RETRIEVER",
        "RERANKER",
    }:
        source_ids = (
            _document_ids(
                attributes
            )
        )

        if source_ids:
            observation[
                "source_ids"
            ] = source_ids

        observation[
            "retrieval_stage"
        ] = (
            "rerank"
            if kind == "RERANKER"
            else "retrieve"
        )

    if kind == "LLM":
        model = attributes.get(
            "llm.model_name"
        )

        if isinstance(
            model,
            str,
        ) and model:
            observation[
                "model_name"
            ] = model

    return observation


def project_openinference_trace(
    payload: Any,
) -> Dict[str, Any]:
    spans = extract_openinference_spans(
        payload
    )

    observations = []

    unmapped = set()

    trace_ids = set()

    for span in spans:
        attributes = _attributes(
            span
        )

        kind = attributes.get(
            "openinference.span.kind"
        )

        if isinstance(
            kind,
            str,
        ):
            normalized = kind.upper()

            if (
                normalized
                not in OPENINFERENCE_KIND_LAYER_MAP
            ):
                unmapped.add(
                    normalized
                )

        observation = (
            _span_observation(
                span
            )
        )

        if observation:
            observations.append(
                observation
            )

            trace_id = observation.get(
                "trace_id"
            )

            if isinstance(
                trace_id,
                str,
            ):
                trace_ids.add(
                    trace_id
                )

    derived_plan_observations = (
        _derived_agent_plan_observations(
            spans
        )
    )

    observations.extend(
        derived_plan_observations
    )

    return {
        "schema_version": (
            COGNITIVE_TRACE_SCHEMA_VERSION
        ),
        "source_system": (
            "openinference"
        ),
        "trace_id": (
            next(iter(trace_ids))
            if len(trace_ids) == 1
            else None
        ),
        "observations": observations,
        "unmapped_span_kinds": sorted(
            unmapped
        ),
        "privacy": {
            "capture_mode": (
                "safe_metadata_only"
            ),
            "raw_input_retained": False,
            "raw_output_retained": False,
        },
        "adapter": {
            "name": (
                "openinference_otlp"
            ),
            "input_span_count": len(
                spans
            ),
            "mapped_span_count": (
                len(observations)
                - len(
                    derived_plan_observations
                )
            ),
            "derived_plan_observation_count": (
                len(
                    derived_plan_observations
                )
            ),
        },
        "limitations": [
            (
                "Only OpenInference span kinds with "
                "a conservative Cognitive mapping "
                "are projected."
            ),
            (
                "CHAIN, PROMPT, EMBEDDING, GUARDRAIL, "
                "and EVALUATOR are preserved as "
                "unmapped kinds rather than guessed."
            ),
            (
                "Raw input.value and output.value "
                "are represented by hashes and "
                "lengths only."
            ),
            (
                "Agent plan evidence is derived only "
                "from observable TOOL descendants of "
                "an AGENT span. Multi-tool ordering "
                "requires distinct captured start "
                "timestamps; ambiguous ordering "
                "remains unevaluated."
            ),
        ],
    }


def load_openinference_trace(
    path: str,
) -> Dict[str, Any]:
    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    return project_openinference_trace(
        payload
    )
