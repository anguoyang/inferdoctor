from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from inferdoctor.core.cognitive import (
    COGNITIVE_TRACE_SCHEMA_VERSION,
    analyze_cognitive_trace,
)
from inferdoctor.core.dify import (
    CHAT_MODES,
    SUPPORTED_LIVE_MODES,
    WORKFLOW_MODES,
    DifyAPIClient,
    DifyConfig,
    DifyError,
    ensure_endpoint_allowed,
    generated_user_id,
    sanitize_endpoint,
    utc_now,
)


DIFY_NODE_LAYER_MAP = {
    "question-classifier": "intent",
    "if-else": "route",
    "agent": "plan",
    "agent-v2": "plan",
    "tool": "action",
    "http-request": "action",
    "code": "action",
    "knowledge-retrieval": "retrieval",
    "llm": "generation",
    "answer": "postprocessing",
}


def _safe_data(
    event: Dict[str, Any],
) -> Dict[str, Any]:
    value = event.get("data")

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


def _node_observation(
    event: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if event.get("event") not in {
        "node_finished",
        "node_retry",
    }:
        return None

    data = _safe_data(event)

    node_type = data.get(
        "node_type"
    )

    if not isinstance(
        node_type,
        str,
    ):
        return None

    layer = DIFY_NODE_LAYER_MAP.get(
        node_type
    )

    if layer is None:
        return None

    observation: Dict[str, Any] = {
        "layer": layer,
        "source": "dify_node",
        "node_type": node_type,
        "status": (
            data.get("status")
            or (
                "retry"
                if event.get("event")
                == "node_retry"
                else "observed"
            )
        ),
        "semantic_correctness": (
            "unknown"
        ),
    }

    for key in (
        "node_id",
        "predecessor_node_id",
        "iteration_id",
        "loop_id",
    ):
        value = data.get(key)

        if isinstance(
            value,
            str,
        ) and value:
            observation[key] = value

    for key in (
        "index",
        "elapsed_time",
        "retry_index",
    ):
        value = data.get(key)

        if isinstance(
            value,
            (int, float),
        ) and not isinstance(
            value,
            bool,
        ):
            observation[key] = value

    if data.get(
        "error_present"
    ) is True:
        observation[
            "error_present"
        ] = True

    return observation


def project_dify_cognitive_trace(
    events: List[
        Dict[str, Any]
    ],
    *,
    app_mode: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    observations: List[
        Dict[str, Any]
    ] = []

    unmapped_node_types = set()

    for event in events:
        if not isinstance(
            event,
            dict,
        ):
            continue

        if event.get("event") in {
            "node_finished",
            "node_retry",
        }:
            data = _safe_data(event)

            node_type = data.get(
                "node_type"
            )

            if isinstance(
                node_type,
                str,
            ):
                if (
                    node_type
                    not in DIFY_NODE_LAYER_MAP
                ):
                    unmapped_node_types.add(
                        node_type
                    )

            observation = (
                _node_observation(
                    event
                )
            )

            if observation:
                observations.append(
                    observation
                )

        if (
            event.get("event")
            == "agent_thought"
        ):
            observations.append({
                "layer": "plan",
                "source": (
                    "dify_agent_thought"
                ),
                "node_type": (
                    "agent_thought"
                ),
                "status": "observed",
                "semantic_correctness": (
                    "unknown"
                ),
            })

            tool = event.get("tool")

            if (
                isinstance(tool, str)
                and tool
            ):
                observations.append({
                    "layer": "action",
                    "source": (
                        "dify_agent_thought"
                    ),
                    "node_type": (
                        "agent_tool_selection"
                    ),
                    "status": "observed",
                    "tool_name": tool,
                    "semantic_correctness": (
                        "unknown"
                    ),
                })

        if (
            event.get("event")
            == "message_end"
        ):
            resources = event.get(
                "retriever_resources"
            )

            if isinstance(
                resources,
                list,
            ):
                observations.append({
                    "layer": "retrieval",
                    "source": (
                        "dify_retriever_resources"
                    ),
                    "node_type": (
                        "retriever_resources"
                    ),
                    "status": "succeeded",
                    "resource_count": len(
                        resources
                    ),
                    "semantic_correctness": (
                        "unknown"
                    ),
                })

    return {
        "schema_version": (
            COGNITIVE_TRACE_SCHEMA_VERSION
        ),
        "timestamp": utc_now(),
        "trace_id": trace_id,
        "source_system": "dify",
        "app_mode": app_mode,
        "observations": observations,
        "unmapped_node_types": sorted(
            unmapped_node_types
        ),
        "privacy": {
            "capture_mode": (
                "safe_metadata_only"
            ),
            "raw_inputs_retained": False,
            "raw_outputs_retained": False,
            "raw_reasoning_retained": False,
        },
        "limitations": [
            (
                "Dify node types are projected into broad cognitive layers."
            ),
            (
                "A successful node execution is not evidence that its semantic decision was correct."
            ),
            (
                "Unmapped node types are preserved rather than guessed."
            ),
        ],
    }


def capture_dify_cognitive_trace(
    config: DifyConfig,
    *,
    query: str,
    client_factory: Callable[
        ...,
        DifyAPIClient,
    ] = DifyAPIClient,
) -> Dict[str, Any]:
    if (
        not isinstance(
            query,
            str,
        )
        or not query.strip()
    ):
        raise DifyError(
            "Dify cognitive capture requires a non-empty query"
        )

    safety = ensure_endpoint_allowed(
        config.app_base_url,
        allow_non_local=(
            config.allow_non_local
        ),
        allow_public=(
            config.allow_public
        ),
    )

    if not safety["allowed"]:
        raise DifyError(
            "Dify app endpoint is not allowed: {0}".format(
                safety["reason"]
            )
        )

    if not config.app_api_key:
        raise DifyError(
            "{0} is not set".format(
                config.app_key_env
            )
        )

    client = client_factory(
        config.app_base_url,
        config.app_api_key,
        timeout=config.timeout,
    )

    info = client.get_info()

    app_mode = str(
        info.get("mode")
        or ""
    ).strip()

    if app_mode not in (
        SUPPORTED_LIVE_MODES
    ):
        raise DifyError(
            "Dify app mode is not supported for live cognitive capture: {0}".format(
                app_mode
                or "unknown"
            )
        )

    user = generated_user_id()

    if app_mode in CHAT_MODES:
        execution = (
            client.run_chat_stream(
                query,
                user=user,
                show_answer=False,
                capture_trace_events=True,
            )
        )

    elif app_mode in WORKFLOW_MODES:
        execution = (
            client.run_workflow_stream(
                query,
                user=user,
                show_answer=False,
                capture_trace_events=True,
            )
        )

    else:
        raise DifyError(
            "Unsupported Dify app mode: {0}".format(
                app_mode
            )
        )

    trace_events = execution.get(
        "trace_events"
    )

    if not isinstance(
        trace_events,
        list,
    ):
        raise DifyError(
            "Dify stream did not return captured trace events"
        )

    trace = project_dify_cognitive_trace(
        trace_events,
        app_mode=app_mode,
    )

    trace["adapter"] = {
        "name": (
            "dify_stream_events"
        ),
        "endpoint": sanitize_endpoint(
            config.app_base_url
        ),
        "event_capture": (
            execution.get(
                "trace_event_capture"
            )
        ),
        "completion_status": (
            execution.get(
                "completion_status"
            )
        ),
        "total_latency_seconds": (
            execution.get(
                "total_latency_seconds"
            )
        ),
    }

    analysis = analyze_cognitive_trace(
        trace
    )

    return {
        "schema_version": (
            "inferdoctor.cognitive.capture.v1"
        ),
        "trace": trace,
        "analysis": analysis,
    }
