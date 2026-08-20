from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional

from inferdoctor.core.dify import (
    DifyAPIClient,
    DifyAPIError,
    DifyConfig,
    ensure_endpoint_allowed,
    sanitize_endpoint,
)
from inferdoctor.core.rag import (
    RAG_TRACE_SCHEMA_VERSION,
    RagError,
    sha256_text,
    utc_now,
    validate_trace_object,
)


def _segment(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("segment")
    return value if isinstance(value, dict) else {}


def _string_value(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _stable_chunk_id(
    record: Dict[str, Any],
    index: int,
) -> str:
    segment = _segment(record)

    value = _string_value(
        segment.get("id"),
        record.get("id"),
        segment.get("index_node_id"),
    )

    if value:
        return value

    encoded = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    return "dify-record-{0}-{1}".format(
        index,
        sha256_text(encoded)[:12],
    )


def _source_id(
    record: Dict[str, Any],
) -> Optional[str]:
    segment = _segment(record)

    document = segment.get("document")
    document = (
        document
        if isinstance(document, dict)
        else {}
    )

    return _string_value(
        segment.get("document_id"),
        record.get("document_id"),
        document.get("id"),
    )


def _source_title(
    record: Dict[str, Any],
) -> Optional[str]:
    segment = _segment(record)

    document = segment.get("document")
    document = (
        document
        if isinstance(document, dict)
        else {}
    )

    return _string_value(
        segment.get("document_name"),
        segment.get("document_title"),
        record.get("document_name"),
        record.get("title"),
        document.get("name"),
        document.get("title"),
    )


def _segment_text(
    record: Dict[str, Any],
) -> Optional[str]:
    segment = _segment(record)

    return _string_value(
        segment.get("content"),
        record.get("content"),
        record.get("text"),
    )


def _candidate_from_record(
    record: Dict[str, Any],
    rank: int,
    *,
    include_content: bool,
) -> Dict[str, Any]:
    segment = _segment(record)
    text = _segment_text(record)

    candidate: Dict[str, Any] = {
        "chunk_id": _stable_chunk_id(
            record,
            rank,
        ),
        "rank": rank,
    }

    source_id = _source_id(record)

    if source_id:
        candidate["source_id"] = source_id

    title = _source_title(record)

    if title:
        candidate["title"] = title

    score = record.get("score")

    if isinstance(score, (int, float)):
        candidate["score"] = float(score)

    position = segment.get("position")

    if isinstance(position, int):
        candidate["position"] = position

    if text is not None:
        candidate["text_length"] = len(text)
        candidate["text_sha256"] = (
            sha256_text(text)
        )

        if include_content:
            candidate["text"] = text

    return candidate


def capture_dify_knowledge_trace(
    config: DifyConfig,
    *,
    query: str,
    top_k: int = 3,
    include_content: bool = False,
    case_id: Optional[str] = None,
    client_factory: Callable[..., DifyAPIClient] = DifyAPIClient,
) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise RagError(
            "Dify RAG capture requires a non-empty query"
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise RagError(
            "Dify RAG capture requires --top-k greater than zero"
        )

    base_url = (
        config.knowledge_base_url
        or config.app_base_url
    )

    safety = ensure_endpoint_allowed(
        base_url,
        allow_non_local=config.allow_non_local,
        allow_public=config.allow_public,
    )

    if not safety["allowed"]:
        raise RagError(
            "Dify knowledge endpoint is not allowed: {0}".format(
                safety["reason"]
            )
        )

    if not config.knowledge_api_key:
        raise RagError(
            "{0} is not set".format(
                config.knowledge_key_env
            )
        )

    if not config.dataset_id:
        raise RagError(
            "DIFY_DATASET_ID or --dataset-id is required"
        )

    client = client_factory(
        base_url,
        config.knowledge_api_key,
        timeout=config.timeout,
    )

    started = time.perf_counter()

    try:
        response = client.retrieve_chunks(
            config.dataset_id,
            query,
            top_k=top_k,
        )
    except DifyAPIError as exc:
        raise RagError(
            "Dify knowledge retrieval failed: {0}".format(
                exc
            )
        ) from exc

    retrieval_ms = int(
        round(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )
    )

    records = response.get("records")

    if not isinstance(records, list):
        raise RagError(
            "Dify knowledge retrieval response did not contain a records list"
        )

    candidates = [
        _candidate_from_record(
            record,
            rank,
            include_content=include_content,
        )
        for rank, record in enumerate(
            (
                record
                for record in records
                if isinstance(record, dict)
            ),
            start=1,
        )
    ]

    timestamp = utc_now()

    trace_id = (
        "dify-knowledge-"
        + sha256_text(
            "{0}|{1}|{2}|{3}".format(
                timestamp,
                config.dataset_id,
                query,
                len(candidates),
            )
        )[:16]
    )

    input_data: Dict[str, Any] = {
        "original_question_sha256": (
            sha256_text(query)
        ),
        "language": "unknown",
    }

    if include_content:
        input_data[
            "original_question"
        ] = query

    retrieval: Dict[str, Any] = {
        "query_sha256": sha256_text(query),
        "top_k_requested": top_k,
        "candidates": candidates,
        "latency_ms": retrieval_ms,
        "backend": "dify_knowledge_api",
        "dataset_id": config.dataset_id,
        "status": "ok",
    }

    if include_content:
        retrieval["query"] = query

    trace: Dict[str, Any] = {
        "schema_version": (
            RAG_TRACE_SCHEMA_VERSION
        ),
        "trace_id": trace_id,
        "timestamp": timestamp,
        "system": {
            "name": "dify",
            "source": "knowledge_api",
        },
        "pipeline": {
            "name": (
                "dify-knowledge-retrieval"
            ),
            "version": "unknown",
        },
        "input": input_data,
        "retrieval": retrieval,
        "context_selection": {},
        "generation": {
            "status": "not_captured",
        },
        "postprocessing": {
            "status": "not_captured",
        },
        "timings": {
            "retrieval_ms": retrieval_ms,
            "total_ms": retrieval_ms,
        },
        "privacy": {
            "content_included": (
                include_content
            ),
            "redaction_applied": (
                not include_content
            ),
            "private_data_present": None,
            "export_mode": (
                "include_content"
                if include_content
                else "redacted"
            ),
        },
        "adapter": {
            "name": (
                "dify_knowledge_api"
            ),
            "scope": "retrieval_only",
            "endpoint": sanitize_endpoint(
                base_url
            ),
        },
        "capture_limitations": [
            (
                "This adapter captures the Dify Knowledge API retrieval result only."
            ),
            (
                "It does not prove which chunks the application later selected into final context."
            ),
            (
                "It does not capture prompt, generation, tool use, or post-processing."
            ),
        ],
    }

    if case_id:
        trace["case_id"] = case_id

    findings = validate_trace_object(trace)

    failures = [
        item
        for item in findings
        if item["status"] == "FAIL"
    ]

    if failures:
        raise RagError(
            "internal Dify adapter produced an invalid RAG trace: {0}".format(
                "; ".join(
                    item["message"]
                    for item in failures
                )
            )
        )

    return trace
