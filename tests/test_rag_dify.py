import json

from inferdoctor.cli import main
from inferdoctor.core.dify import DifyConfig
from inferdoctor.core.rag import (
    RAG_CASE_SCHEMA_VERSION,
    diagnose_rag,
    validate_trace_object,
)
from inferdoctor.core.rag_dify import (
    capture_dify_knowledge_trace,
)


class FakeDifyKnowledgeClient:
    def __init__(
        self,
        base_url,
        api_key=None,
        timeout=30.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def retrieve_chunks(
        self,
        dataset_id,
        query,
        *,
        top_k=3,
    ):
        assert dataset_id == "dataset-id"
        assert query == "return policy"
        assert top_k == 5

        return {
            "query": query,
            "records": [
                {
                    "score": 0.91,
                    "segment": {
                        "id": "segment-1",
                        "position": 3,
                        "document_id": (
                            "document-policy"
                        ),
                        "content": (
                            "Returns are allowed "
                            "for 30 days."
                        ),
                    },
                },
                {
                    "score": 0.72,
                    "segment": {
                        "id": "segment-2",
                        "position": 1,
                        "document_id": (
                            "document-other"
                        ),
                        "content": (
                            "Other synthetic text."
                        ),
                    },
                },
            ],
        }


class FakeEmptyKnowledgeClient(
    FakeDifyKnowledgeClient
):
    def retrieve_chunks(
        self,
        dataset_id,
        query,
        *,
        top_k=3,
    ):
        return {
            "query": query,
            "records": [],
        }


def config():
    return DifyConfig(
        app_base_url=(
            "http://127.0.0.1:5001/v1"
        ),
        knowledge_base_url=(
            "http://127.0.0.1:5001/v1"
        ),
        knowledge_api_key="secret",
        dataset_id="dataset-id",
    )


def rag_case():
    return {
        "schema_version": (
            RAG_CASE_SCHEMA_VERSION
        ),
        "case_id": "case-1",
        "question": "return policy",
        "language": "en",
        "category": "retrieval",
        "why_bad": "Expected source missing.",
        "expected_sources": [
            {
                "source_id": (
                    "document-policy"
                ),
                "required": True,
            }
        ],
        "required_facts": [],
        "forbidden_claims": [],
    }


def test_dify_capture_is_valid_redacted_trace():
    trace = capture_dify_knowledge_trace(
        config(),
        query="return policy",
        top_k=5,
        client_factory=(
            FakeDifyKnowledgeClient
        ),
    )

    failures = [
        item
        for item in validate_trace_object(
            trace
        )
        if item["status"] == "FAIL"
    ]

    assert failures == []

    assert (
        trace["adapter"]["scope"]
        == "retrieval_only"
    )

    assert (
        "original_question"
        not in trace["input"]
    )

    assert (
        trace["privacy"][
            "content_included"
        ]
        is False
    )

    candidates = trace[
        "retrieval"
    ]["candidates"]

    assert len(candidates) == 2

    assert (
        candidates[0]["source_id"]
        == "document-policy"
    )

    assert candidates[0]["rank"] == 1
    assert candidates[0]["score"] == 0.91

    assert "text" not in candidates[0]

    assert (
        candidates[0]["text_sha256"]
    )

    assert (
        "selected_chunk_ids"
        not in trace["context_selection"]
    )


def test_dify_capture_can_explicitly_include_content():
    trace = capture_dify_knowledge_trace(
        config(),
        query="return policy",
        top_k=5,
        include_content=True,
        client_factory=(
            FakeDifyKnowledgeClient
        ),
    )

    assert (
        trace["input"][
            "original_question"
        ]
        == "return policy"
    )

    assert (
        trace["retrieval"][
            "candidates"
        ][0]["text"]
        == "Returns are allowed for 30 days."
    )

    assert (
        trace["privacy"][
            "content_included"
        ]
        is True
    )


def test_dify_retrieval_only_trace_does_not_invent_context_failure():
    trace = capture_dify_knowledge_trace(
        config(),
        query="return policy",
        top_k=5,
        client_factory=(
            FakeDifyKnowledgeClient
        ),
    )

    result = diagnose_rag(
        rag_case(),
        trace,
    )

    categories = {
        item["category"]
        for item in result["diagnoses"]
    }

    assert (
        "retrieval_failure"
        not in categories
    )

    assert (
        "context_selection_failure"
        not in categories
    )

    assert (
        "insufficient_evidence"
        in categories
    )

    assert (
        result["attribution"][
            "first_broken_layer"
        ]
        is None
    )


def test_dify_empty_retrieval_can_prove_retrieval_failure():
    trace = capture_dify_knowledge_trace(
        config(),
        query="return policy",
        top_k=5,
        client_factory=(
            FakeEmptyKnowledgeClient
        ),
    )

    result = diagnose_rag(
        rag_case(),
        trace,
    )

    assert (
        result["attribution"][
            "first_broken_layer"
        ]
        == "retrieval"
    )


def test_rag_capture_cli_writes_trace(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "trace.json"

    monkeypatch.setenv(
        "DIFY_KNOWLEDGE_API_KEY",
        "secret",
    )

    monkeypatch.setattr(
        "inferdoctor.cli.capture_dify_knowledge_trace",
        lambda config, **kwargs: (
            capture_dify_knowledge_trace(
                config,
                query=kwargs["query"],
                top_k=5,
                client_factory=(
                    FakeDifyKnowledgeClient
                ),
            )
        ),
    )

    exit_code = main([
        "rag",
        "capture",
        "dify-knowledge",
        "--dataset-id",
        "dataset-id",
        "--query",
        "return policy",
        "--top-k",
        "5",
        "--output",
        str(output),
    ])

    assert exit_code == 0
    assert output.exists()

    trace = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert (
        trace["schema_version"]
        == "inferdoctor.rag.trace.v1"
    )
