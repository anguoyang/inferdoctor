import io
import json
import urllib.error

import pytest

from inferdoctor.core import openai_compatible
from inferdoctor.core.openai_compatible import (
    OpenAICompatibleTransportError,
    chat_completions_url,
    create_chat_completion,
    extract_chat_text,
    list_models,
    models_url,
)


class _Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def getcode(self):
        return self.status

    def read(self, _limit):
        return self.data


def test_openai_compatible_urls_share_v1_normalization():
    assert models_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1/models"
    assert models_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/models"
    assert chat_completions_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1/chat/completions"


def test_models_request_uses_bearer_key_without_returning_it(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return _Response(b'{"data": [{"id": "fixture"}]}')

    monkeypatch.setattr(openai_compatible.urllib.request, "urlopen", fake_urlopen)

    response = list_models(
        "https://provider.example/v1",
        timeout=4.0,
        api_key="test-secret-key",
    )

    assert captured == {
        "url": "https://provider.example/v1/models",
        "authorization": "Bearer test-secret-key",
        "timeout": 4.0,
    }
    assert response.status == 200
    assert response.json_data["data"][0]["id"] == "fixture"
    assert "test-secret-key" not in repr(response)


def test_chat_completion_uses_shared_transport_and_extracts_content(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response(b'{"choices": [{"message": {"content": "OK"}}]}')

    monkeypatch.setattr(openai_compatible.urllib.request, "urlopen", fake_urlopen)
    response = create_chat_completion(
        "https://provider.example/v1",
        payload={"model": "fixture", "messages": []},
        timeout=3.0,
        api_key="secret",
    )

    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["payload"]["model"] == "fixture"
    assert extract_chat_text(response.json_data) == "OK"


def test_http_error_is_returned_as_bounded_status_evidence(monkeypatch):
    error = urllib.error.HTTPError(
        "https://provider.example/v1/models",
        401,
        "unauthorized",
        None,
        io.BytesIO(b'{"error": "unauthorized"}'),
    )
    monkeypatch.setattr(
        openai_compatible.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    response = list_models(
        "https://provider.example/v1",
        timeout=2.0,
        api_key="secret",
    )

    assert response.status == 401
    assert response.json_valid is True


def test_network_error_uses_transport_error_without_key(monkeypatch):
    monkeypatch.setattr(
        openai_compatible.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("connection refused")
        ),
    )

    with pytest.raises(OpenAICompatibleTransportError, match="connection refused") as exc:
        list_models(
            "https://provider.example/v1",
            timeout=2.0,
            api_key="do-not-leak",
        )

    assert "do-not-leak" not in str(exc.value)
