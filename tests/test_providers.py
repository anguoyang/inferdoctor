import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from inferdoctor.cli import main
from inferdoctor.core.openai_compatible import (
    OpenAICompatibleResponse,
    OpenAICompatibleTransportError,
)
from inferdoctor.core.providers import (
    ProviderError,
    get_provider_preset,
    list_provider_presets,
    render_provider_check,
    run_provider_check,
)


def _response(status=200, data=None, valid=True, elapsed_ms=12):
    return OpenAICompatibleResponse(
        url="https://api.orcarouter.ai/v1/models",
        status=status,
        json_data=data,
        json_valid=valid,
        elapsed_ms=elapsed_ms,
        body_bytes=20,
    )


def test_orcarouter_preset_is_provider_metadata_only():
    provider = get_provider_preset("orcarouter")

    assert provider.protocol == "openai-compatible"
    assert provider.base_url == "https://api.orcarouter.ai/v1"
    assert provider.api_key_env == "ORCAROUTER_API_KEY"
    assert provider.default_model == "orcarouter/auto"
    assert provider.partner_url is None
    assert [item.id for item in list_provider_presets()] == ["orcarouter"]


def test_provider_check_requires_explicit_public_allow():
    with pytest.raises(ProviderError, match="--allow-public"):
        run_provider_check(
            get_provider_preset("orcarouter"),
            api_key="secret",
        )


def test_missing_key_sends_no_request_and_fails_configuration(monkeypatch):
    probe = patch("inferdoctor.core.providers.list_models")
    with probe as models_probe:
        result = run_provider_check(
            get_provider_preset("orcarouter"),
            api_key=None,
            allow_public=True,
        )

    assert result["status"] == "FAIL"
    assert result["request_sent"] is False
    assert result["models_probe"]["status"] == "UNKNOWN"
    models_probe.assert_not_called()


def test_models_check_passes_without_chat_and_keeps_unknown_metrics(monkeypatch):
    models_probe = monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(
            data={"data": [{"id": "orcarouter/auto"}, {"id": "fixture"}]}
        ),
    )
    chat = patch("inferdoctor.core.providers.create_chat_completion")
    with chat as chat_probe:
        result = run_provider_check(
            get_provider_preset("orcarouter"),
            api_key="secret",
            allow_public=True,
        )

    assert models_probe is None
    assert result["status"] == "UNKNOWN"
    assert result["models_probe"]["model_count"] == 2
    assert result["models_probe"]["selected_model_listed"] is True

    catalog = [
        item
        for item in result["checks"]
        if item["name"] == "model_catalog"
    ][-1]

    availability = [
        item
        for item in result["checks"]
        if item["name"] == "model_availability"
    ][-1]

    assert catalog["status"] == "PASS"
    assert availability["status"] == "UNKNOWN"
    assert result["chat_smoke"]["status"] == "SKIP"
    assert result["metrics"]["ttft_ms"] is None
    assert result["metrics"]["total_latency_ms"] is None
    assert result["pricing"]["status"] == "UNKNOWN"
    assert result["cost"]["total_compute_cost"] is None
    chat_probe.assert_not_called()


def test_models_route_unsupported_is_unknown_not_fail(monkeypatch):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(status=404, data={"error": "not found"}),
    )

    result = run_provider_check(
        get_provider_preset("orcarouter"),
        api_key="secret",
        allow_public=True,
    )

    assert result["status"] == "UNKNOWN"
    assert result["models_probe"]["status"] == "UNKNOWN"
    model_check = [item for item in result["checks"] if item["name"] == "model_availability"][-1]
    assert model_check["status"] == "UNKNOWN"


def test_invalid_models_shape_keeps_authentication_unknown(monkeypatch):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(data={"models": []}),
    )

    result = run_provider_check(
        get_provider_preset("orcarouter"),
        api_key="secret",
        allow_public=True,
    )

    assert result["status"] == "UNKNOWN"
    auth = [item for item in result["checks"] if item["name"] == "authentication"][-1]
    assert auth["status"] == "UNKNOWN"


def test_authentication_rejection_is_fail(monkeypatch):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(status=401, data={"error": "bad key"}),
    )

    result = run_provider_check(
        get_provider_preset("orcarouter"),
        api_key="secret",
        allow_public=True,
    )

    assert result["status"] == "FAIL"
    auth = [item for item in result["checks"] if item["name"] == "authentication"][-1]
    assert auth["status"] == "FAIL"


def test_transport_failure_keeps_model_availability_unknown(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OpenAICompatibleTransportError("connection refused")

    monkeypatch.setattr("inferdoctor.core.providers.list_models", fail)
    result = run_provider_check(
        get_provider_preset("orcarouter"),
        api_key="secret",
        allow_public=True,
    )

    assert result["status"] == "FAIL"
    assert result["models_probe"]["model_count"] is None
    assert "connection refused" in json.dumps(result)


def test_optional_chat_smoke_can_establish_invocation_without_models_route(monkeypatch):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(status=404, data={"error": "not found"}),
    )
    monkeypatch.setattr(
        "inferdoctor.core.providers.create_chat_completion",
        lambda *_args, **_kwargs: _response(
            data={"choices": [{"message": {"content": "OK"}}]},
            elapsed_ms=34,
        ),
    )

    result = run_provider_check(
        get_provider_preset("orcarouter"),
        api_key="secret",
        allow_public=True,
        smoke=True,
    )

    assert result["status"] == "PASS"
    assert result["models_probe"]["status"] == "UNKNOWN"
    assert result["chat_smoke"]["status"] == "PASS"
    assert result["chat_smoke"]["selected_model_invoked"] is True
    assert result["metrics"]["total_latency_ms"] == 34
    auth = [item for item in result["checks"] if item["name"] == "authentication"][-1]
    availability = [item for item in result["checks"] if item["name"] == "model_availability"][-1]
    assert auth["status"] == "PASS"
    assert availability["status"] == "PASS"
    assert result["metrics"]["ttft_ms"] is None
    assert result["chat_smoke"]["response_content_retained"] is False


def test_api_key_and_partner_metadata_do_not_influence_diagnosis(monkeypatch):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(
            data={"data": [{"id": "orcarouter/auto"}]}
        ),
    )
    provider = get_provider_preset("orcarouter")
    partner_provider = replace(provider, partner_url="https://partner.invalid/ref")

    baseline = run_provider_check(
        provider,
        api_key="first-secret",
        allow_public=True,
    )
    partner = run_provider_check(
        partner_provider,
        api_key="second-secret",
        allow_public=True,
    )

    assert baseline["status"] == partner["status"] == "UNKNOWN"
    assert [item["status"] for item in baseline["checks"]] == [
        item["status"] for item in partner["checks"]
    ]
    serialized = json.dumps([baseline, partner])
    assert "first-secret" not in serialized
    assert "second-secret" not in serialized


def test_provider_cli_list_show_and_check(monkeypatch, capsys):
    assert main(["provider", "list"]) == 0
    assert "orcarouter" in capsys.readouterr().out

    assert main(["provider", "show", "orcarouter"]) == 0
    show_output = capsys.readouterr().out
    assert "orcarouter/auto" in show_output
    assert "partner_url: none" in show_output

    monkeypatch.setenv("ORCAROUTER_API_KEY", "cli-secret")
    with patch(
        "inferdoctor.cli.run_provider_check",
        return_value={
            "status": "UNKNOWN",
            "provider": {"id": "orcarouter", "display_name": "OrcaRouter"},
            "endpoint": "https://api.orcarouter.ai/v1",
            "model": "orcarouter/auto",
            "api_key_env": "ORCAROUTER_API_KEY",
            "api_key_present": True,
            "checks": [],
            "metrics": {"ttft_ms": None, "total_latency_ms": None},
            "pricing": {"status": "UNKNOWN"},
        },
    ) as check:
        assert main(
            [
                "provider",
                "check",
                "--provider",
                "orcarouter",
                "--allow-public",
                "--format",
                "json",
            ]
        ) == 0

    output = capsys.readouterr().out
    assert '"status": "UNKNOWN"' in output
    assert "cli-secret" not in output
    check.assert_called_once_with(
        get_provider_preset("orcarouter"),
        api_key="cli-secret",
        timeout=10.0,
        allow_non_local=False,
        allow_public=True,
        smoke=False,
        model=None,
    )


def test_provider_check_console_renders_unknown_explicitly():
    result = {
        "status": "UNKNOWN",
        "provider": {"id": "fixture", "display_name": "Fixture"},
        "endpoint": "https://provider.example/v1",
        "model": "fixture",
        "api_key_env": "FIXTURE_KEY",
        "api_key_present": True,
        "checks": [],
        "metrics": {"ttft_ms": None, "total_latency_ms": None},
        "pricing": {"status": "UNKNOWN"},
    }

    rendered = render_provider_check(result)

    assert "Status: UNKNOWN" in rendered
    assert "TTFT: UNKNOWN" in rendered
    assert "Pricing: UNKNOWN" in rendered



def test_provider_check_malformed_key_fails_safely_without_secret():
    bad_key = "prefix\nvery-secret-value"

    result = run_provider_check(
        get_provider_preset("orcarouter"),
        api_key=bad_key,
        allow_public=True,
    )

    serialized = json.dumps(result)

    assert result["status"] == "FAIL"
    assert result["request_sent"] is True
    assert "very-secret-value" not in serialized
    assert bad_key not in serialized
    assert "invalid whitespace" in serialized



def test_chat_403_is_model_access_failure_not_authentication_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(
            data={
                "data": [
                    {
                        "id": "orcarouter/free"
                    }
                ]
            }
        ),
    )

    monkeypatch.setattr(
        "inferdoctor.core.providers.create_chat_completion",
        lambda *_args, **_kwargs: _response(
            status=403,
            data={
                "error": {
                    "type": "orcarouter_api_error",
                    "message": (
                        "This API key does not have access "
                        "to model orcarouter/free."
                    ),
                }
            },
            elapsed_ms=25,
        ),
    )

    result = run_provider_check(
        get_provider_preset(
            "orcarouter"
        ),
        api_key="secret",
        allow_public=True,
        smoke=True,
        model="orcarouter/free",
    )

    assert (
        result["status"]
        == "FAIL"
    )

    assert (
        result["models_probe"][
            "selected_model_listed"
        ]
        is True
    )

    auth_checks = [
        item
        for item
        in result["checks"]
        if item["name"]
        == "authentication"
    ]

    assert any(
        item["status"] == "PASS"
        for item in auth_checks
    )

    assert not any(
        item["status"] == "FAIL"
        for item in auth_checks
    )

    access = [
        item
        for item
        in result["checks"]
        if item["name"]
        == "model_access"
    ][-1]

    assert (
        access["status"]
        == "FAIL"
    )

    assert (
        result["chat_smoke"][
            "http_status"
        ]
        == 403
    )


def test_chat_401_remains_authentication_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        "inferdoctor.core.providers.list_models",
        lambda *_args, **_kwargs: _response(
            status=404,
            data={
                "error": "unsupported"
            },
        ),
    )

    monkeypatch.setattr(
        "inferdoctor.core.providers.create_chat_completion",
        lambda *_args, **_kwargs: _response(
            status=401,
            data={
                "error": {
                    "message": "bad key"
                }
            },
        ),
    )

    result = run_provider_check(
        get_provider_preset(
            "orcarouter"
        ),
        api_key="secret",
        allow_public=True,
        smoke=True,
        model="orcarouter/free",
    )

    auth = [
        item
        for item
        in result["checks"]
        if (
            item["name"]
            == "authentication"
            and item["status"]
            == "FAIL"
        )
    ]

    assert auth
