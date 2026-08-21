import json
from dataclasses import replace

import pytest

from inferdoctor.cli import main
from inferdoctor.core.openai_compatible import (
    OpenAICompatibleResponse,
    OpenAICompatibleTransportError,
)
from inferdoctor.core.provider_compare import (
    ProviderCompareError,
    render_provider_compare,
    run_provider_compare,
)
from inferdoctor.core.providers import (
    ProviderError,
    custom_openai_compatible_target,
    get_provider_preset,
    provider_target_from_preset,
)
from inferdoctor.core.rag import RAG_CASE_SCHEMA_VERSION


def _case():
    return {
        "schema_version": RAG_CASE_SCHEMA_VERSION,
        "case_id": "provider-compare-fixture",
        "question": "State the fictional return and defect policy codes.",
        "language": "en",
        "category": "deterministic-provider-comparison",
        "why_bad": "A missing policy code or invented no-refund claim is objectively wrong.",
        "current_answer": "",
        "expected_answer": "RETURN-14 and DEFECT-30 apply.",
        "expected_sources": [],
        "required_facts": [
            {
                "fact_id": "return-code",
                "description": "Return policy code",
                "match_terms": ["RETURN-14"],
                "match_mode": "exact_phrase",
            },
            {
                "fact_id": "defect-code",
                "description": "Defect policy code",
                "match_terms": ["DEFECT-30"],
                "match_mode": "exact_phrase",
            },
        ],
        "forbidden_claims": [
            {
                "claim_id": "no-refunds",
                "description": "Invented no-refund claim",
                "match_terms": ["no refunds"],
                "match_mode": "any_term",
            }
        ],
        "expected_behavior": "Return only the supported fictional policy codes.",
        "metadata": {"fixture": True},
    }


def _response(status=200, data=None, elapsed_ms=20, valid=True):
    return OpenAICompatibleResponse(
        url="https://fixture.invalid/v1",
        status=status,
        json_data=data,
        json_valid=valid,
        elapsed_ms=elapsed_ms,
        body_bytes=40,
    )


def _targets(provider=None, custom_endpoint="http://127.0.0.1:8000/v1"):
    preset = provider or get_provider_preset("orcarouter")
    return [
        provider_target_from_preset(preset, model="orcarouter/free"),
        custom_openai_compatible_target(
            target_id="local",
            display_name="Local vLLM",
            base_url=custom_endpoint,
            model="local-model",
        ),
    ]


def _install_success_mocks(
    monkeypatch,
    *,
    orca_answer="RETURN-14 and DEFECT-30 apply. ORCA-RAW-CONTENT",
    local_answer="RETURN-14 and DEFECT-30 apply. LOCAL-RAW-CONTENT",
    models_status=200,
):
    calls = {"models": [], "chat": []}

    def models(endpoint, *, timeout, api_key=None):
        calls["models"].append((endpoint, timeout, api_key))
        model = "orcarouter/free" if "orcarouter" in endpoint else "local-model"
        data = {"data": [{"id": model}]} if models_status == 200 else {"error": "unsupported"}
        return _response(status=models_status, data=data)

    def chat(endpoint, *, payload, timeout, api_key=None):
        calls["chat"].append(
            {
                "endpoint": endpoint,
                "payload": payload,
                "timeout": timeout,
                "api_key": api_key,
            }
        )
        answer = orca_answer if "orcarouter" in endpoint else local_answer
        latency = 1400 if "orcarouter" in endpoint else 900
        return _response(
            data={"choices": [{"message": {"content": answer}}]},
            elapsed_ms=latency,
        )

    monkeypatch.setattr("inferdoctor.core.provider_compare.list_models", models)
    monkeypatch.setattr("inferdoctor.core.provider_compare.create_chat_completion", chat)
    return calls


def _run(monkeypatch, **mock_options):
    calls = _install_success_mocks(monkeypatch, **mock_options)
    result = run_provider_compare(
        _targets(),
        _case(),
        api_keys={"orcarouter": "provider-secret", "local": None},
        allow_public=True,
    )
    return result, calls


def test_orcarouter_vs_no_key_local_happy_path_uses_identical_workload(monkeypatch):
    result, calls = _run(monkeypatch)

    assert result["status"] == "PASS"
    assert result["target_order"] == ["orcarouter", "local"]
    assert len(calls["chat"]) == 2
    first_payload = dict(calls["chat"][0]["payload"])
    second_payload = dict(calls["chat"][1]["payload"])
    assert first_payload.pop("model") == "orcarouter/free"
    assert second_payload.pop("model") == "local-model"
    assert first_payload == second_payload
    assert calls["chat"][0]["api_key"] == "provider-secret"
    assert calls["chat"][1]["api_key"] is None

    orca, local = result["targets"]
    assert orca["workload_sha256"] == local["workload_sha256"]
    assert orca["checks"]["model_access"]["status"] == "PASS"
    assert local["checks"]["authentication"]["status"] == "SKIP"
    assert local["checks"]["model_access"]["status"] == "PASS"
    assert local["quality"]["status"] == "PASS"
    assert local["first_broken_layer"] is None
    assert orca["metrics"]["observed_total_latency_ms"] == 1400
    assert local["metrics"]["observed_total_latency_ms"] == 900
    assert local["metrics"]["ttft_ms"] is None
    assert result["recommendations"] == []
    assert local["cost"]["status"] == "UNKNOWN"

    serialized = json.dumps(result)
    assert "ORCA-RAW-CONTENT" not in serialized
    assert "LOCAL-RAW-CONTENT" not in serialized
    assert "provider-secret" not in serialized
    assert all(
        not item["invocation"]["response_content_retained"]
        for item in result["targets"]
    )


def test_public_endpoint_requires_allow_public_before_any_request(monkeypatch):
    calls = _install_success_mocks(monkeypatch)

    with pytest.raises(ProviderError, match="--allow-public"):
        run_provider_compare(
            _targets(),
            _case(),
            api_keys={"orcarouter": "secret"},
        )

    assert calls["models"] == []
    assert calls["chat"] == []


def test_lan_endpoint_requires_allow_non_local_before_any_request(monkeypatch):
    calls = _install_success_mocks(monkeypatch)

    with pytest.raises(ProviderError, match="--allow-non-local"):
        run_provider_compare(
            _targets(custom_endpoint="http://192.168.1.20:8000/v1"),
            _case(),
            api_keys={"orcarouter": "secret"},
            allow_public=True,
        )

    assert calls["models"] == []
    assert calls["chat"] == []


def test_url_credentials_are_rejected_without_exposing_them(monkeypatch):
    calls = _install_success_mocks(monkeypatch)
    targets = _targets(
        custom_endpoint="http://user:top-secret@127.0.0.1:8000/v1"
    )

    with pytest.raises(ProviderError, match="URL credentials") as exc_info:
        run_provider_compare(
            targets,
            _case(),
            api_keys={"orcarouter": "secret"},
            allow_public=True,
        )

    assert "top-secret" not in str(exc_info.value)
    assert calls["models"] == []
    assert calls["chat"] == []


@pytest.mark.parametrize(
    ("status", "layer", "check_name"),
    (
        (401, "authentication", "authentication"),
        (403, "model_access", "model_access"),
        (402, "billing_or_credit", "billing_or_credit"),
        (429, "quota_or_rate_limit", "quota_or_rate_limit"),
    ),
)
def test_invocation_http_failures_have_conservative_first_layer(
    monkeypatch,
    status,
    layer,
    check_name,
):
    _install_success_mocks(monkeypatch)

    def chat(endpoint, *, payload, timeout, api_key=None):
        if "orcarouter" in endpoint:
            return _response(status=status, data={"error": "fixture"}, elapsed_ms=31)
        return _response(
            data={
                "choices": [
                    {"message": {"content": "RETURN-14 and DEFECT-30 apply."}}
                ]
            },
            elapsed_ms=22,
        )

    monkeypatch.setattr("inferdoctor.core.provider_compare.create_chat_completion", chat)
    result = run_provider_compare(
        _targets(),
        _case(),
        api_keys={"orcarouter": "secret"},
        allow_public=True,
    )

    failed, successful = result["targets"]
    assert result["status"] == "FAIL"
    assert failed["status"] == "FAIL"
    assert failed["checks"][check_name]["status"] == "FAIL"
    assert failed["first_broken_layer"] == layer
    assert failed["quality"]["status"] == "UNKNOWN"
    assert successful["status"] == "PASS"
    assert successful["quality"]["status"] == "PASS"
    if status == 403:
        assert failed["checks"]["authentication"]["status"] == "PASS"


def test_unsupported_models_is_unknown_but_successful_invocation_proves_access(monkeypatch):
    result, _calls = _run(monkeypatch, models_status=404)

    for target in result["targets"]:
        assert target["models_probe"]["status"] == "UNKNOWN"
        assert target["checks"]["model_catalog"]["status"] == "UNKNOWN"
        assert target["checks"]["model_access"]["status"] == "PASS"
        assert target["invocation"]["status"] == "PASS"
        assert target["status"] == "PASS"


def test_catalog_presence_does_not_prove_access(monkeypatch):
    _install_success_mocks(monkeypatch)

    def chat(endpoint, *, payload, timeout, api_key=None):
        if "orcarouter" in endpoint:
            return _response(status=403, data={"error": "denied"})
        return _response(
            data={
                "choices": [
                    {"message": {"content": "RETURN-14 and DEFECT-30 apply."}}
                ]
            }
        )

    monkeypatch.setattr("inferdoctor.core.provider_compare.create_chat_completion", chat)
    result = run_provider_compare(
        _targets(),
        _case(),
        api_keys={"orcarouter": "secret"},
        allow_public=True,
    )
    failed = result["targets"][0]

    assert failed["models_probe"]["selected_model_listed"] is True
    assert failed["checks"]["model_catalog"]["status"] == "PASS"
    assert failed["checks"]["model_access"]["status"] == "FAIL"


def test_unlisted_catalog_model_remains_unknown_when_invocation_succeeds(
    monkeypatch,
):
    _install_success_mocks(monkeypatch)

    def models(endpoint, *, timeout, api_key=None):
        return _response(data={"data": [{"id": "different-model"}]})

    monkeypatch.setattr("inferdoctor.core.provider_compare.list_models", models)
    result = run_provider_compare(
        _targets(),
        _case(),
        api_keys={"orcarouter": "secret"},
        allow_public=True,
    )

    for target in result["targets"]:
        assert target["models_probe"]["status"] == "PASS"
        assert target["models_probe"]["selected_model_listed"] is False
        assert target["checks"]["model_catalog"]["status"] == "UNKNOWN"
        assert target["checks"]["model_access"]["status"] == "PASS"
        assert target["status"] == "PASS"


def test_required_fact_failure_is_verification_first_broken_layer(monkeypatch):
    result, _calls = _run(
        monkeypatch,
        local_answer="RETURN-14 applies, but the other code is omitted.",
    )
    local = result["targets"][1]

    assert local["checks"]["generation"]["status"] == "PASS"
    assert local["quality"]["required_facts_matched"] == 1
    assert local["quality"]["required_facts_total"] == 2
    assert local["quality"]["status"] == "FAIL"
    assert local["first_broken_layer"] == "verification"
    missing = local["quality"]["required_fact_checks"]["results"][1]
    assert missing["missing_terms"] == ["DEFECT-30"]


def test_forbidden_claim_detection_is_derived_without_raw_answer(monkeypatch):
    answer = "RETURN-14 and DEFECT-30 apply, but there are no refunds. RAW-FORBIDDEN"
    result, _calls = _run(monkeypatch, local_answer=answer)
    local = result["targets"][1]

    assert local["quality"]["forbidden_claims_matched"] == 1
    assert local["quality"]["forbidden_claim_checks"]["hits"] == ["no-refunds"]
    assert local["quality"]["status"] == "FAIL"
    assert local["first_broken_layer"] == "verification"
    assert "RAW-FORBIDDEN" not in json.dumps(result)


def test_transport_failure_for_one_target_preserves_other_target_evidence(monkeypatch):
    _install_success_mocks(monkeypatch)

    def models(endpoint, *, timeout, api_key=None):
        if "orcarouter" in endpoint:
            raise OpenAICompatibleTransportError("fixture connection refused")
        return _response(data={"data": [{"id": "local-model"}]})

    def chat(endpoint, *, payload, timeout, api_key=None):
        if "orcarouter" in endpoint:
            raise OpenAICompatibleTransportError("fixture connection refused")
        return _response(
            data={
                "choices": [
                    {"message": {"content": "RETURN-14 and DEFECT-30 apply."}}
                ]
            },
            elapsed_ms=18,
        )

    monkeypatch.setattr("inferdoctor.core.provider_compare.list_models", models)
    monkeypatch.setattr("inferdoctor.core.provider_compare.create_chat_completion", chat)
    result = run_provider_compare(
        _targets(),
        _case(),
        api_keys={"orcarouter": "secret"},
        allow_public=True,
    )

    failed, successful = result["targets"]
    assert failed["checks"]["connectivity"]["status"] == "FAIL"
    assert failed["checks"]["authentication"]["status"] == "UNKNOWN"
    assert failed["first_broken_layer"] == "connectivity"
    assert failed["request_sent"] is None
    assert failed["models_probe"]["request_sent"] is None
    assert failed["invocation"]["request_sent"] is None
    assert successful["status"] == "PASS"
    assert successful["quality"]["required_facts_matched"] == 2
    assert successful["metrics"]["observed_total_latency_ms"] == 18


@pytest.mark.parametrize(
    ("bad_key", "error_message", "secret_fragment", "safe_error_fragment"),
    (
        (
            "prefix\nvery-secret-value",
            "API key contains invalid whitespace or control characters",
            "very-secret-value",
            "invalid whitespace",
        ),
        (
            {"token": "typed-secret-value"},
            "API key must be a string",
            "typed-secret-value",
            "not a string",
        ),
    ),
    ids=("whitespace", "non-string"),
)
def test_malformed_api_key_is_authentication_failure_without_secret_retention(
    monkeypatch,
    bad_key,
    error_message,
    secret_fragment,
    safe_error_fragment,
):
    _install_success_mocks(monkeypatch)

    def models(endpoint, *, timeout, api_key=None):
        if "orcarouter" in endpoint:
            raise OpenAICompatibleTransportError(error_message)
        return _response(data={"data": [{"id": "local-model"}]})

    def chat(endpoint, *, payload, timeout, api_key=None):
        if "orcarouter" in endpoint:
            raise OpenAICompatibleTransportError(error_message)
        return _response(
            data={
                "choices": [
                    {"message": {"content": "RETURN-14 and DEFECT-30 apply."}}
                ]
            }
        )

    monkeypatch.setattr("inferdoctor.core.provider_compare.list_models", models)
    monkeypatch.setattr("inferdoctor.core.provider_compare.create_chat_completion", chat)
    result = run_provider_compare(
        _targets(),
        _case(),
        api_keys={"orcarouter": bad_key},
        allow_public=True,
    )
    failed = result["targets"][0]

    assert failed["request_attempted"] is True
    assert failed["request_sent"] is None
    assert failed["checks"]["connectivity"]["status"] == "UNKNOWN"
    assert failed["checks"]["authentication"]["status"] == "FAIL"
    assert failed["metrics"]["sample_count"] == 0
    assert failed["models_probe"]["status"] == "FAIL"
    assert failed["models_probe"]["request_sent"] is None
    assert failed["models_probe"]["http_status"] is None
    assert failed["checks"]["model_catalog"]["status"] == "UNKNOWN"
    assert failed["invocation"]["status"] == "FAIL"
    assert failed["invocation"]["request_sent"] is None
    assert failed["invocation"]["http_status"] is None
    assert failed["first_broken_layer"] == "authentication"
    serialized = json.dumps(result)
    assert secret_fragment not in serialized
    assert safe_error_fragment in serialized


def _diagnostic_projection(result):
    return {
        "status": result["status"],
        "target_order": result["target_order"],
        "targets": [
            {
                "id": item["target"]["id"],
                "status": item["status"],
                "checks": {
                    key: value["status"] for key, value in item["checks"].items()
                },
                "quality": item["quality"],
                "metrics": item["metrics"],
                "first_broken_layer": item["first_broken_layer"],
            }
            for item in result["targets"]
        ],
        "differences": result["differences"],
        "observations": result["observations"],
        "recommendations": result["recommendations"],
    }


def test_partner_metadata_cannot_change_comparison(monkeypatch):
    calls = _install_success_mocks(monkeypatch)
    provider = get_provider_preset("orcarouter")
    changed_partner = replace(provider, partner_url="https://partner.invalid/other")

    baseline = run_provider_compare(
        _targets(provider),
        _case(),
        api_keys={"orcarouter": "first-secret"},
        allow_public=True,
    )
    changed = run_provider_compare(
        _targets(changed_partner),
        _case(),
        api_keys={"orcarouter": "second-secret"},
        allow_public=True,
    )

    assert _diagnostic_projection(baseline) == _diagnostic_projection(changed)
    assert (
        baseline["targets"][0]["target"]["provider_metadata"]["partner_url"]
        != changed["targets"][0]["target"]["provider_metadata"]["partner_url"]
    )
    serialized_evidence = json.dumps(
        {
            "differences": changed["differences"],
            "observations": changed["observations"],
            "recommendations": changed["recommendations"],
        }
    )
    assert "partner.invalid" not in serialized_evidence
    assert all("partner.invalid" not in endpoint for endpoint, *_ in calls["models"])
    assert all(
        "partner.invalid" not in call["endpoint"] for call in calls["chat"]
    )
    assert "first-secret" not in json.dumps(baseline)
    assert "second-secret" not in json.dumps(changed)


def test_required_custom_api_key_env_fails_only_that_target(monkeypatch):
    calls = _install_success_mocks(monkeypatch)
    targets = [
        provider_target_from_preset(
            get_provider_preset("orcarouter"),
            model="orcarouter/free",
        ),
        custom_openai_compatible_target(
            target_id="secured-local",
            display_name="Secured Local",
            base_url="http://127.0.0.1:8000/v1",
            model="local-model",
            api_key_env="LOCAL_API_KEY",
        ),
    ]
    result = run_provider_compare(
        targets,
        _case(),
        api_keys={"orcarouter": "secret", "secured-local": None},
        allow_public=True,
    )

    secured = result["targets"][1]
    assert secured["request_sent"] is False
    assert secured["first_broken_layer"] == "authentication"
    assert secured["target"]["api_key_env"] == "LOCAL_API_KEY"
    assert len(calls["chat"]) == 1


def test_compare_requires_valid_case_and_unique_targets():
    invalid = dict(_case(), question="")
    with pytest.raises(ProviderCompareError, match="invalid RAG Case"):
        run_provider_compare(_targets(), invalid, allow_public=True)

    duplicate = _targets()
    duplicate[1] = replace(duplicate[1], id="orcarouter")
    with pytest.raises(ProviderCompareError, match="ids must be unique"):
        run_provider_compare(duplicate, _case(), allow_public=True)


def test_provider_compare_cli_json_and_human_output(monkeypatch, tmp_path, capsys):
    _install_success_mocks(monkeypatch)
    monkeypatch.setenv("ORCAROUTER_API_KEY", "cli-secret")
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case()), encoding="utf-8")
    args = [
        "provider",
        "compare",
        "--provider",
        "orcarouter",
        "--provider-model",
        "orcarouter/free",
        "--custom-endpoint",
        "http://127.0.0.1:8000/v1",
        "--custom-model",
        "local-model",
        "--custom-label",
        "Local vLLM",
        "--case",
        str(case_path),
        "--allow-public",
    ]

    assert main(args + ["--format", "json"]) == 0
    json_output = capsys.readouterr().out
    parsed = json.loads(json_output)
    assert parsed["schema_version"] == "inferdoctor.provider.compare.v1"
    assert parsed["target_order"] == ["orcarouter", "custom"]
    assert "cli-secret" not in json_output

    assert main(args) == 0
    console = capsys.readouterr().out
    assert "InferDoctor Provider Compare" in console
    assert "Local vLLM" in console
    assert "Required facts" in console
    assert "First broken layer" in console
    assert "no winner" in console


def test_console_renderer_uses_unknown_not_inferred_ttft(monkeypatch):
    result, _calls = _run(monkeypatch)
    rendered = render_provider_compare(result)

    assert "TTFT" in rendered
    assert "UNKNOWN" in rendered
    assert "not a benchmark" in rendered


def test_console_renderer_preserves_zero_forbidden_claim_matches(monkeypatch):
    result, _calls = _run(monkeypatch)
    assert all(
        target["quality"]["forbidden_claims_matched"] == 0
        for target in result["targets"]
    )

    rendered = render_provider_compare(result)
    forbidden_line = next(
        line for line in rendered.splitlines() if line.startswith("Forbidden claims")
    )
    assert forbidden_line.split() == ["Forbidden", "claims", "0", "0"]
    assert "UNKNOWN" not in forbidden_line
