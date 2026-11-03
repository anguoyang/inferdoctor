from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from inferdoctor.core.checker import Checker
from inferdoctor.core.config import Config
from inferdoctor.core.http import (
    HTTPCheckError,
    describe_http_error,
    get_url,
    response_raw_data,
)
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.core.openai_compatible import models_url as _models_url


def _suggested_v1_base(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return urlunsplit((parts.scheme, parts.netloc, path + "/v1", "", ""))


class OpenAICompatibleChecker(Checker):
    endpoint_name: str
    service_label: str

    def run(self, config: Config) -> CheckResult:
        base_url = config.endpoints[self.endpoint_name]
        models_url = _models_url(base_url)
        raw_base = {
            "base_url": base_url,
            "models_url": models_url,
            "reachable": False,
        }
        svc_label = self.service_label

        try:
            response = get_url(models_url, timeout=config.timeout)
        except HTTPCheckError as exc:
            return CheckResult(
                name=self.name,
                status=Status.SKIP,
                summary="{0} endpoint is not reachable".format(svc_label),
                details=["{0}: {1}".format(models_url, describe_http_error(exc))],
                suggestions=[
                    "Start {0} or verify endpoints.{1}.".format(
                        svc_label, self.endpoint_name
                    ),
                    "Retry with: inferdoctor check {0} --endpoint {1}".format(
                        self.name, base_url
                    ),
                ],
                raw_data=raw_base,
                translation_key="openai.not_reachable",
                translation_args={"service_label": svc_label},
            )

        raw_data = response_raw_data(response)
        raw_data.update(raw_base)
        raw_data["reachable"] = True

        if response.status in (401, 403):
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="{0} is reachable but requires authentication".format(
                    svc_label
                ),
                details=["{0} returned HTTP {1}.".format(models_url, response.status)],
                suggestions=[
                    "The endpoint requires authentication. Set an API key or check the service configuration.",
                    "Verify that /v1/models is allowed by the server or reverse proxy.",
                ],
                raw_data=raw_data,
                translation_key="openai.requires_auth",
                translation_args={"service_label": svc_label},
            )

        if response.status == 404:
            suggested_base = _suggested_v1_base(base_url)
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="{0} models route returned HTTP 404".format(svc_label),
                details=["Probe URL: {0}".format(models_url)],
                suggestions=[
                    "The service responded, but /v1/models was not found.",
                    "Your base URL may be wrong. Try adding /v1 to the endpoint URL.",
                    "Try: inferdoctor check {0} --endpoint {1}".format(
                        self.name, suggested_base
                    ),
                ],
                raw_data=raw_data,
                translation_key="openai.not_found",
                translation_args={"service_label": svc_label},
            )

        if not 200 <= response.status < 300:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="{0} returned HTTP {1}".format(svc_label, response.status),
                details=["Probe URL: {0}".format(models_url)],
                suggestions=[
                    "Inspect the server or proxy logs for this HTTP status.",
                    "Verify the base URL with --endpoint.",
                ],
                raw_data=raw_data,
                translation_key="openai.http_status",
                translation_args={"service_label": svc_label, "status": response.status},
            )

        if response.json_data is None:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="{0} returned invalid JSON".format(svc_label),
                details=["{0} returned HTTP 200 but not JSON.".format(models_url)],
                suggestions=[
                    "Check that this is the OpenAI-compatible API, not a web UI route."
                ],
                raw_data=raw_data,
                translation_key="openai.invalid_json",
                translation_args={"service_label": svc_label},
            )

        if not isinstance(response.json_data, dict) or not isinstance(
            response.json_data.get("data"), list
        ):
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="{0} response is not OpenAI-compatible".format(svc_label),
                details=["Expected a JSON object containing a 'data' list."],
                suggestions=[
                    "Verify the endpoint exposes the OpenAI-compatible /v1/models route."
                ],
                raw_data=raw_data,
                translation_key="openai.not_compatible",
                translation_args={"service_label": svc_label},
            )

        model_count = len(response.json_data["data"])
        raw_data["model_count"] = model_count
        return CheckResult(
            name=self.name,
            status=Status.PASS,
            summary="{0} OpenAI-compatible API is ready".format(svc_label),
            details=[
                "{0} returned HTTP 200 with {1} model(s).".format(
                    models_url, model_count
                )
            ],
            raw_data=raw_data,
            translation_key="openai.ready",
            translation_args={"service_label": svc_label, "count": model_count},
        )
