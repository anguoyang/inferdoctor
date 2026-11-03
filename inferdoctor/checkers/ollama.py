from __future__ import annotations

import shutil

from inferdoctor.core.checker import Checker
from inferdoctor.core.config import Config
from inferdoctor.core.http import (
    HTTPCheckError,
    describe_http_error,
    get_url,
    join_url,
    response_raw_data,
)
from inferdoctor.core.models import CheckResult, Status


class OllamaChecker(Checker):
    name = "ollama"

    def run(self, config: Config) -> CheckResult:
        executable = shutil.which("ollama")
        url = join_url(config.endpoints["ollama"], "api/tags")

        try:
            response = get_url(url, timeout=config.timeout)
        except HTTPCheckError as exc:
            status = Status.WARN if executable else Status.SKIP
            summary = (
                "Ollama is installed but its API is not reachable"
                if executable
                else "Ollama was not found and its API is not reachable"
            )
            translation_key = (
                "ollama.installed_not_reachable"
                if executable
                else "ollama.not_found_not_reachable"
            )
            suggestions = [
                "Start Ollama or update endpoints.ollama.",
                "No action is needed if Ollama is not used on this machine.",
            ]
            return CheckResult(
                name=self.name,
                status=status,
                summary=summary,
                details=["{0}: {1}".format(url, describe_http_error(exc))],
                suggestions=suggestions,
                raw_data={
                    "ollama_path": executable,
                    "url": url,
                    "reachable": False,
                },
                translation_key=translation_key,
            )

        raw_data = response_raw_data(response)
        raw_data.update(
            {
                "ollama_path": executable,
                "reachable": True,
            }
        )
        if response.status != 200:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="Ollama API responded with HTTP {0}".format(response.status),
                details=["Probe URL: {0}".format(url)],
                suggestions=[
                    "Verify the Ollama endpoint and inspect the Ollama service logs."
                ],
                raw_data=raw_data,
                translation_key="ollama.http_error",
                translation_args={"status": response.status},
            )

        models = None
        if isinstance(response.json_data, dict):
            model_data = response.json_data.get("models")
            if isinstance(model_data, list):
                models = len(model_data)
                raw_data["model_count"] = models

        details = ["{0} returned HTTP 200.".format(url)]
        if models is not None:
            details.append("{0} model(s) available.".format(models))
        if executable is None:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="Ollama API is reachable but the CLI is not on PATH",
                details=details,
                suggestions=[
                    "Add the Ollama CLI to PATH if local command access is expected."
                ],
                raw_data=raw_data,
                translation_key="ollama.api_reachable_cli_missing",
            )

        return CheckResult(
            name=self.name,
            status=Status.PASS,
            summary="Ollama CLI and API are available",
            details=details,
            raw_data=raw_data,
            translation_key="ollama.ready",
        )
