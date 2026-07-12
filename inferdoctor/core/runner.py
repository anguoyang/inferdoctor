from __future__ import annotations

from typing import Iterable, List

from inferdoctor.core.checker import Checker
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.i18n import t


def run_checks(checkers: Iterable[Checker], config: Config) -> List[CheckResult]:
    results = []
    for checker in checkers:
        try:
            results.append(checker.run(config))
        except Exception as exc:
            results.append(
                CheckResult(
                    name=checker.name,
                    status=Status.FAIL,
                    summary="Checker failed unexpectedly",
                    details=["{0}: {1}".format(type(exc).__name__, exc)],
                    suggestions=[
                        "Run with the latest InferDoctor version and report this error."
                    ],
                    raw_data={"exception_type": type(exc).__name__},
                    translation_key="runner.checker_failed",
                )
            )
    return results
