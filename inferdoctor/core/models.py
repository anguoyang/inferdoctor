from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    name: str
    status: Status
    summary: str
    details: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    # i18n: translation_key identifies the TRANSLATIONS entry for the summary.
    # The renderer calls t(f"checker.{translation_key}", language, **translation_args)
    # to get the localized summary, falling back to the English `summary` field.
    translation_key: Optional[str] = None
    translation_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "details": self.details,
            "suggestions": self.suggestions,
            "raw_data": self.raw_data,
        }
