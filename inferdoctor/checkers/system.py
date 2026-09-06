from __future__ import annotations

import os
import platform
import sys
from typing import Optional

from inferdoctor.core.checker import Checker
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status


def _available_memory_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        except (OSError, ValueError, TypeError):
            pass
    return None


def _format_bytes(value: int) -> str:
    gib = value / (1024 ** 3)
    return "{0:.1f} GiB".format(gib)


class SystemChecker(Checker):
    name = "system"

    def run(self, config: Config) -> CheckResult:
        del config
        memory = _available_memory_bytes()
        os_name = platform.platform()
        architecture = platform.machine() or "unknown"
        python_version = platform.python_version()

        details = [
            "OS: {0}".format(os_name),
            "Python: {0}".format(python_version),
            "Architecture: {0}".format(architecture),
        ]
        if memory is not None:
            details.append("Available memory: {0}".format(_format_bytes(memory)))
        else:
            details.append("Available memory: unavailable")

        return CheckResult(
            name=self.name,
            status=Status.PASS,
            summary="System information collected",
            details=details,
            raw_data={
                "os": os_name,
                "python_version": python_version,
                "python_executable": sys.executable,
                "architecture": architecture,
                "available_memory_bytes": memory,
            },
            translation_key="system.ok",
        )
