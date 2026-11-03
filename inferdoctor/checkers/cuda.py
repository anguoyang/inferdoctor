from __future__ import annotations

import os
import re
import shutil
import subprocess

from inferdoctor.core.checker import Checker
from inferdoctor.core.commands import run_command
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status


CUDA_VERSION_RE = re.compile(r"release\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)


class CudaChecker(Checker):
    name = "cuda"

    def run(self, config: Config) -> CheckResult:
        del config
        executable = shutil.which("nvcc")
        environment = {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDA_PATH": os.environ.get("CUDA_PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        }

        if executable is None:
            configured = [key for key, value in environment.items() if value]
            details = ["nvcc is not available on PATH."]
            if configured:
                details.append(
                    "CUDA-related environment variables set: {0}".format(
                        ", ".join(configured)
                    )
                )
            return CheckResult(
                name=self.name,
                status=Status.SKIP,
                summary="CUDA compiler was not found",
                details=details,
                suggestions=[
                    "No action is needed for CPU-only inference or many prebuilt runtimes such as Ollama.",
                    "Install CUDA toolkit only if you need nvcc for compilation or a runtime that requires it.",
                ],
                raw_data={"nvcc_path": None, "environment": environment},
                translation_key="cuda.not_found",
            )

        try:
            completed = run_command([executable, "--version"])
        except subprocess.TimeoutExpired:
            return CheckResult(
                name=self.name,
                status=Status.FAIL,
                summary="nvcc timed out",
                suggestions=["Check the CUDA toolkit installation."],
                raw_data={"nvcc_path": executable, "environment": environment},
                translation_key="cuda.timed_out",
            )
        except OSError as exc:
            return CheckResult(
                name=self.name,
                status=Status.FAIL,
                summary="nvcc could not be executed",
                details=[str(exc)],
                suggestions=["Repair the CUDA toolkit installation."],
                raw_data={"nvcc_path": executable, "environment": environment},
                translation_key="cuda.exec_error",
            )

        output = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        ).strip()
        if completed.returncode != 0:
            return CheckResult(
                name=self.name,
                status=Status.FAIL,
                summary="nvcc reported an error",
                details=[output or "nvcc exited with a non-zero status."],
                suggestions=["Check the CUDA toolkit installation."],
                raw_data={
                    "nvcc_path": executable,
                    "environment": environment,
                    "returncode": completed.returncode,
                },
                translation_key="cuda.reported_error",
            )

        match = CUDA_VERSION_RE.search(output)
        version = match.group(1) if match else None
        status = Status.PASS if version else Status.WARN
        summary = (
            "CUDA toolkit {0} detected".format(version)
            if version
            else "nvcc is available but its CUDA version was not parsed"
        )
        suggestions = (
            []
            if version
            else ["Run 'nvcc --version' and verify the toolkit output."]
        )
        if version:
            translation_key = "cuda.version_detected"
            translation_args = {"version": version}
        else:
            translation_key = "cuda.version_not_parsed"
            translation_args = {}
        return CheckResult(
            name=self.name,
            status=status,
            summary=summary,
            details=["nvcc: {0}".format(executable)],
            suggestions=suggestions,
            raw_data={
                "nvcc_path": executable,
                "cuda_version": version,
                "environment": environment,
            },
            translation_key=translation_key,
            translation_args=translation_args,
        )
