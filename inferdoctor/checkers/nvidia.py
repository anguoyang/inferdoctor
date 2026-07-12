from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List

from inferdoctor.core.checker import Checker
from inferdoctor.core.commands import run_command
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status


class NvidiaChecker(Checker):
    name = "nvidia"

    def run(self, config: Config) -> CheckResult:
        del config
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return CheckResult(
                name=self.name,
                status=Status.SKIP,
                summary="nvidia-smi was not found",
                details=["No NVIDIA management CLI is available on PATH."],
                suggestions=[
                    "If this machine has an NVIDIA GPU, install or repair its driver.",
                    "Skip this check on CPU-only or non-NVIDIA systems.",
                ],
                raw_data={"nvidia_smi_path": None},
                translation_key="nvidia.not_found",
            )

        command = [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = run_command(command)
        except subprocess.TimeoutExpired:
            return CheckResult(
                name=self.name,
                status=Status.FAIL,
                summary="nvidia-smi timed out",
                suggestions=["Check whether the NVIDIA driver is responsive."],
                raw_data={"nvidia_smi_path": executable},
                translation_key="nvidia.timed_out",
            )
        except OSError as exc:
            return CheckResult(
                name=self.name,
                status=Status.FAIL,
                summary="nvidia-smi could not be executed",
                details=[str(exc)],
                suggestions=["Repair the NVIDIA driver installation."],
                raw_data={"nvidia_smi_path": executable},
                translation_key="nvidia.exec_error",
            )

        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout).strip()
            return CheckResult(
                name=self.name,
                status=Status.FAIL,
                summary="nvidia-smi reported an error",
                details=[error or "nvidia-smi exited with a non-zero status."],
                suggestions=[
                    "Check that the NVIDIA kernel driver is loaded and matches user-space tools."
                ],
                raw_data={
                    "nvidia_smi_path": executable,
                    "returncode": completed.returncode,
                },
                translation_key="nvidia.reported_error",
            )

        gpus: List[Dict[str, object]] = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                memory_mib = int(fields[1])
            except ValueError:
                memory_mib = None
            gpus.append(
                {
                    "name": fields[0],
                    "memory_total_mib": memory_mib,
                    "driver_version": fields[2],
                }
            )

        if not gpus:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="nvidia-smi ran but no GPU data was parsed",
                details=[completed.stdout.strip() or "The command returned no output."],
                suggestions=["Run nvidia-smi directly and inspect its output."],
                raw_data={"nvidia_smi_path": executable},
                translation_key="nvidia.no_gpu_data",
            )

        details = [
            "{0}: {1} MiB VRAM, driver {2}".format(
                gpu["name"],
                gpu["memory_total_mib"]
                if gpu["memory_total_mib"] is not None
                else "unknown",
                gpu["driver_version"],
            )
            for gpu in gpus
        ]
        return CheckResult(
            name=self.name,
            status=Status.PASS,
            summary="{0} NVIDIA GPU(s) detected".format(len(gpus)),
            details=details,
            raw_data={"nvidia_smi_path": executable, "gpus": gpus},
            translation_key="nvidia.detected",
            translation_args={"count": len(gpus)},
        )
