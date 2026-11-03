from __future__ import annotations

import shutil
import subprocess

from inferdoctor.core.checker import Checker
from inferdoctor.core.commands import run_command
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status


class DockerChecker(Checker):
    name = "docker"

    def run(self, config: Config) -> CheckResult:
        del config
        executable = shutil.which("docker")
        if executable is None:
            return CheckResult(
                name=self.name,
                status=Status.SKIP,
                summary="Docker CLI was not found",
                details=["docker is not available on PATH."],
                suggestions=[
                    "No action is needed if you do not run local AI services in containers.",
                    "Install Docker only if your local stack depends on containers.",
                ],
                raw_data={"docker_path": None, "daemon_reachable": False},
                translation_key="docker.not_found",
            )

        try:
            completed = run_command([executable, "info", "--format", "{{.ServerVersion}}"])
        except subprocess.TimeoutExpired:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="Docker daemon check timed out",
                suggestions=["Run docker info directly and check whether the daemon is responsive."],
                raw_data={"docker_path": executable, "daemon_reachable": False},
                translation_key="docker.timed_out",
            )
        except OSError as exc:
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="Docker CLI could not be executed",
                details=[str(exc)],
                suggestions=["Check the Docker CLI installation."],
                raw_data={"docker_path": executable, "daemon_reachable": False},
                translation_key="docker.exec_error",
            )

        version = completed.stdout.strip()
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            return CheckResult(
                name=self.name,
                status=Status.WARN,
                summary="Docker CLI is installed but the daemon is not reachable",
                details=[detail or "docker info exited with a non-zero status."],
                suggestions=[
                    "Start Docker Desktop or the Docker daemon if your local AI stack uses containers.",
                    "No containers are started by InferDoctor.",
                ],
                raw_data={
                    "docker_path": executable,
                    "daemon_reachable": False,
                    "returncode": completed.returncode,
                },
                translation_key="docker.daemon_not_reachable",
            )

        return CheckResult(
            name=self.name,
            status=Status.PASS,
            summary="Docker CLI and daemon are reachable",
            details=["Docker daemon version: {0}".format(version or "unknown")],
            raw_data={
                "docker_path": executable,
                "daemon_reachable": True,
                "server_version": version or None,
            },
            translation_key="docker.reachable",
        )
