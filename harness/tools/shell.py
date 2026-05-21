from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable


def make_shell_tool(workspace: Path, default_timeout_sec: int = 30) -> Callable:
    """Return the run_shell tool bound to the given workspace."""
    workspace = workspace.resolve()

    def run_shell(command: str, timeout_sec: int = default_timeout_sec) -> dict:
        """Run a bash command inside the sandboxed workspace.

        The command runs with cwd set to the workspace, a stripped environment,
        and a configurable timeout. Returns exit_code, stdout, stderr, and a
        timed_out flag.
        """
        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": str(workspace),
            "USER": os.environ.get("USER", "harness"),
            "LANG": "C.UTF-8",
        }
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {
                "exit_code": -1,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
            }
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }

    return run_shell
