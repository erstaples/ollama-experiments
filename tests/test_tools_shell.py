import pytest

from harness.tools.shell import make_shell_tool


def test_run_shell_basic(tmp_path):
    (tmp_path / "marker").write_text("hi")
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("ls")
    assert result["exit_code"] == 0
    assert "marker" in result["stdout"]
    assert result["timed_out"] is False


def test_run_shell_cwd_is_workspace(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("pwd")
    assert result["stdout"].strip() == str(tmp_path.resolve())


def test_run_shell_captures_stderr(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("echo oops >&2; exit 3")
    assert result["exit_code"] == 3
    assert "oops" in result["stderr"]


def test_run_shell_timeout(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("sleep 5", timeout_sec=1)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_run_shell_env_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET", "must-not-leak")
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("echo ${SECRET:-empty}")
    assert "must-not-leak" not in result["stdout"]
    assert "empty" in result["stdout"]


def test_run_shell_home_is_workspace(tmp_path):
    run_shell = make_shell_tool(tmp_path, default_timeout_sec=10)
    result = run_shell("echo $HOME")
    assert result["stdout"].strip() == str(tmp_path.resolve())
