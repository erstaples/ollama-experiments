import os
import pytest

from harness.tools._sandbox import resolve_in_workspace, SandboxError


def test_relative_path_resolves(tmp_path):
    resolved = resolve_in_workspace("subdir/file.txt", tmp_path)
    assert resolved == tmp_path / "subdir" / "file.txt"


def test_dot_resolves_to_workspace(tmp_path):
    assert resolve_in_workspace(".", tmp_path) == tmp_path


def test_dot_dot_escape_rejected(tmp_path):
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve_in_workspace("../etc/passwd", tmp_path)


def test_absolute_outside_rejected(tmp_path):
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve_in_workspace("/etc/passwd", tmp_path)


def test_absolute_inside_workspace_allowed(tmp_path):
    inside = tmp_path / "ok.txt"
    resolved = resolve_in_workspace(str(inside), tmp_path)
    assert resolved == inside


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "link"
    os.symlink(outside, link)
    with pytest.raises(SandboxError, match="escapes workspace"):
        resolve_in_workspace("link/secret.txt", tmp_path)
