import pytest

from harness.tools.filesystem import make_filesystem_tools


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("bravo")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("charlie")
    return tmp_path


def test_list_directory_root(ws):
    tools = make_filesystem_tools(ws)
    result = tools["list_directory"](".")
    assert sorted(result) == ["a.txt", "b.txt", "sub"]


def test_list_directory_subdir(ws):
    tools = make_filesystem_tools(ws)
    assert tools["list_directory"]("sub") == ["c.txt"]


def test_list_directory_escape_returns_error(ws):
    tools = make_filesystem_tools(ws)
    result = tools["list_directory"]("../")
    assert isinstance(result, dict) and "error" in result


def test_read_file(ws):
    tools = make_filesystem_tools(ws)
    assert tools["read_file"]("a.txt") == "alpha"


def test_read_file_truncates(ws):
    (ws / "big.txt").write_text("x" * 200)
    tools = make_filesystem_tools(ws)
    result = tools["read_file"]("big.txt", max_bytes=50)
    assert len(result) <= 100
    assert "truncated" in result.lower()


def test_read_file_missing(ws):
    tools = make_filesystem_tools(ws)
    result = tools["read_file"]("nope.txt")
    assert isinstance(result, dict) and "error" in result


def test_write_file_creates(ws):
    tools = make_filesystem_tools(ws)
    result = tools["write_file"]("new.txt", "hello")
    assert result["bytes_written"] == 5
    assert (ws / "new.txt").read_text() == "hello"


def test_write_file_creates_parents(ws):
    tools = make_filesystem_tools(ws)
    tools["write_file"]("deep/nested/file.txt", "data")
    assert (ws / "deep" / "nested" / "file.txt").read_text() == "data"


def test_write_file_append(ws):
    tools = make_filesystem_tools(ws)
    tools["write_file"]("a.txt", " extra", append=True)
    assert (ws / "a.txt").read_text() == "alpha extra"


def test_write_file_escape_returns_error(ws):
    tools = make_filesystem_tools(ws)
    result = tools["write_file"]("../oops.txt", "x")
    assert isinstance(result, dict) and "error" in result
    assert not (ws.parent / "oops.txt").exists()
