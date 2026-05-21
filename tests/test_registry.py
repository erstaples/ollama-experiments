import pytest

from harness.registry import ToolRegistry


def test_register_simple_function():
    r = ToolRegistry()

    @r.tool
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"hello, {name}"

    schemas = r.schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "greet"
    assert schema["function"]["description"] == "Say hello to someone."
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["name"] == {"type": "string"}
    assert params["required"] == ["name"]


def test_default_arg_is_not_required():
    r = ToolRegistry()

    @r.tool
    def list_dir(path: str = ".") -> list[str]:
        """List a directory."""
        return []

    params = r.schemas()[0]["function"]["parameters"]
    assert params["properties"]["path"] == {"type": "string"}
    assert params["required"] == []


def test_supported_scalar_types():
    r = ToolRegistry()

    @r.tool
    def mixed(s: str, i: int, f: float, b: bool) -> str:
        """Mixed scalars."""
        return ""

    props = r.schemas()[0]["function"]["parameters"]["properties"]
    assert props["s"] == {"type": "string"}
    assert props["i"] == {"type": "integer"}
    assert props["f"] == {"type": "number"}
    assert props["b"] == {"type": "boolean"}


def test_list_type():
    r = ToolRegistry()

    @r.tool
    def join(parts: list[str]) -> str:
        """Join parts."""
        return ""

    props = r.schemas()[0]["function"]["parameters"]["properties"]
    assert props["parts"] == {"type": "array", "items": {"type": "string"}}


def test_optional_type():
    r = ToolRegistry()

    @r.tool
    def maybe(name: str | None = None) -> str:
        """Maybe name."""
        return ""

    params = r.schemas()[0]["function"]["parameters"]
    assert params["properties"]["name"] == {"type": "string"}
    assert params["required"] == []


def test_unsupported_type_raises():
    r = ToolRegistry()
    with pytest.raises(TypeError, match="unsupported"):
        @r.tool
        def bad(thing: dict) -> str:
            """No."""
            return ""


def test_missing_docstring_raises():
    r = ToolRegistry()
    with pytest.raises(ValueError, match="docstring"):
        @r.tool
        def undocumented(x: str) -> str:
            return x


def test_dispatch_calls_function():
    r = ToolRegistry()

    @r.tool
    def upper(text: str) -> str:
        """Uppercase."""
        return text.upper()

    result = r.dispatch("upper", {"text": "hi"})
    assert result == "HI"


def test_dispatch_unknown_tool_returns_error_dict():
    r = ToolRegistry()
    result = r.dispatch("nope", {})
    assert result == {"error": "no such tool: nope"}


def test_dispatch_catches_function_exception():
    r = ToolRegistry()

    @r.tool
    def boom(x: int) -> int:
        """Explodes."""
        raise RuntimeError("boom")

    result = r.dispatch("boom", {"x": 1})
    assert result == {"error": "boom"}


def test_register_function_directly_without_decorator():
    r = ToolRegistry()

    def hello(name: str) -> str:
        """Hello."""
        return f"hi {name}"

    r.register(hello)
    assert r.dispatch("hello", {"name": "x"}) == "hi x"
