from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

_SCALAR_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _is_union(origin: Any) -> bool:
    return origin is Union or origin is types.UnionType


def _type_to_json_schema(py_type: Any) -> dict:
    origin = get_origin(py_type)

    if _is_union(origin):
        non_none = [a for a in get_args(py_type) if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])
        raise TypeError(f"unsupported Union type: {py_type}")

    if origin is list:
        (inner,) = get_args(py_type)
        return {"type": "array", "items": _type_to_json_schema(inner)}

    if py_type in _SCALAR_MAP:
        return {"type": _SCALAR_MAP[py_type]}

    raise TypeError(f"unsupported parameter type: {py_type!r}")


def _is_optional(py_type: Any) -> bool:
    origin = get_origin(py_type)
    return _is_union(origin) and type(None) in get_args(py_type)


@dataclass
class _ToolEntry:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if not func.__doc__:
            raise ValueError(f"tool {func.__name__!r} requires a docstring")

        description = func.__doc__.strip().split("\n\n", 1)[0].strip()
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            if param_name not in hints:
                raise TypeError(f"parameter {param_name!r} of {func.__name__!r} lacks a type hint")
            py_type = hints[param_name]
            properties[param_name] = _type_to_json_schema(py_type)
            if param.default is inspect.Parameter.empty and not _is_optional(py_type):
                required.append(param_name)

        self._tools[func.__name__] = _ToolEntry(
            name=func.__name__,
            description=description,
            func=func,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )
        return func

    tool = register

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters,
                },
            }
            for entry in self._tools.values()
        ]

    def dispatch(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            return {"error": f"no such tool: {name}"}
        try:
            return self._tools[name].func(**args)
        except Exception as exc:
            return {"error": str(exc)}

    def names(self) -> list[str]:
        return list(self._tools.keys())
