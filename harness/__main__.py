from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from harness.config import Config


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="harness", description="Local LLM tool-calling harness")
    parser.add_argument("--ui", choices=["repl", "streamlit"], default="repl")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--ctx", type=int, default=None, help="num_ctx")
    parser.add_argument("--workspace", type=Path, default=None, help="Sandbox root for filesystem and shell tools")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.ui == "streamlit":
        env = os.environ.copy()
        if args.model:
            env["OLLAMA_MODEL"] = args.model
        if args.temp is not None:
            env["TEMPERATURE"] = str(args.temp)
        if args.ctx is not None:
            env["NUM_CTX"] = str(args.ctx)
        if args.workspace is not None:
            env["WORKSPACE_ROOT"] = str(args.workspace)

        page = Path(__file__).parent / "ui_streamlit.py"
        os.execvpe("streamlit", ["streamlit", "run", str(page)], env)

    config = Config.from_env(
        model=args.model,
        temperature=args.temp,
        num_ctx=args.ctx,
        workspace=args.workspace,
    )

    from harness.ui_repl import run
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
