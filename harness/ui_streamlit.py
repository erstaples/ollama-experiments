from __future__ import annotations

import json

import streamlit as st

from harness.agent import (
    Agent,
    AgentError,
    TextChunk,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
)
from harness.config import Config
from harness.ollama_client import OllamaClient
from harness.tools import build_registry


st.set_page_config(page_title="ollama-experiments", page_icon="🦙", layout="wide")


def _build_agent(config: Config) -> Agent:
    config.workspace.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(
        host=config.ollama_host,
        model=config.model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
    )
    registry = build_registry(config.workspace, shell_timeout_sec=config.shell_timeout_sec)
    return Agent(client=client, registry=registry, max_iterations=config.max_tool_iterations)


def _init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = Config.from_env()
    if "agent" not in st.session_state:
        st.session_state.agent = _build_agent(st.session_state.config)
    if "visible_messages" not in st.session_state:
        st.session_state.visible_messages = []


def _render_history() -> None:
    for msg in st.session_state.visible_messages:
        with st.chat_message(msg["role"]):
            if msg.get("content"):
                st.markdown(msg["content"])
            for tc in msg.get("tool_calls", []):
                with st.expander(f"🔧 {tc['name']}", expanded=False):
                    st.json(tc["args"])
                    st.markdown("**Result:**")
                    st.json(tc["result"])


def _run_turn(user_input: str) -> None:
    agent: Agent = st.session_state.agent
    st.session_state.visible_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        buffer = ""
        tool_call_records: dict[str, dict] = {}
        ordered_call_ids: list[str] = []
        expanders: dict = {}

        try:
            for event in agent.run_turn(user_input):
                if isinstance(event, TextChunk):
                    buffer += event.text
                    text_placeholder.markdown(buffer)
                elif isinstance(event, ToolCallStart):
                    ordered_call_ids.append(event.call_id)
                    tool_call_records[event.call_id] = {"name": event.name, "args": event.args, "result": None}
                    exp = st.expander(f"🔧 {event.name}", expanded=False)
                    expanders[event.call_id] = exp
                    with exp:
                        st.json(event.args)
                        st.markdown("**Result:** _pending..._")
                elif isinstance(event, ToolCallResult):
                    tool_call_records[event.call_id]["result"] = event.result
                    exp = expanders[event.call_id]
                    with exp:
                        st.markdown("**Result:**")
                        if event.error:
                            st.error(event.error)
                        else:
                            st.json(event.result)
                elif isinstance(event, AgentError):
                    st.error(event.message)
        except Exception as exc:
            st.error(f"agent error: {exc}")
            return

    st.session_state.visible_messages.append({
        "role": "assistant",
        "content": buffer,
        "tool_calls": [tool_call_records[cid] for cid in ordered_call_ids],
    })


def main() -> None:
    _init_state()
    config: Config = st.session_state.config

    with st.sidebar:
        st.title("🦙 Harness")
        st.caption("Local LLM playground")
        st.markdown(f"**Model:** `{config.model}`")
        st.markdown(f"**Host:** `{config.ollama_host}`")
        st.markdown(f"**Workspace:** `{config.workspace}`")
        st.markdown(f"**Temperature:** `{config.temperature}`")
        st.markdown(f"**num_ctx:** `{config.num_ctx}`")
        st.markdown(f"**Max iterations:** `{config.max_tool_iterations}`")
        st.divider()
        if st.button("Reset conversation", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.visible_messages = []
            st.rerun()

    st.title("Chat")
    _render_history()

    user_input = st.chat_input("Ask the model anything (it can list files, run shell, search web)")
    if user_input:
        _run_turn(user_input)


main()
