"""Streamlit playground for the ReAct agent.

Features:
 - Enter a prompt
 - Select which tools the agent may use (multi-select)
 - Stream and display ALL intermediate reasoning / tool messages
 - Show the final AI answer separately

Run:
  uv run streamlit run src/streamlit/agent_app.py
"""

from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

from loguru import logger
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import BaseTool

from llm.document_search_tools import (  # noqa: E402  (import after path tweak)
    search_headwaters_company_info,
    search_local_text_documents,
    get_local_document_content,
    search_headwaters_local_docs,
)


def get_available_tools() -> Dict[str, BaseTool]:
    """Return mapping of selectable tool names to tool instances."""
    return {
        "Headwaters Company Info": search_headwaters_company_info,
        "Local Text Document Search": search_local_text_documents,
        "Get Local Document Content": get_local_document_content,
        "Headwaters Local Docs": search_headwaters_local_docs,
    }


def build_agent(selected: List[BaseTool]):
    """Instantiate a ReAct agent for the chosen tools.

    Parameters
    ----------
    selected: list of tool objects

    Returns
    -------
    Runnable agent executor.
    """
    model = init_chat_model("gpt-4", model_provider="openai")
    return create_react_agent(model, selected)


def render_message(step_index: int, message: BaseMessage) -> None:
    """Render a single streamed message block in the UI.

    Differentiates message types for readability.
    """
    role: str = getattr(message, "type", "message")
    with st.container():
        st.markdown(f"**Step {step_index} - {role}**")
        # 1) Show any tool call metadata (function calling / tool invocation)
        #    Different integrations (OpenAI, Anthropic, etc.) may expose tool_calls
        #    either as an attribute or inside additional_kwargs. We handle both.
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:  # Newer LC API
            tool_calls = message.tool_calls  # type: ignore[assignment]
        else:  # Fallback to raw additional kwargs
            additional = getattr(message, "additional_kwargs", {}) or {}
            if isinstance(additional, dict) and additional.get("tool_calls"):
                tool_calls = additional.get("tool_calls")  # type: ignore[assignment]

        if tool_calls:
            st.caption("Tool invocation(s):")
            for idx, tc in enumerate(tool_calls, start=1):
                # tc may be a dict or an object; use getattr fallback
                name = (
                    tc.get("name")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", "<unknown>")
                )
                # Args could live under 'args', 'arguments' (stringified JSON), or inside 'input'
                raw_args = None
                if isinstance(tc, dict):
                    raw_args = (
                        tc.get("args")
                        or tc.get("arguments")
                        or tc.get("input")
                        or tc.get("parameters")
                    )
                else:  # object style
                    raw_args = (
                        getattr(tc, "args", None)
                        or getattr(tc, "arguments", None)
                        or getattr(tc, "input", None)
                        or getattr(tc, "parameters", None)
                    )
                st.code(f"[{idx}] {name} args={raw_args}")

        # 2) Tool result messages (after a tool runs) usually have type == 'tool' and
        #    often a 'name' attribute. Highlight them for clarity.
        if role == "tool":
            tool_name = getattr(message, "name", None)
            if tool_name:
                st.caption(f"Tool result from: {tool_name}")

        # 3) Display main message content (which could be str or list parts)
        content = message.content
        if isinstance(content, list):  # Some tool outputs may be structured
            for idx, part in enumerate(content):
                st.write(f"Part {idx + 1}:")
                st.write(part)
        else:
            st.write(content)


def main() -> None:  # noqa: D401 - short description obvious from module docstring
    st.set_page_config(page_title="Agent Playground", page_icon="🧠", layout="wide")
    st.title("🧠 ReAct Agent Playground")
    st.caption(
        "Test the LLM ReAct agent with selectable tools. All intermediate messages are shown."
    )

    tools_map = get_available_tools()
    tool_names: List[str] = list(tools_map.keys())

    with st.sidebar:
        st.header("Configuration")
        chosen_tool_names: List[str] = st.multiselect(
            "Allowed Tools (multi-select)",
            options=tool_names,
            default=tool_names,  # Enable all by default
            help="Select which tools the agent can call during reasoning.",
        )
        st.slider(
            "(Info) Temperature (placeholder)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Placeholder for future temperature control (not applied currently).",
        )
        thread_id: str = st.text_input(
            "Thread ID", value=str(uuid4()), help="Used to isolate conversation state."
        )

    prompt: str = st.text_area(
        "Prompt",
        value="株式会社ヘッドウォーターズの作ったアプリについて教えてください。",
        height=160,
    )
    run_button = st.button("Run Agent", type="primary")

    if run_button:
        if not prompt.strip():
            st.warning("Enter a prompt first.")
            return

        selected_tools: List[BaseTool] = [
            tools_map[name] for name in chosen_tool_names if name in tools_map
        ]
        logger.info(
            f"Launching agent with {len(selected_tools)} tools: {chosen_tool_names} thread_id={thread_id}"
        )

        try:
            agent_executor = build_agent(selected_tools)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Failed to create agent: {exc}")
            logger.exception("Agent build failed")
            return

        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=prompt)

        st.subheader("Streamed Messages")
        message_area = st.container()
        final_answer: str | None = None

        with st.spinner("Running agent..."):
            for idx, step in enumerate(
                agent_executor.stream(
                    {"messages": [input_message]}, config, stream_mode="values"
                )
            ):
                messages: List[BaseMessage] = step.get("messages", [])  # type: ignore[assignment]
                if not messages:
                    continue
                last_message: BaseMessage = messages[-1]
                with message_area:
                    render_message(idx + 1, last_message)
                # Heuristic: treat 'ai' typed messages as potential final answers
                if getattr(last_message, "type", "") == "ai":
                    final_answer = (
                        last_message.content
                        if isinstance(last_message.content, str)
                        else str(last_message.content)
                    )

        st.divider()
        st.subheader("Final Answer")
        if final_answer:
            st.write(final_answer)
        else:
            st.info(
                "No AI final answer captured (the model may have only produced tool messages)."
            )

        st.caption(
            "All intermediate messages above mirror what pretty_print() would show, rendered inline."
        )


if __name__ == "__main__":  # pragma: no cover - manual execution path
    main()
