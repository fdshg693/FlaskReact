"""Streamlit playground for the ReAct agent.

Features:
 - Enter a prompt
 - Select model and provider from dropdowns
 - Select which tools the agent may use (multi-select)
 - Stream and display ALL intermediate reasoning / tool messages
 - Show the final AI answer separately

Run:
  uv run streamlit run src/streamlit/agent_app.py
"""

from __future__ import annotations

from typing import List

import streamlit as st
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger

from llm import AgentPrompt, LLMModel, ModelProvider, agent_run
from llm.tools.others.sample_tools import add_numbers
from llm.tools.search.tavily_search import tavily_search_func
from llm.tools.search.local_document_search import search_local_text_documents


# Available tools registry
AVAILABLE_TOOLS: dict[str, BaseTool] = {
    "add_numbers": add_numbers,
    "tavily_search": tavily_search_func,
    "search_local_documents": search_local_text_documents,
}


def format_message(msg: BaseMessage) -> tuple[str, str]:
    """Format a message for display.

    Returns:
        Tuple of (message_type, formatted_content)
    """
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            tool_info = "\n".join(
                f"  - Tool: {tc['name']}\n    Args: {tc['args']}"
                for tc in msg.tool_calls
            )
            return ("ai_tool", f"🤖 **AI calling tools:**\n{tool_info}")
        else:
            return ("ai", f"🤖 **AI:** {msg.content}")
    elif isinstance(msg, ToolMessage):
        return ("tool", f"🔧 **Tool Result ({msg.name}):**\n```\n{msg.content}\n```")
    elif hasattr(msg, "content"):
        return ("other", f"💬 **Message:** {msg.content}")
    else:
        return ("other", f"💬 **Message:** {str(msg)}")


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Agent Playground",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 ReAct Agent Playground")
    st.markdown("Test LangChain agents with different models and tools")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        st.subheader("Model Settings")
        selected_provider = st.selectbox(
            "Provider",
            options=[p.value for p in ModelProvider],
            index=0,
        )

        selected_model = st.selectbox(
            "Model",
            options=[m.value for m in LLMModel],
            index=2,  # Default to GPT-4O-MINI
        )

        # Tools selection
        st.subheader("Tools")
        selected_tool_names = st.multiselect(
            "Select tools for the agent",
            options=list(AVAILABLE_TOOLS.keys()),
            default=["add_numbers"],
            help="Choose which tools the agent can use",
        )

        # Display selected tools info
        if selected_tool_names:
            st.info(f"✅ {len(selected_tool_names)} tool(s) selected")
            for tool_name in selected_tool_names:
                tool = AVAILABLE_TOOLS[tool_name]
                with st.expander(f"📖 {tool_name}"):
                    st.code(tool.description)
        else:
            st.warning("⚠️ No tools selected")

    # Main content area
    st.header("💭 Prompt")
    user_prompt = st.text_area(
        "Enter your prompt:",
        value="Add the numbers 1.5, 2.5, and 3.0 together.",
        height=100,
        placeholder="What would you like the agent to do?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button("🚀 Run Agent", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear Results", use_container_width=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = None

    if clear_button:
        st.session_state.messages = []
        st.session_state.final_answer = None
        st.rerun()

    if run_button:
        if not user_prompt.strip():
            st.error("❌ Please enter a prompt")
            return

        if not selected_tool_names:
            st.warning("⚠️ No tools selected - agent will run without tools")

        # Clear previous results
        st.session_state.messages = []
        st.session_state.final_answer = None

        # Prepare tools
        selected_tools: List[BaseTool] = [
            AVAILABLE_TOOLS[name] for name in selected_tool_names
        ]

        # Create prompt
        prompt = AgentPrompt(content=user_prompt)

        # Progress indicator
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Agent is thinking...")

        # Results container
        results_container = st.container()

        try:
            # Run agent with streaming
            for message in agent_run(
                prompt,
                selected_tools,
                LLMModel(selected_model),
                ModelProvider(selected_provider),
            ):
                st.session_state.messages.append(message)

                # Display intermediate steps in real-time
                with results_container:
                    st.subheader("🔍 Agent Thinking Process")
                    for msg in st.session_state.messages:
                        msg_type, formatted = format_message(msg)

                        if msg_type == "ai":
                            st.markdown(formatted)
                        elif msg_type == "ai_tool":
                            st.info(formatted)
                        elif msg_type == "tool":
                            st.code(formatted, language="text")
                        else:
                            st.text(formatted)

                        st.divider()

            # Extract final answer (last AI message without tool calls)
            for msg in reversed(st.session_state.messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                    st.session_state.final_answer = msg.content
                    break

            # Clear progress indicator
            progress_container.empty()

        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Error running agent: {str(e)}")
            logger.exception("Agent execution failed")
            return

    # Display final answer if available
    if st.session_state.final_answer:
        st.success("✅ Agent completed successfully!")
        st.subheader("🎯 Final Answer")
        st.markdown(
            f"""
            <div style="
                background-color: #e8f5e9;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4caf50;
            ">
                {st.session_state.final_answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display thinking process if available
    if st.session_state.messages and not run_button:
        with st.expander("🔍 View Agent Thinking Process", expanded=False):
            for msg in st.session_state.messages:
                msg_type, formatted = format_message(msg)

                if msg_type == "ai":
                    st.markdown(formatted)
                elif msg_type == "ai_tool":
                    st.info(formatted)
                elif msg_type == "tool":
                    st.code(formatted, language="text")
                else:
                    st.text(formatted)

                st.divider()


if __name__ == "__main__":
    main()
