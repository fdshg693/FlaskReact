from __future__ import annotations

import streamlit as st

from llm.langchain_custom.examples.simple_call import main as call_simple_model


def main() -> None:
    st.set_page_config(page_title="LangChain Chat", page_icon="💬", layout="centered")

    st.title("LangChain Chat")
    st.caption(
        "Single-turn Q&A Streamlit chat using your shared LangChain model initializer."
    )

    with st.sidebar:
        with st.expander("🔍 Debug: Session State", expanded=False):
            st.json(dict(st.session_state))

    last_prompt = st.session_state.get("last_prompt")
    last_answer = st.session_state.get("last_answer")

    if last_prompt:
        with st.chat_message("user"):
            st.markdown(last_prompt)
    if last_answer:
        with st.chat_message("assistant"):
            st.markdown(last_answer)

    prompt = st.chat_input("Type a message")
    if not prompt:
        return

    st.chat_message("user").markdown(prompt)
    st.session_state.last_prompt = prompt
    st.session_state.last_answer = None

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = call_simple_model(prompt)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Model call failed: {exc}")
                return

        st.markdown(answer)

    st.session_state.last_answer = answer
    st.rerun()


if __name__ == "__main__":
    main()
