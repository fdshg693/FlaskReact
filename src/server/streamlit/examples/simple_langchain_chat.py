"""
ユーザーとのシングルターンQ&Aチャットを提供するStreamlitアプリケーション
"""

from __future__ import annotations

import streamlit as st

from config import load_dotenv_workspace


def main() -> None:
    # .envを読み込む（既存のシステム環境変数は上書きしない）
    load_dotenv_workspace()

    # 実行時に import して、import-time 副作用を避ける
    from llm.langchain_custom.examples.simple_call import main as call_simple_model

    st.set_page_config(page_title="LangChain Chat", page_icon="💬", layout="centered")

    st.title("LangChain Chat")
    st.caption(
        "Single-turn Q&A Streamlit chat using your shared LangChain model initializer."
    )

    with st.sidebar:
        with st.expander("🔍 Debug: Session State", expanded=False):
            st.json(dict(st.session_state))

    prompt = st.chat_input("Type a message")

    # On submit, render only the current exchange.
    if prompt:
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            body = st.empty()  # 回答表示専用の場所を確保
            body.markdown("")  # 先に空で上書き（これで前回の残像を消す）
            with st.spinner("Thinking..."):
                try:
                    answer = call_simple_model(prompt)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Model call failed: {exc}")
                    st.stop()

            body.markdown(answer)

        st.session_state.last_prompt = prompt
        st.session_state.last_answer = answer
        st.stop()

    # Otherwise, show only the last exchange.
    last_prompt = st.session_state.get("last_prompt")
    last_answer = st.session_state.get("last_answer")

    if last_prompt:
        with st.chat_message("user"):
            st.markdown(last_prompt)
    if last_answer:
        with st.chat_message("assistant"):
            st.markdown(last_answer)


if __name__ == "__main__":
    main()
