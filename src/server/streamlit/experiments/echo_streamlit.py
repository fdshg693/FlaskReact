"""
ユーザーの入力をそのまま反復するシンプルなStreamlitアプリ
"""

from __future__ import annotations

import streamlit as st
import time


def main() -> None:
    st.set_page_config(page_title="Echo Chat", page_icon="💬", layout="centered")

    st.title("Echo Chat")
    st.caption("Echo Streamlit app that repeats user input back to them.")

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
                time.sleep(1)  # Simulate processing delay
                try:
                    answer = prompt  # Echo the user input
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
