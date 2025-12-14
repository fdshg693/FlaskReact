import streamlit as st
from openai import OpenAI
from typing import Literal, TypedDict

from config import load_dotenv_workspace

# .envを読み込む
load_dotenv_workspace()

client = OpenAI()


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def main() -> None:
    st.set_page_config(page_title="LangChain Chat", page_icon="💬", layout="centered")

    st.title("LangChain Chat")
    st.caption(
        "Single-turn Q&A Streamlit chat using your shared LangChain model initializer."
    )

    # セッションの状態を初期化
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    messages = st.session_state["messages"]

    with st.sidebar:
        with st.expander("🔍 Debug: Session State", expanded=False):
            st.json(dict(st.session_state))

    prompt = st.chat_input("Type a message")

    # On submit, render only the current exchange.
    if prompt:
        messages.append({"role": "user", "content": prompt})
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


def call_simple_model(prompt: str) -> str:
    response = client.responses.create(model="gpt-5-nano", input=prompt)
    return response.output_text
