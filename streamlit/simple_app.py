import streamlit as st

st.title("Streamlit サンプルページ")

st.write("こんにちは！これはStreamlitで作成したシンプルなページです。")

name = st.text_input("お名前を入力してください：")
if name:
    st.success(f"ようこそ、{name}さん！")

number = st.slider("好きな数字を選んでください", 0, 100, 50)
st.write(f"選択した数字: {number}")
