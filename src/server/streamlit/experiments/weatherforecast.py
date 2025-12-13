"""
OpenWeatherMap APIを使用して日本の主要都市の天気予報を表示するStreamlitアプリケーション
"""

import streamlit as st
import requests
import os
from datetime import datetime

# TODO:
# - APIキーを環境変数から取得しているが、.envから読み込む設定は未実装
# - paramsとしてAPIキーを渡してしまっているので、Bearer等の認証方式に変更するべき
# - weather_dataが単なるdictであるため、型ヒントが曖昧
#   → Pydanticモデルなどで型定義を行うと理想。最低でも、TypeDictやカスタムクラスで型定義を行うべき

# OpenWeatherMap API の設定
# APIキーは https://openweathermap.org/api で無料アカウントを作成して取得してください
# .envファイルの OPENWEATHER_API_KEY に設定するか、サイドバーで入力してください
API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city: str, api_key: str) -> dict | None:
    """指定された都市の天気情報を取得"""
    params = {
        "q": f"{city},JP",  # 日本の都市を指定
        "appid": api_key,
        "units": "metric",  # 摂氏温度
        "lang": "ja",  # 日本語
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"天気情報の取得に失敗しました: {e}")
        return None


def display_weather(weather_data: dict) -> None:
    """天気情報を表示"""
    if not weather_data:
        return

    # 基本情報
    st.subheader(f"📍 {weather_data['name']}")

    # 天気の説明
    weather_main = weather_data["weather"][0]["main"]  # noqa: F841
    weather_desc = weather_data["weather"][0]["description"]
    weather_icon = weather_data["weather"][0]["icon"]

    # アイコン表示
    icon_url = f"http://openweathermap.org/img/wn/{weather_icon}@2x.png"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(icon_url, width=100)

    with col2:
        st.markdown(f"### {weather_desc}")
        st.markdown(f"**気温**: {weather_data['main']['temp']:.1f}°C")
        st.markdown(f"**体感温度**: {weather_data['main']['feels_like']:.1f}°C")

    # 詳細情報
    st.markdown("---")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("湿度", f"{weather_data['main']['humidity']}%")
        st.metric("気圧", f"{weather_data['main']['pressure']} hPa")

    with col4:
        st.metric("風速", f"{weather_data['wind']['speed']} m/s")
        if "deg" in weather_data["wind"]:
            st.metric("風向", f"{weather_data['wind']['deg']}°")

    with col5:
        st.metric("最高気温", f"{weather_data['main']['temp_max']:.1f}°C")
        st.metric("最低気温", f"{weather_data['main']['temp_min']:.1f}°C")

    # 可視性と雲量
    if "visibility" in weather_data:
        st.markdown(f"**可視性**: {weather_data['visibility'] / 1000:.1f} km")

    if "clouds" in weather_data:
        st.markdown(f"**雲量**: {weather_data['clouds']['all']}%")

    # 日の出・日の入り
    st.markdown("---")
    col6, col7 = st.columns(2)

    with col6:
        sunrise = datetime.fromtimestamp(weather_data["sys"]["sunrise"])
        st.markdown(f"🌅 **日の出**: {sunrise.strftime('%H:%M')}")

    with col7:
        sunset = datetime.fromtimestamp(weather_data["sys"]["sunset"])
        st.markdown(f"🌇 **日の入り**: {sunset.strftime('%H:%M')}")


def format_weather_as_chat_response(weather_data: dict) -> str:
    """天気情報をチャット形式のテキストに整形"""
    if not weather_data:
        return "申し訳ございません。天気情報の取得に失敗しました。"

    city_name = weather_data["name"]
    temp = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    weather_desc = weather_data["weather"][0]["description"]
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]

    sunrise = datetime.fromtimestamp(weather_data["sys"]["sunrise"])
    sunset = datetime.fromtimestamp(weather_data["sys"]["sunset"])

    response = f"""
{city_name}の現在の天気をお知らせします！

🌡️ **気温**: {temp:.1f}°C（体感温度: {feels_like:.1f}°C）
☁️ **天気**: {weather_desc}
💧 **湿度**: {humidity}%
💨 **風速**: {wind_speed} m/s
🌅 **日の出**: {sunrise.strftime("%H:%M")}
🌇 **日の入り**: {sunset.strftime("%H:%M")}

他に気になる地名があれば教えてください！
"""
    return response


def main() -> None:
    st.set_page_config(page_title="天気予報", page_icon="🌤️", layout="centered")

    st.title("🌤️ 天気予報アプリ")
    st.markdown("日本の主要都市の現在の天気を確認できます")

    # APIキーの確認
    if API_KEY == "YOUR_API_KEY_HERE":
        st.warning("""
        ⚠️ APIキーが設定されていません
        
        以下の手順でAPIキーを取得・設定してください：
        1. [OpenWeatherMap](https://openweathermap.org/api) でアカウントを作成
        2. 無料プラン (Free) を選択してAPIキーを取得
        3. 環境変数 `OPENWEATHER_API_KEY` に設定、または下のサイドバーで入力
        """)

    # サイドバーでAPIキー入力
    with st.sidebar:
        st.header("⚙️ 設定")
        api_key_input = st.text_input(
            "APIキー",
            value=API_KEY if API_KEY != "YOUR_API_KEY_HERE" else "",
            type="password",
            help="OpenWeatherMap APIキーを入力してください",
        )

        st.markdown("---")
        st.markdown("### 📚 使い方")
        st.markdown("""
        1. 地方を選択
        2. 「天気を取得」ボタンをクリック
        3. 現在の天気情報が表示されます
        """)

    # 使用するAPIキー
    current_api_key = api_key_input if api_key_input else API_KEY

    # 日本の主要都市
    cities = {
        "北海道": "Sapporo",
        "東北（仙台）": "Sendai",
        "関東（東京）": "Tokyo",
        "関東（横浜）": "Yokohama",
        "中部（名古屋）": "Nagoya",
        "北陸（金沢）": "Kanazawa",
        "関西（大阪）": "Osaka",
        "関西（京都）": "Kyoto",
        "中国（広島）": "Hiroshima",
        "四国（松山）": "Matsuyama",
        "九州（福岡）": "Fukuoka",
        "九州（鹿児島）": "Kagoshima",
        "沖縄（那覇）": "Naha",
    }

    # 地方選択
    selected_region = st.selectbox(
        "地方を選択してください",
        options=list(cities.keys()),
        index=2,  # デフォルトは東京
    )

    # 実行ボタン
    if st.button("🔍 天気を取得", type="primary", use_container_width=True):
        if current_api_key == "YOUR_API_KEY_HERE" or not current_api_key:
            st.error("APIキーを設定してください")
        else:
            with st.spinner(f"{selected_region}の天気を取得中..."):
                city_name = cities[selected_region]
                weather_data = get_weather(city_name, current_api_key)

                if weather_data:
                    st.success("天気情報を取得しました！")
                    display_weather(weather_data)

    # チャット形式のセクション
    st.markdown("---")
    st.header("💬 チャット形式で天気を聞く")
    st.markdown("地名を入力すると、AIが天気情報を教えてくれます")

    # セッションステートの初期化
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # チャット履歴の表示
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # チャット入力
    if prompt := st.chat_input("地名を入力してください（例: 東京、大阪、福岡）"):
        # APIキーのチェック
        if current_api_key == "YOUR_API_KEY_HERE" or not current_api_key:
            st.error("APIキーを設定してください")
        else:
            # ユーザーメッセージを追加
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # アシスタントの応答
            with st.chat_message("assistant"):
                with st.spinner("天気情報を取得中..."):
                    weather_data = get_weather(prompt, current_api_key)

                    if weather_data:
                        response = format_weather_as_chat_response(weather_data)
                    else:
                        response = f"申し訳ございません。「{prompt}」の天気情報を取得できませんでした。\n\n日本の都市名（例: 東京、大阪、札幌など）を入力してください。"

                    st.markdown(response)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response}
                    )


if __name__ == "__main__":
    main()
