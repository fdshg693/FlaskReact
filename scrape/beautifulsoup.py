from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

# ローカルの../data/headwaters.html を取得して、.ir-list の直下の <dl> の <a> タグ href を出力する
path = os.path.join(
    os.path.dirname(__file__), "../data/headwaters.html"
)  # ローカルのHTMLファイルのパスを指定
try:
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
except Exception as e:
    print(f"エラー: {e}")
    exit(1)

soup = BeautifulSoup(html_content, "html.parser")

# .ir-list を見つけて直下の <dl> を取得
ir_list = soup.find(class_="ir-list")
if not ir_list:
    print("エラー: .ir-list が見つかりませんでした")
    exit(1)

dls = ir_list.find_all("dl", recursive=False)

# 各 <dl> 配下の <a> タグ href を出力
for dl in dls:
    for a in dl.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        print(href)
