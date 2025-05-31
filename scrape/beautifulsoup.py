from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os


def get_link_urls(path, class_name):
    try:
        with open(path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        print(f"エラー: {e}")
        exit(1)

    soup = BeautifulSoup(html_content, "html.parser")

    # .ir-list を見つけて直下の <dl> を取得
    ir_list = soup.find(class_=class_name)
    if not ir_list:
        print(f"エラー: .{class_name} が見つかりませんでした")
        exit(1)

    dls = ir_list.find_all("dl", recursive=False)

    href_list = []

    for dl in dls:
        for a in dl.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            href_list.append(href)

    return href_list


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "../data/headwaters.html")
    # ローカルのHTMLファイルからリンクを取得
    links = get_link_urls(path, "ir-list")
    for link in links:
        print(link)
