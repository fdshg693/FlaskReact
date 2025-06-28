from pathlib import Path
from typing import List

from bs4 import BeautifulSoup, Tag


def get_link_urls(path: Path, class_name: str) -> List[str]:
    """Extract href URLs from anchor tags within elements of a specific class.

    Args:
        path: Path to the HTML file
        class_name: CSS class name to search for

    Returns:
        List of href URLs found in the HTML

    Raises:
        SystemExit: If file cannot be read or class not found
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            html_content: str = f.read()
    except Exception as e:
        print(f"エラー: {e}")
        exit(1)

    soup = BeautifulSoup(html_content, "html.parser")

    # .ir-list を見つけて直下の <dl> を取得
    ir_list = soup.find(class_=class_name)
    if not ir_list or not isinstance(ir_list, Tag):
        print(f"エラー: .{class_name} が見つかりませんでした")
        exit(1)

    dls = ir_list.find_all("dl", recursive=False)

    href_list: List[str] = []

    for dl in dls:
        if isinstance(dl, Tag):
            for a in dl.find_all("a"):
                if isinstance(a, Tag):
                    href = a.get("href")
                    if href and isinstance(href, str):
                        href_list.append(href)

    return href_list


if __name__ == "__main__":
    file_path: Path = Path(__file__).parent / "../data/headwaters.html"
    # ローカルのHTMLファイルからリンクを取得
    links: List[str] = get_link_urls(file_path, "ir-list")
    for link in links:
        print(link)
