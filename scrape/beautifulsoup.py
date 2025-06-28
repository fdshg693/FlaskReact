from pathlib import Path
from typing import List

from bs4 import BeautifulSoup, Tag


def get_link_urls(html_file_path: Path, target_class_name: str) -> List[str]:
    """Extract href URLs from anchor tags within elements of a specific class.

    Args:
        html_file_path: Path to the HTML file
        target_class_name: CSS class name to search for

    Returns:
        List of href URLs found in the HTML

    Raises:
        SystemExit: If file cannot be read or class not found
    """
    try:
        with html_file_path.open("r", encoding="utf-8") as file_handle:
            html_content: str = file_handle.read()
    except Exception as e:
        print(f"エラー: {e}")
        exit(1)

    soup = BeautifulSoup(html_content, "html.parser")

    # .ir-list を見つけて直下の <dl> を取得
    target_element = soup.find(class_=target_class_name)
    if not target_element or not isinstance(target_element, Tag):
        print(f"エラー: .{target_class_name} が見つかりませんでした")
        exit(1)

    definition_lists = target_element.find_all("dl", recursive=False)

    extracted_urls: List[str] = []

    for definition_list in definition_lists:
        if isinstance(definition_list, Tag):
            for anchor_tag in definition_list.find_all("a"):
                if isinstance(anchor_tag, Tag):
                    href_url = anchor_tag.get("href")
                    if href_url and isinstance(href_url, str):
                        extracted_urls.append(href_url)

    return extracted_urls


if __name__ == "__main__":
    html_file_path: Path = Path(__file__).parent / "../data/headwaters.html"
    # ローカルのHTMLファイルからリンクを取得
    extracted_links: List[str] = get_link_urls(html_file_path, "ir-list")
    for link_url in extracted_links:
        print(link_url)
