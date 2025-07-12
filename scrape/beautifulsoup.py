from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger


def get_all_links_from_url(url: str, timeout: int = 30) -> List[str]:
    """Extract all href URLs from anchor tags in the HTML page.

    Args:
        url: URL to fetch HTML from
        timeout: Request timeout in seconds (default: 30)

    Returns:
        List of all href URLs found in the HTML

    Raises:
        requests.RequestException: If the HTTP request fails
    """
    try:
        logger.info(f"Fetching all links from URL: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        html_content = response.text
        logger.success(f"Successfully fetched HTML from {url}")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch HTML from {url}: {e}")
        raise

    soup = BeautifulSoup(html_content, "html.parser")
    extracted_urls: List[str] = []

    for anchor_tag in soup.find_all("a"):
        if isinstance(anchor_tag, Tag):
            href_url = anchor_tag.get("href")
            if href_url and isinstance(href_url, str):
                # Convert relative URLs to absolute URLs
                absolute_url = urljoin(url, href_url)
                extracted_urls.append(absolute_url)

    logger.info(f"Found {len(extracted_urls)} links")
    return extracted_urls


def _extract_links_from_html(
    html_content: str, target_class_name: str, base_url: Optional[str] = None
) -> List[str]:
    """Extract links from HTML content with a specific class name.

    Args:
        html_content: HTML content as string
        target_class_name: CSS class name to search for
        base_url: Base URL for converting relative URLs to absolute URLs

    Returns:
        List of href URLs found in the HTML

    Raises:
        ValueError: If the target class is not found
    """
    soup = BeautifulSoup(html_content, "html.parser")

    target_element = soup.find(class_=target_class_name)
    if not target_element or not isinstance(target_element, Tag):
        error_msg = f"Class '{target_class_name}' not found in HTML"
        logger.error(error_msg)
        raise ValueError(error_msg)

    definition_lists = target_element.find_all("dl", recursive=False)
    extracted_urls: List[str] = []

    for definition_list in definition_lists:
        if isinstance(definition_list, Tag):
            for anchor_tag in definition_list.find_all("a"):
                if isinstance(anchor_tag, Tag):
                    href_url = anchor_tag.get("href")
                    if href_url and isinstance(href_url, str):
                        # Convert relative URLs to absolute URLs if base_url is provided
                        if base_url:
                            absolute_url = urljoin(base_url, href_url)
                            extracted_urls.append(absolute_url)
                        else:
                            extracted_urls.append(href_url)

    logger.info(f"Found {len(extracted_urls)} links in class '{target_class_name}'")
    return extracted_urls


def get_link_urls(html_file_path: Path, target_class_name: str) -> List[str]:
    """Extract href URLs from anchor tags within elements of a specific class from a local HTML file.

    Args:
        html_file_path: Path to the HTML file
        target_class_name: CSS class name to search for

    Returns:
        List of href URLs found in the HTML

    Raises:
        FileNotFoundError: If the HTML file cannot be found
        ValueError: If the target class is not found
        IOError: If the file cannot be read
    """
    try:
        with html_file_path.open("r", encoding="utf-8") as file_handle:
            html_content: str = file_handle.read()
        logger.info(f"Successfully read HTML file: {html_file_path}")
    except FileNotFoundError:
        error_msg = f"HTML file not found: {html_file_path}"
        logger.error(error_msg)
        raise
    except IOError as e:
        error_msg = f"Failed to read HTML file {html_file_path}: {e}"
        logger.error(error_msg)
        raise

    return _extract_links_from_html(html_content, target_class_name)


if __name__ == "__main__":
    # ローカルHTMLファイルからリンクを取得するテスト
    html_file_path: Path = Path(__file__).parent / "../data/headwaters.html"

    try:
        logger.info("Testing local HTML file link extraction...")
        extracted_links: List[str] = get_link_urls(html_file_path, "ir-list")
        logger.info(f"Found {len(extracted_links)} links from local file:")
        for link_url in extracted_links:
            print(f"  - {link_url}")
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error(f"Failed to extract links from local file: {e}")
