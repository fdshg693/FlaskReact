from langchain_core.tools import tool
import os


@tool
def searchLocalDocuments() -> str:
    """Lists all names and summary content of text files about headwaters Company."""

    data_path = os.path.join(os.path.dirname(__file__), "../data/")
    # data_pathの直下にあるTXTファイルを検索する
    txt_files = [f for f in os.listdir(data_path) if f.endswith(".txt")]

    if not txt_files:
        return "No text files found in the data directory."

    # ファイル名と１行目の内容をリストで返す
    file_info = []
    for txt_file in txt_files:
        with open(os.path.join(data_path, txt_file), "r") as f:
            first_line = f.readline().strip()
            file_info.append(f"{txt_file}: {first_line}")
    return "\n".join(file_info) if file_info else "No text files found."


@tool
def getLocalDocuments(txtFileName: str) -> str:
    """Fetches most recent text about headwaters Company.
    Args:
        txtFileName (str): The name of the text file to fetch.
        Example: "oooo.txt"
    Returns:
        str: The content of the text file.
    """

    txt_path = os.path.join(os.path.dirname(__file__), "../data/", txtFileName)

    with open(txt_path, "r") as f:
        docs = f.readlines()

    return "".join(docs) if docs else "No documents found."


if __name__ == "__main__":
    # Test the searchLocalDocuments function
    print(searchLocalDocuments())
