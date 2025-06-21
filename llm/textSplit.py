from langchain_text_splitters.character import RecursiveCharacterTextSplitter


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Splits the input text into chunks of specified size with overlap.

    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    sample_text = (
        f"This is a sample text that will be split into smaller chunks.\n"
        f"The text is long enough to demonstrate the splitting functionality.\n"
        f"Each chunk will have a specified size and overlap with the next chunk."
    )
    chunks = split_text(sample_text, chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk}")
