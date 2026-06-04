from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_file(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )

    chunks = splitter.split_text(text)

    return chunks