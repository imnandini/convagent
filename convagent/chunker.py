from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_file(file_path: str) -> list[str]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix != ".txt":
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Only .txt files are supported."
        )

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )

    return splitter.split_text(text)