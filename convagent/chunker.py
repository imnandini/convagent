from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text_file(file_path: str, chunk_size: int = 1024, chunk_overlap: int = 20) -> list[str]:
    path = Path(file_path)

    if path.suffix != ".txt":
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Only .txt files are supported."
        )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    text = path.read_text(encoding="utf-8")

    print(f"Read file: {file_path} with length: {len(text)} characters")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    print(f"Chunking file: {file_path} with chunk size: {chunk_size} and overlap: {chunk_overlap}")
    return splitter.split_text(text)