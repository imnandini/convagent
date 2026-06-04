from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def save_to_vectordb(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B"
    )

    print(f"Creating vector store with {len(chunks)} chunks")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("Persisting vector store to disk")
    return vectordb