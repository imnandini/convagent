from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def retrieve(query: str):

    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B"
    )

    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    results = vectordb.similarity_search(
        query=query,
        k=3
    )

    return results