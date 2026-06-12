from langchain_chroma import Chroma

from convagent.embedder import get_embeddings


def save_to_vectordb(chunks, metadatas=None):

    print(f"Creating vector store with {len(chunks)} chunks")

    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings()
    )

    # Drop previously ingested chunks so re-running ingest doesn't duplicate them
    vectordb.reset_collection()

    vectordb.add_texts(
        texts=chunks,
        metadatas=metadatas
    )

    print("Persisting vector store to disk")
    return vectordb
