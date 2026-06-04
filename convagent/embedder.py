import torch
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": device},
        )

    def create_embeddings(self, chunks: list[str]):
        return self.model.embed_documents(chunks)