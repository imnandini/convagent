from chunker import chunk_text_file
from vectorstore import save_to_vectordb

chunks = chunk_text_file("data/LSTM.txt", chunk_size=1024, chunk_overlap=20)

print(f"Created {len(chunks)} chunks")

save_to_vectordb(chunks)

print("Embeddings stored successfully")