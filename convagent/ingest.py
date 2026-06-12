from pathlib import Path

from convagent.chunker import chunk_text_file
from convagent.vectorstore import save_to_vectordb

data_folder = "data"

all_chunks = []
all_metadatas = []

# Go through every .txt file inside the data folder
for file_path in sorted(Path(data_folder).glob("*.txt")):
    chunks = chunk_text_file(str(file_path), chunk_size=1024, chunk_overlap=20)
    print(f"Created {len(chunks)} chunks from {file_path.name}")

    all_chunks.extend(chunks)
    # Remember which file each chunk came from
    all_metadatas.extend([{"source": file_path.name} for _ in chunks])

print(f"Total chunks across all files: {len(all_chunks)}")

save_to_vectordb(all_chunks, metadatas=all_metadatas)

print("Embeddings stored successfully")
