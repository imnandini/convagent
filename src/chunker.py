from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """ 
This is sample document. It contains multiple sentences. The purpose of this document is to demonstrate how to use the RecursiveCharacterTextSplitter from the langchain_text_splitters library. The text splitter will break this document into smaller chunks based on the specified parameters.
"""
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")