from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read file
with open("data/sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

# Split text
chunks = splitter.split_text(text)

# Print chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)