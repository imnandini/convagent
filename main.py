from convagent.retriever import retrieve

query = "What is LSTM?"

results = retrieve(query)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print(doc.page_content)