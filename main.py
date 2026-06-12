import sys

from convagent.retriever import METHODS, retrieve

query = sys.argv[1] if len(sys.argv) > 1 else "how do neural networks render 3D scenes from images?"
K = 10

print(f"Query: {query}")

# Run every retrieval method on the same query
results_by_method = {m: retrieve(query, method=m, k=K) for m in METHODS}

# Show the top results per method
for method, results in results_by_method.items():
    print(f"\n{'=' * 70}")
    print(f"Method: {method}")
    print("=" * 70)
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        snippet = " ".join(doc.page_content.split())[:120]
        print(f"{i:2d}. [{score:10.4f}] ({source}) {snippet}")

# Compare how much each method agrees with the others on the top-K set
print(f"\n{'=' * 70}")
print(f"Overlap of top-{K} results between methods (out of {K})")
print("=" * 70)
top_sets = {m: {doc.page_content for doc, _ in r} for m, r in results_by_method.items()}
header = " " * 10 + "".join(f"{m:>10}" for m in METHODS)
print(header)
for m1 in METHODS:
    row = f"{m1:>10}"
    for m2 in METHODS:
        row += f"{len(top_sets[m1] & top_sets[m2]):>10}"
    print(row)
