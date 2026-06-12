import sys

from convagent.qa import answer_query

if len(sys.argv) < 2:
    print('Usage: python ask.py "your question here"')
    sys.exit(1)

question = sys.argv[1]
print(f"Question: {question}\n")

result = answer_query(question, method="hybrid", k=5)

print("Answer:")
print(result["answer"])

print(f"\n{'=' * 70}")
print("Chunks used as context:")
print("=" * 70)
for i, src in enumerate(result["sources"], start=1):
    print(f"{i}. [{src['score']:.4f}] ({src['source']}) {src['snippet']}")
