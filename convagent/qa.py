import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from convagent.retriever import retrieve

# Read OPENAI_API_KEY from the .env file in the project root
load_dotenv()

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def answer_query(question: str, method: str = "hybrid", k: int = 5) -> dict:
    """Retrieve the top-k chunks for the question and ask the LLM to answer from them.

    Returns a dict with the answer and the chunks that were used as context.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root."
        )

    # Step 1: find the chunks most relevant to the question
    results = retrieve(question, method=method, k=k)

    # Step 2: join the chunks into one context block, labelled by source file
    context = "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for i, (doc, _score) in enumerate(results, start=1)
    )

    # Step 3: send the context + question to the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "score": score,
                "snippet": " ".join(doc.page_content.split())[:120],
            }
            for doc, score in results
        ],
    }
