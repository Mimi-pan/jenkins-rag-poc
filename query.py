"""
query.py — Load the FAISS index and answer questions from Jenkins documentation.

Usage:
    python query.py "How do I create a Jenkins pipeline?"
"""

import sys

from rag_core import (
    FALLBACK,
    evaluate_retrieval,
    load_pipeline,
    query_rag,
)


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "<your question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Error: question cannot be empty.")
        sys.exit(1)

    print(f"\n🔍 Question: {question}\n")

    vectorstore, chunks, bm25 = load_pipeline()
    if bm25:
        print("   BM25 index loaded ✓")

    decision = evaluate_retrieval(question, vectorstore, chunks=chunks, bm25=bm25)
    if not decision.results:
        print(f"💬 Answer:\n{FALLBACK}\n")
        return

    if decision.best_score is not None and decision.best_score > 0:
        print(f"   Best similarity distance: {decision.best_score:.4f}")

    if decision.should_fallback:
        print(f"\n💬 Answer:\n{FALLBACK}\n")
        return

    print("Querying Ollama...")
    answer, _ = query_rag(question, vectorstore=vectorstore, chunks=chunks, bm25=bm25)
    print(f"\n💬 Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
