"""
query.py — Load the FAISS index and answer questions from Jenkins documentation.

Usage:
    python query.py "How do I create a Jenkins pipeline?"
"""

import sys

from rag_core import (
    FALLBACK,
    SIMILARITY_THRESHOLD,
    build_context,
    has_question_support,
    load_pipeline,
    query_rag,
    retrieve,
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

    results = retrieve(vectorstore, question, chunks=chunks, bm25=bm25)
    if not results:
        print(f"💬 Answer:\n{FALLBACK}\n")
        return

    best_score = results[0][1]
    if best_score > 0:
        print(f"   Best similarity distance: {best_score:.4f}")
        if best_score > SIMILARITY_THRESHOLD:
            print(f"\n💬 Answer:\n{FALLBACK}\n")
            return

    context, _ = build_context(results)
    if not context or not has_question_support(question, results):
        print(f"\n💬 Answer:\n{FALLBACK}\n")
        return

    print("Querying Ollama...")
    answer, _ = query_rag(question, vectorstore=vectorstore, chunks=chunks, bm25=bm25)
    print(f"\n💬 Answer:\n{answer}\n")


if __name__ == "__main__":
    main()
