"""
query.py — Load the FAISS index and answer questions from Jenkins documentation.

Usage:
    python query.py "How do I create a Jenkins pipeline?"
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

INDEX_PATH = "jenkins_index"
TOP_K = 3

# FAISS similarity_search_with_score returns distance for FAISS.
# Lower = more similar. Tune this later if needed.
SIMILARITY_THRESHOLD = 3.0

FALLBACK = "I could not find this in the Jenkins documentation."

SYSTEM_PROMPT = """You are a Jenkins documentation assistant.
Answer the user's question STRICTLY using the context provided below.

Rules:
- Do NOT add information that is not present in the context.
- If the context does not contain enough information, respond exactly with:
  "I could not find this in the Jenkins documentation."
- Always cite the source URL(s) at the end of your answer under "Source(s):".
- Be concise and factual.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_index(embeddings: OllamaEmbeddings) -> FAISS:
    if not os.path.isdir(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_PATH}/'. "
            "Run `python ingest.py` first."
        )
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(vectorstore: FAISS, question: str) -> list[tuple]:
    """Return list of (Document, score) tuples. Lower score = more similar."""
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)

    if "plugin" not in question.lower():
        return results

    plugin_boost = 1.0
    reranked = []
    for doc, score in results:
        adjusted_score = score
        if doc.metadata.get("source_type") == "jenkins_plugin":
            adjusted_score -= plugin_boost
        reranked.append((doc, score, adjusted_score))

    reranked.sort(key=lambda item: item[2])
    return [(doc, original_score) for doc, original_score, _ in reranked]


def build_context(results: list[tuple]) -> tuple[str, list[str]]:
    """Build a combined context string and deduplicated source list."""
    context_parts = []
    sources = []

    for doc, _ in results:
        if doc.page_content:
            context_parts.append(doc.page_content.strip())

        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts).strip()
    return context, sources


def ask_llm(question: str, context: str, sources: list[str]) -> str:
    llm = OllamaLLM(
        model="mistral",
        temperature=0,
        verbose=False,
    )

    source_text = ", ".join(sources) if sources else "unknown"

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Source URLs available:
{source_text}
"""

    response = llm.invoke(prompt)

    if isinstance(response, str):
        return response.strip()

    return str(response).strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "<your question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Error: question cannot be empty.")
        sys.exit(1)

    print(f"\n🔍 Question: {question}\n")

    # 1. Load index
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vectorstore = load_index(embeddings)

    # 2. Retrieve relevant chunks
    results = retrieve(vectorstore, question)

    if not results:
        print(f"💬 Answer:\n{FALLBACK}\n")
        return

    # 3. Check best retrieval score
    best_score = results[0][1]
    print(f"   Best similarity distance: {best_score:.4f}")

    if best_score > SIMILARITY_THRESHOLD:
        print(f"\n💬 Answer:\n{FALLBACK}\n")
        return

    # 4. Build context
    context, sources = build_context(results)

    if not context:
        print(f"\n💬 Answer:\n{FALLBACK}\n")
        return

    # 5. Query LLM
    print("Querying Ollama...")
    answer = ask_llm(question, context, sources)

    # 6. Safety net (strong fallback)
    if not answer or len(answer.strip()) < 10 or "i could not find" in answer.lower() or          "not in the provided" in answer.lower() or "no mention" in answer.lower():
        answer = FALLBACK

    print(f"\n💬 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()