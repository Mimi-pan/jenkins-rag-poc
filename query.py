"""
query.py — Load the FAISS index and answer questions from Jenkins documentation.

Usage:
    python query.py "How do I create a Jenkins pipeline?"
"""

import os
import sys
import pickle
import re
import hashlib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rank_bm25 import BM25Okapi

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

INDEX_PATH = "jenkins_index"
TOP_K = 3

# FAISS similarity_search_with_score returns distance for FAISS.
# Lower = more similar. Tune this later if needed.
SIMILARITY_THRESHOLD = 1.5

FALLBACK = "I could not find this in the Jenkins documentation."
STOPWORDS = {
    "a", "an", "and", "are", "best", "can", "do", "does", "for", "help",
    "how", "i", "in", "is", "it", "my", "of", "provide", "provides",
    "should", "tell", "the", "to", "what", "with",
}
UNSUPPORTED_PATTERNS = [
    "i could not find",
    "not in the provided",
    "no mention",
    "not explicitly mentioned",
    "not explicitly state",
    "not explicitly specify",
    "not available in the provided context",
    "based on the provided context",
    "the provided context does not",
    "the context does not",
    "the documentation does not specify",
    "there isn't a specific",
    "there is no specific",
    "it depends",
    "however, the context",
    "however, it",
]

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


def load_chunks() -> list[dict]:
    """Load raw chunks saved by ingest.py."""
    path = os.path.join(INDEX_PATH, "chunks.pkl")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def build_bm25(chunks: list[dict]) -> BM25Okapi | None:
    """Build a BM25 index from chunk texts."""
    if not chunks:
        return None
    tokenized = [tokenize_text(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for lexical retrieval while keeping plugin-like terms intact."""
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def make_result_key(text: str, metadata: dict | None = None) -> str:
    """Create a stable key for deduplicating retrieval results."""
    source = str((metadata or {}).get("source", "unknown"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def extract_query_terms(question: str) -> list[str]:
    """Keep only meaningful query terms for support checks."""
    return [
        token for token in tokenize_text(question)
        if token not in STOPWORDS and len(token) >= 3
    ]


def has_question_support(question: str, results: list[tuple]) -> bool:
    """Reject answers when retrieved text does not support the query terms."""
    terms = extract_query_terms(question)
    if not terms:
        return True

    searchable_parts = []
    for doc, _ in results:
        metadata = doc.metadata or {}
        searchable_parts.extend([
            doc.page_content,
            str(metadata.get("title", "")),
            str(metadata.get("source", "")),
            str(metadata.get("plugin_name", "")),
            str(metadata.get("plugin_id", "")),
        ])

    haystack = "\n".join(part.lower() for part in searchable_parts if part)
    matched_terms = [term for term in terms if term in haystack]

    if len(terms) == 1:
        return len(matched_terms) == 1

    required_matches = min(2, len(terms))
    return len(set(matched_terms)) >= required_matches


def retrieve(
    vectorstore: FAISS,
    question: str,
    chunks: list[dict] | None = None,
    bm25: BM25Okapi | None = None,
) -> list[tuple]:
    """Hybrid FAISS + BM25 retrieval with Reciprocal Rank Fusion (RRF)."""
    CANDIDATE_K = min(10, TOP_K * 3)
    RRF_K = 60
    lowered = question.lower()

    # ── 1. FAISS retrieval ────────────────────────────────────────────────────
    faiss_results = vectorstore.similarity_search_with_score(question, k=CANDIDATE_K)

    rrf_scores: dict[str, float] = {}
    doc_store:  dict[str, Document] = {}
    faiss_score_store: dict[str, float] = {}

    for rank, (doc, score) in enumerate(faiss_results):
        key = make_result_key(doc.page_content, doc.metadata)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
        doc_store[key] = doc
        faiss_score_store[key] = score

    # ── 2. BM25 retrieval ─────────────────────────────────────────────────────
    if bm25 and chunks:
        tokens = tokenize_text(question)
        bm25_scores = bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:CANDIDATE_K]

        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] <= 0:
                break
            chunk = chunks[idx]
            key = make_result_key(chunk["text"], chunk.get("metadata", {}))
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            if key not in doc_store:
                doc_store[key] = Document(
                    page_content=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
                faiss_score_store[key] = 0.0   # BM25-only doc has no FAISS distance

    # ── 3. Plugin-aware boost ─────────────────────────────────────────────────
    candidates = []
    for key, rrf in rrf_scores.items():
        doc = doc_store[key]
        adjusted_rrf = rrf

        if "plugin" in lowered:
            meta = doc.metadata or {}
            plugin_name = str(meta.get("plugin_name", "")).lower()
            plugin_id   = str(meta.get("plugin_id",   "")).lower()
            if meta.get("source_type") == "jenkins_plugin":
                adjusted_rrf += 0.02
            if plugin_name and plugin_name in lowered:
                adjusted_rrf += 0.05
            if plugin_id and plugin_id in lowered:
                adjusted_rrf += 0.05

        candidates.append((doc, faiss_score_store[key], adjusted_rrf))

    # ── 4. Sort by RRF descending, return TOP_K ───────────────────────────────
    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(doc, score) for doc, score, _ in candidates[:TOP_K]]


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


def should_force_fallback(answer: str) -> bool:
    low = answer.lower().strip()
    return (
        not low
        or len(low) < 10
        or any(pattern in low for pattern in UNSUPPORTED_PATTERNS)
    )


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

    # 2. Load BM25 (keyword retrieval)
    chunks = load_chunks()
    bm25 = build_bm25(chunks)
    if bm25:
        print("   BM25 index loaded ✓")

    # 3. Retrieve relevant chunks (hybrid FAISS + BM25)
    results = retrieve(vectorstore, question, chunks=chunks, bm25=bm25)

    if not results:
        print(f"💬 Answer:\n{FALLBACK}\n")
        return

    # 4. Check best retrieval score (FAISS distance; 0.0 means BM25-only doc)
    best_score = results[0][1]
    if best_score > 0:
        print(f"   Best similarity distance: {best_score:.4f}")
        if best_score > SIMILARITY_THRESHOLD:
            print(f"\n💬 Answer:\n{FALLBACK}\n")
            return

    # 5. Build context
    context, sources = build_context(results)

    if not context:
        print(f"\n💬 Answer:\n{FALLBACK}\n")
        return

    if not has_question_support(question, results):
        print(f"\n💬 Answer:\n{FALLBACK}\n")
        return

    # 6. Query LLM
    print("Querying Ollama...")
    answer = ask_llm(question, context, sources)

    # 7. Safety net (strong fallback)
    if should_force_fallback(answer):
        answer = FALLBACK

    print(f"\n💬 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
