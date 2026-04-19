"""
Shared RAG core for the Jenkins documentation assistant.
"""

import hashlib
import os
import pickle
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rank_bm25 import BM25Okapi

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral")

INDEX_PATH = "jenkins_index"
TOP_K = 3
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.5"))
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

SYSTEM_PROMPT = """
You are a Jenkins assistant using retrieved context.

Rules:
If the answer is not explicitly supported by the context, respond exactly with:"I could not find this in the Jenkins documentation."
- Do not infer, guess, or generalize beyond the retrieved text.
- If the exact answer is not explicitly supported by the context, say that the information is not available in the retrieved context.
- If the question mentions "plugin", prioritize information from plugins.jenkins.io sources.
- If both documentation and plugin pages exist, prefer plugin-specific answers when relevant.
- If the question is about a specific plugin (e.g., Git plugin, Kubernetes plugin), try to match the plugin name in the context.
- Always include the source URL(s) at the end of your answer under "Sources:".
- Keep the answer concise and factual.
"""

WORKFLOW_KEYWORDS = {
    "pipeline": ["pipeline", "jenkinsfile", "stage"],
    "credentials": ["credential", "credentials", "secret", "token", "password"],
    "troubleshooting": ["troubleshoot", "debug", "failure", "failed", "error", "issue", "broken"],
}


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


def detect_workflow_mode(question: str) -> str | None:
    """Detect common workflow-oriented questions that benefit from step-by-step answers."""
    lowered = question.lower()
    for mode, keywords in WORKFLOW_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return mode
    return None


def build_response_instructions(question: str) -> str:
    """Add response-shape guidance for common workflow questions."""
    mode = detect_workflow_mode(question)
    if mode is None:
        return "Answer in a short paragraph followed by a brief source-backed summary if helpful."

    if mode == "pipeline":
        return (
            "Answer with numbered steps for setting up or understanding the pipeline. "
            "Mention relevant Jenkinsfile concepts or stages only if supported by the context."
        )

    if mode == "credentials":
        return (
            "Answer with numbered steps for handling credentials safely in Jenkins. "
            "Mention required Jenkins features or syntax only if explicitly supported by the context."
        )

    return (
        "Answer with numbered troubleshooting steps. "
        "Start with the most direct checks suggested by the context and avoid unsupported advice."
    )


def has_question_support(question: str, results: list[tuple[Document, float]]) -> bool:
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


@lru_cache(maxsize=1)
def create_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def load_index(embeddings: OllamaEmbeddings | None = None) -> FAISS:
    if not os.path.isdir(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{INDEX_PATH}/'. "
            "Run `python ingest.py` first."
        )
    return FAISS.load_local(
        INDEX_PATH,
        embeddings or create_embeddings(),
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


def load_pipeline() -> tuple[FAISS, list[dict], BM25Okapi | None]:
    embeddings = create_embeddings()
    vectorstore = load_index(embeddings)
    chunks = load_chunks()
    bm25 = build_bm25(chunks)
    return vectorstore, chunks, bm25


def retrieve(
    vectorstore: FAISS,
    question: str,
    chunks: list[dict] | None = None,
    bm25: BM25Okapi | None = None,
) -> list[tuple[Document, float]]:
    """Hybrid FAISS + BM25 retrieval with Reciprocal Rank Fusion (RRF)."""
    candidate_k = min(10, TOP_K * 3)
    rrf_k = 60
    lowered = question.lower()

    faiss_results = vectorstore.similarity_search_with_score(question, k=candidate_k)

    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, Document] = {}
    faiss_score_store: dict[str, float] = {}

    for rank, (doc, score) in enumerate(faiss_results):
        key = make_result_key(doc.page_content, doc.metadata)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rrf_k + rank + 1)
        doc_store[key] = doc
        faiss_score_store[key] = score

    if bm25 and chunks:
        tokens = tokenize_text(question)
        bm25_scores = bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:candidate_k]

        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] <= 0:
                break
            chunk = chunks[idx]
            key = make_result_key(chunk["text"], chunk.get("metadata", {}))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rrf_k + rank + 1)
            if key not in doc_store:
                doc_store[key] = Document(
                    page_content=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
                faiss_score_store[key] = 0.0

    candidates = []
    for key, rrf_score in rrf_scores.items():
        doc = doc_store[key]
        adjusted_rrf = rrf_score

        if "plugin" in lowered:
            metadata = doc.metadata or {}
            plugin_name = str(metadata.get("plugin_name", "")).lower()
            plugin_id = str(metadata.get("plugin_id", "")).lower()
            if metadata.get("source_type") == "jenkins_plugin":
                adjusted_rrf += 0.02
            if plugin_name and plugin_name in lowered:
                adjusted_rrf += 0.05
            if plugin_id and plugin_id in lowered:
                adjusted_rrf += 0.05

        candidates.append((doc, faiss_score_store[key], adjusted_rrf))

    candidates.sort(key=lambda item: item[2], reverse=True)
    return [(doc, score) for doc, score, _ in candidates[:TOP_K]]


def build_context(results: list[tuple[Document, float]]) -> tuple[str, list[str]]:
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


def strip_inline_sources(answer: str) -> str:
    """Keep sources in the UI panel only, not in the model answer body."""
    cleaned = re.sub(r"(?is)\n*\**source\(s\)\**:\s*.*$", "", answer).strip()
    cleaned = re.sub(r"(?is)\n*\**sources\**:\s*.*$", "", cleaned).strip()
    return cleaned


def ask_llm(question: str, context: str, sources: list[str]) -> str:
    llm = OllamaLLM(
        model=OLLAMA_LLM_MODEL,
        temperature=0,
        verbose=False,
    )

    source_text = ", ".join(sources) if sources else "unknown"
    response_instructions = build_response_instructions(question)
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Response format:
{response_instructions}

Source URLs available:
{source_text}
"""

    response = llm.invoke(prompt)
    answer = response.strip() if isinstance(response, str) else str(response).strip()
    return strip_inline_sources(answer)


def query_rag(
    question: str,
    vectorstore: FAISS | None = None,
    chunks: list[dict] | None = None,
    bm25: BM25Okapi | None = None,
) -> tuple[str, list[str]]:
    """Run the full RAG pipeline. Returns (answer, sources)."""
    if vectorstore is None or chunks is None:
        vectorstore, chunks, default_bm25 = load_pipeline()
        if bm25 is None:
            bm25 = default_bm25

    results = retrieve(vectorstore, question, chunks=chunks, bm25=bm25)
    if not results:
        return FALLBACK, []

    best_score = results[0][1]
    if best_score > 0 and best_score > SIMILARITY_THRESHOLD:
        return FALLBACK, []

    context, sources = build_context(results)
    if not context or not has_question_support(question, results):
        return FALLBACK, []

    answer = ask_llm(question, context, sources)
    if should_force_fallback(answer):
        return FALLBACK, []

    return answer, sources
