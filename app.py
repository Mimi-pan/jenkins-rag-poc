"""
app.py — Streamlit chat UI for Jenkins RAG PoC
Usage:
    streamlit run app.py
"""

import os
import re
import pickle
import hashlib
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
INDEX_PATH             = "jenkins_index"
TOP_K                  = 3
SIMILARITY_THRESHOLD   = 3.0
FALLBACK               = "I could not find this in the Jenkins documentation."
STOPWORDS = {
    "a", "an", "and", "are", "best", "can", "do", "does", "for", "help",
    "how", "i", "in", "is", "it", "my", "of", "provide", "provides",
    "should", "tell", "the", "to", "what", "with",
}

SYSTEM_PROMPT = """
You are a Jenkins assistant using retrieved context.

Rules:
If the answer is not explicitly supported by the context, respond exactly with:"I could not find this in the Jenkins documentation."
- Do not infer, guess, or generalize beyond the retrieved text.
- If the exact answer is not explicitly supported by the context, say that the information is not available in the retrieved context.
- If the question mentions "plugin", prioritize information from plugins.jenkins.io sources.
- If both documentation and plugin pages exist, prefer plugin-specific answers when relevant.
- If the question is about a specific plugin (e.g., Git plugin, Kubernetes plugin), try to match the plugin name in the context.
- Do not include a "Source(s):" or "Sources:" section in the answer.
- Do not paste URLs in the answer body.

Answer clearly and concisely.
"""

# ── Load index (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_index():
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_chunks() -> list[dict]:
    path = os.path.join(INDEX_PATH, "chunks.pkl")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_bm25():
    chunks = load_chunks()
    if not chunks:
        return None
    tokenized = [tokenize_text(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def make_result_key(text: str, metadata: dict | None = None) -> str:
    source = str((metadata or {}).get("source", "unknown"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def extract_query_terms(question: str) -> list[str]:
    return [
        token for token in tokenize_text(question)
        if token not in STOPWORDS and len(token) >= 3
    ]


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


def retrieve_hybrid(
    vectorstore: FAISS,
    question: str,
    chunks: list[dict] | None = None,
    bm25: BM25Okapi | None = None,
) -> list[tuple[Document, float]]:
    """Hybrid FAISS + BM25 retrieval with plugin-aware reranking."""
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


def query_rag(question: str) -> tuple[str, list[str]]:
    """Run the full RAG pipeline. Returns (answer, sources)."""
    vectorstore = load_index()
    chunks = load_chunks()
    bm25 = load_bm25()
    results = retrieve_hybrid(vectorstore, question, chunks=chunks, bm25=bm25)

    if not results:
        return FALLBACK, []

    best_score = results[0][1]
    if best_score > SIMILARITY_THRESHOLD:
        return FALLBACK, []

    # Build context
    context_parts, sources = [], []
    for doc, _ in results:
        if doc.page_content:
            context_parts.append(doc.page_content.strip())
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts).strip()
    if not context:
        return FALLBACK, []

    if not has_question_support(question, results):
        return FALLBACK, []

    # Query LLM
    llm = OllamaLLM(model="mistral", temperature=0, verbose=False)
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Source URLs available:
{", ".join(sources)}
"""
    answer = llm.invoke(prompt).strip()
    answer = strip_inline_sources(answer)

    # Safety net
    unsupported_patterns = [
        "i could not find",
        "not in the provided",
        "no mention",
        "not explicitly mentioned",
        "it seems",
        "likely",
        "appears to",
        "not available in the provided context",
    ]
    if (
        not answer
        or len(answer) < 10
        or any(pattern in answer.lower() for pattern in unsupported_patterns)
    ):
        return FALLBACK, []

    return answer, sources


def strip_inline_sources(answer: str) -> str:
    """Keep sources in the UI panel only, not in the model answer body."""
    cleaned = re.sub(r"(?is)\n*\**source\(s\)\**:\s*.*$", "", answer).strip()
    cleaned = re.sub(r"(?is)\n*\**sources\**:\s*.*$", "", cleaned).strip()
    return cleaned


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jenkins AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Jenkins AI Assistant")
st.caption("Powered by local RAG + Ollama · No hallucinations · No cloud")

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input("Ask anything about Jenkins..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching Jenkins docs..."):
            answer, sources = query_rag(question)

        st.markdown(answer)

        if sources and FALLBACK not in answer:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.markdown(f"- [{src}]({src})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })
