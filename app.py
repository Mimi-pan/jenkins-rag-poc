"""
app.py — Streamlit chat UI for Jenkins RAG PoC
Usage:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
INDEX_PATH             = "jenkins_index"
TOP_K                  = 8
SIMILARITY_THRESHOLD   = 3.0
FALLBACK               = "I could not find this in the Jenkins documentation."

SYSTEM_PROMPT = """You are a Jenkins documentation assistant.
Answer the user's question STRICTLY using the context provided below.

Rules:
- Do NOT add information that is not present in the context.
- If the context does not contain enough information, respond exactly with:
  "I could not find this in the Jenkins documentation."
- Always cite the source URL(s) at the end of your answer under "Source(s):".
- Be concise and factual.
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


def query_rag(question: str) -> tuple[str, list[str]]:
    """Run the full RAG pipeline. Returns (answer, sources)."""
    vectorstore = load_index()

    # Retrieve
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
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

    # Safety net
    if (not answer or len(answer) < 10
            or "i could not find" in answer.lower()
            or "not in the provided" in answer.lower()):
        return FALLBACK, []

    return answer, sources


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

        full_response = answer
        if sources and FALLBACK not in answer:
            full_response += "\n\n**Sources:** " + ", ".join(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })
