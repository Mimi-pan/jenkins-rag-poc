"""
app.py — Streamlit chat UI for Jenkins RAG PoC
Usage:
    streamlit run app.py
"""

import streamlit as st

from rag_core import FALLBACK, load_pipeline, query_rag


@st.cache_resource
def load_pipeline_cached():
    return load_pipeline()


st.set_page_config(
    page_title="Jenkins AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Jenkins AI Assistant")
st.caption("Powered by local RAG + Ollama · No hallucinations · No cloud")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask anything about Jenkins..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching Jenkins docs..."):
            vectorstore, chunks, bm25 = load_pipeline_cached()
            answer, sources = query_rag(
                question,
                vectorstore=vectorstore,
                chunks=chunks,
                bm25=bm25,
            )

        st.markdown(answer)

        if sources and FALLBACK not in answer:
            with st.expander("📚 Sources"):
                for src in sources:
                    st.markdown(f"- [{src}]({src})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })
