"""
ingest.py — Crawl Jenkins docs, chunk, embed, and store in a FAISS index.
"""

import os
import re
import pickle
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

URLS = [
    "https://www.jenkins.io/doc/book/pipeline/",
    "https://www.jenkins.io/doc/book/pipeline/jenkinsfile/",
    "https://www.jenkins.io/doc/book/pipeline/syntax/",
    "https://www.jenkins.io/doc/book/using/using-credentials/",
    "https://www.jenkins.io/doc/book/system-administration/monitoring/",
    "https://www.jenkins.io/doc/book/troubleshooting/",
    "https://www.jenkins.io/doc/book/managing/",
    "https://www.jenkins.io/doc/book/managing/plugins/",
    "https://www.jenkins.io/doc/book/system-administration/reverse-proxy-configuration-troubleshooting/",
]

INDEX_PATH = "jenkins_index"

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_text(url: str) -> str:
    """Download a page and return clean visible text."""
    headers = {"User-Agent": "Jenkins-RAG-PoC/1.0 (GSoC 2026)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation / footer noise
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    # Prefer the main content block
    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def crawl_pages(urls: list[str]) -> list[dict]:
    """Return list of {url, text} dicts."""
    pages = []
    for url in urls:
        print(f"  Fetching: {url}")
        try:
            text = fetch_text(url)
            pages.append({"url": url, "text": text})
            print(f"    → {len(text):,} characters")
        except Exception as exc:
            print(f"    ✗ Failed ({exc})")
    return pages


def split_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Split each page into ~500-token chunks with 50-token overlap.
    We approximate 1 token ≈ 4 characters (OpenAI rule-of-thumb).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500 * 4,       # ≈ 500 tokens
        chunk_overlap=50 * 4,     # ≈ 50 tokens overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        parts = splitter.split_text(page["text"])
        for part in parts:
            chunks.append({"url": page["url"], "text": part})
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Jenkins RAG — Ingestion Pipeline ===\n")

    # 1. Crawl
    print("[1/4] Crawling Jenkins documentation pages …")
    pages = crawl_pages(URLS)
    print(f"      Total pages fetched: {len(pages)}\n")

    # 2. Chunk
    print("[2/4] Splitting text into chunks …")
    chunks = split_into_chunks(pages)
    print(f"      Total chunks: {len(chunks)}\n")

    # 3. Embed
    print(f"[3/4] Creating embeddings with Ollama ({OLLAMA_EMBEDDING_MODEL}) …")
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    texts    = [c["text"] for c in chunks]
    metadatas = [{"source": c["url"]} for c in chunks]

    # 4. Build & save FAISS index
    print("[4/4] Building FAISS index and saving to disk …")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(INDEX_PATH)
    print(f"      Index saved → '{INDEX_PATH}/'")

    # Also persist raw chunks for debugging / inspection
    with open(f"{INDEX_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("\n✅ Ingestion complete!")
    print(f"   {len(chunks)} chunks across {len(pages)} pages indexed.")


if __name__ == "__main__":
    main()
