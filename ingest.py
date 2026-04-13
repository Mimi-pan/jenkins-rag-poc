"""
ingest.py — Crawl Jenkins docs, chunk, embed, and store in a FAISS index.
"""

import os
import re
import pickle
from urllib.parse import urlparse
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

SOURCES = [
    {"url": "https://www.jenkins.io/doc/book/pipeline/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/pipeline/jenkinsfile/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/pipeline/syntax/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/using/using-credentials/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/system-administration/monitoring/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/troubleshooting/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/managing/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/managing/plugins/", "source_type": "jenkins_docs"},
    {"url": "https://www.jenkins.io/doc/book/system-administration/reverse-proxy-configuration-troubleshooting/", "source_type": "jenkins_docs"},
    {"url": "https://plugins.jenkins.io/workflow-aggregator/", "source_type": "jenkins_plugin"},
    {"url": "https://plugins.jenkins.io/git/", "source_type": "jenkins_plugin"},
    {"url": "https://plugins.jenkins.io/kubernetes/", "source_type": "jenkins_plugin"},
]

INDEX_PATH = "jenkins_index"

# ── Helpers ──────────────────────────────────────────────────────────────────

def derive_title(soup: BeautifulSoup) -> str:
    """Best-effort page title extraction."""
    title_tag = soup.find("h1") or soup.find("title")
    if not title_tag:
        return ""
    return re.sub(r"\s+", " ", title_tag.get_text(" ", strip=True)).strip()


def build_page_metadata(url: str, source_type: str, soup: BeautifulSoup) -> dict:
    """Build stable metadata for a fetched page."""
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    metadata = {
        "source": url,
        "source_type": source_type,
        "domain": parsed.netloc,
        "title": derive_title(soup),
    }

    if source_type == "jenkins_plugin":
        plugin_id = path_parts[0] if path_parts else ""
        metadata["plugin_id"] = plugin_id
        metadata["plugin_name"] = metadata["title"] or plugin_id.replace("-", " ").title()

    return metadata


def extract_docs_text(soup: BeautifulSoup) -> str:
    """Extract visible text from a jenkins.io docs page."""
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_plugin_text(soup: BeautifulSoup) -> str:
    """Extract main plugin-page content while trimming site chrome."""
    for tag in soup(["nav", "footer", "script", "style", "header", "noscript"]):
        tag.decompose()

    # Plugin pages are not identical to jenkins.io docs pages, so prefer
    # likely content containers before falling back to the whole body.
    selectors = [
        "main",
        "article",
        '[role="main"]',
        ".plugin-content",
        ".container .row",
    ]
    main = None
    for selector in selectors:
        main = soup.select_one(selector)
        if main:
            break
    if main is None:
        main = soup.body

    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_page(url: str, source_type: str) -> dict:
    """Download a page and return normalized text plus metadata."""
    headers = {"User-Agent": "Jenkins-RAG-PoC/1.0 (GSoC 2026)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    metadata = build_page_metadata(url, source_type, soup)
    if source_type == "jenkins_plugin":
        text = extract_plugin_text(soup)
    else:
        text = extract_docs_text(soup)

    return {
        "url": url,
        "text": text,
        "metadata": metadata,
    }


def crawl_pages(sources: list[dict]) -> list[dict]:
    """Return normalized page records with text and metadata."""
    
    pages = []

    for source in sources:
        url = source["url"]
        source_type = source.get("source_type", "jenkins_docs")

        print("INDEXING:", url)

        try:
            page = fetch_page(url, source_type)
            pages.append(page)
            print(f" -> {len(page['text'])} characters")

        except Exception as exc:
            print(f" X Failed ({exc})")

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
            chunk = {
                "url": page["url"],
                "text": part,
                "metadata": dict(page.get("metadata", {})),
            }
            chunks.append(chunk)
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Jenkins RAG — Ingestion Pipeline ===\n")

    # 1. Crawl
    print("[1/4] Crawling Jenkins documentation pages …")
    pages = crawl_pages(SOURCES)
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

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

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
