# Jenkins Documentation RAG — Proof of Concept

> **GSoC 2026 PoC** for the Jenkins AI Chatbot Plugin
> Related PR: [jenkinsci/resources-ai-chatbot-plugin#318](https://github.com/jenkinsci/resources-ai-chatbot-plugin/pull/318)

A fully local Retrieval-Augmented Generation (RAG) system that answers questions about Jenkins by grounding every response in the official Jenkins documentation. It **never hallucinates** — if the answer isn't in the indexed docs, it says so.

Runs 100% locally using [Ollama](https://ollama.com) — no cloud API or API key required.

---

## Architecture

```
Jenkins Docs (9 pages)
        │
        ▼
   ingest.py
  ┌──────────────────────────────────────┐
  │ 1. Crawl pages (BeautifulSoup)       │
  │ 2. Chunk text (500 tok / 50 overlap) │
  │ 3. Embed (Ollama nomic-embed-text)   │
  │ 4. Store → FAISS index on disk       │
  └──────────────────────────────────────┘
        │
        ▼  (jenkins_index/)
   query.py
  ┌──────────────────────────────────────┐
  │ 1. Load FAISS index                  │
  │ 2. Embed question (Ollama)           │
  │ 3. Retrieve top-3 chunks             │
  │ 4. Similarity threshold check        │
  │ 5. Mistral → grounded answer         │
  └──────────────────────────────────────┘
```

---

## Indexed Documentation Pages

| Page | URL |
|---|---|
| Pipeline Overview | https://www.jenkins.io/doc/book/pipeline/ |
| Jenkinsfile | https://www.jenkins.io/doc/book/pipeline/jenkinsfile/ |
| Pipeline Syntax | https://www.jenkins.io/doc/book/pipeline/syntax/ |
| Using Credentials | https://www.jenkins.io/doc/book/using/using-credentials/ |
| Monitoring | https://www.jenkins.io/doc/book/system-administration/monitoring/ |
| Troubleshooting | https://www.jenkins.io/doc/book/troubleshooting/ |
| Managing Jenkins | https://www.jenkins.io/doc/book/managing/ |
| Managing Plugins | https://www.jenkins.io/doc/book/managing/plugins/ |
| Reverse Proxy Troubleshooting | https://www.jenkins.io/doc/book/system-administration/reverse-proxy-configuration-troubleshooting/ |

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Required Ollama models pulled:

```bash
ollama pull nomic-embed-text   # embedding model (~270 MB)
ollama pull mistral             # LLM for answering (~4.4 GB)
```

---

## Installation

```bash
# 1. Clone the project
git clone https://github.com/jenkinsci/resources-ai-chatbot-plugin.git
cd resources-ai-chatbot-plugin

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (optional — defaults work out of the box)
cp .env.example .env
# Edit .env only if Ollama runs on a non-default URL or you want a different model
```

---

## Running

### Step 1 — Ingest (run once, or whenever docs change)

```bash
python ingest.py
```

Expected output:
```
=== Jenkins RAG — Ingestion Pipeline ===

[1/4] Crawling Jenkins documentation pages …
  Fetching: https://www.jenkins.io/doc/book/pipeline/
    → 45,210 characters
  ...
      Total pages fetched: 9

[2/4] Splitting text into chunks …
      Total chunks: 312

[3/4] Creating embeddings with Ollama (nomic-embed-text) …
[4/4] Building FAISS index and saving to disk …
      Index saved → 'jenkins_index/'

✅ Ingestion complete!
   312 chunks across 9 pages indexed.
```

### Step 2 — Query

```bash
# Windows
set KMP_DUPLICATE_LIB_OK=TRUE
python query.py "<your question here>"

# macOS / Linux
python query.py "<your question here>"
```

### Step 3 — Quality Check (optional)

Run the full automated test suite (7 questions, 3 types):

```bash
python test_quality.py
```

---

## Example Queries

### In-scope — Jenkinsfile basics

```bash
python query.py "What is a Jenkinsfile and why should I use it?"
```

![Query demo](demo_query.png)

### In-scope — Parallel stages

```bash
python query.py "How to run parallel stages in Jenkins?"
```

### Out-of-scope — Fallback demonstration

```bash
python query.py "What is the best restaurant in Bangkok?"
```

```
🔍 Question: What is the best restaurant in Bangkok?

   Best similarity distance: 4.9201

💬 Answer:
I could not find this in the Jenkins documentation.
```

---

## Quality Test Results

Running `python test_quality.py` produces a full pass across all 7 test cases:

![Test results](demo_tests.png)

---

## Project Structure

```
jenkins-rag-poc/
├── ingest.py           # Crawl → chunk → embed → FAISS index
├── query.py            # Load index → retrieve → Mistral → answer
├── test_quality.py     # Automated quality check (Type A / B / C)
├── requirements.txt    # Python dependencies
├── .env.example        # Config template (copy to .env)
├── .env                # Your local config (git-ignored)
└── jenkins_index/      # Generated FAISS index (git-ignored)
    ├── index.faiss
    ├── index.pkl
    └── chunks.pkl
```

---

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |

> The LLM model (`mistral`) is currently hardcoded in `query.py` and `test_quality.py`.

---

## How Hallucination Is Prevented

Three layers of protection:

1. **Similarity threshold** — if the best L2 distance > `3.0`, the system returns a fallback message without calling the LLM at all.
2. **Strict system prompt** — the LLM is instructed to answer *only* from the provided context and to say "I could not find this" if the context is insufficient.
3. **Safety net** — the LLM's response is checked for fallback phrases (`"i could not find"`, `"not in the provided"`) and replaced with a standard message if matched.

---

## Security Notes

- `.env` is **never** committed to version control.
- No API keys or cloud credentials required — all inference runs locally via Ollama.
- The system prompt prevents the LLM from using knowledge outside the retrieved context.

---

## Related Resources

- [jenkinsci/resources-ai-chatbot-plugin PR #318](https://github.com/jenkinsci/resources-ai-chatbot-plugin/pull/318)
- [Jenkins GSoC 2026 — AI Chatbot to Guide User Workflow](https://www.jenkins.io/projects/gsoc/2026/project-ideas/ai-chatbot-to-guide-user-workflow/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Ollama](https://ollama.com)
- [LangChain FAISS integration](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
