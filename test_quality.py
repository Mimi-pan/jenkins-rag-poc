"""
test_quality.py — Answer Quality Check for Jenkins RAG PoC

Runs Type A / B / C test cases and prints a pass/fail summary.
Checks: retrieval relevance, fallback behaviour, hallucination guard, speed.

Usage:
    python test_quality.py
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
INDEX_PATH             = "jenkins_index"
TOP_K                  = 3
SIMILARITY_THRESHOLD   = 3.0
SPEED_LIMIT_SEC        = 30
FALLBACK_MSG           = "I could not find this in the Jenkins documentation."

SYSTEM_PROMPT = """You are a Jenkins documentation assistant.
Answer the user's question STRICTLY using the context provided below.

Rules:
- Do NOT add information that is not present in the context.
- If the context does not contain enough information, respond exactly with:
  "I could not find this in the Jenkins documentation."
- Always cite the source URL(s) at the end of your answer under "Source(s):".
- Be concise and factual.
"""

# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Type A: In-scope (must answer correctly) ──────────────────────────────
    {
        "type": "A",
        "question": "What is a Jenkinsfile?",
        "expect": "in_scope",
        "keywords": ["jenkinsfile", "pipeline", "groovy"],
    },
    {
        "type": "A",
        "question": "How to run parallel stages in Jenkins?",
        "expect": "in_scope",
        "keywords": ["parallel", "stage", "pipeline"],
    },
    {
        "type": "A",
        "question": "How to trigger a Jenkins build automatically?",
        "expect": "in_scope",
        "keywords": ["trigger", "webhook", "scm", "cron", "automatically"],
    },
    # ── Type B: Edge case (borderline — should hint even without an exact answer) ──────
    {
        "type": "B",
        "question": "My build is slow, what should I do?",
        "expect": "hint",          # accepts either a hint or fallback
        "keywords": [],
    },
    {
        "type": "B",
        "question": "Jenkins keeps crashing",
        "expect": "hint",
        "keywords": [],
    },
    # ── Type C: Out-of-scope (not in docs — must return fallback) ─────────────
    {
        "type": "C",
        "question": "What is the best restaurant in Bangkok?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "How to use Docker Compose?",
        "expect": "fallback",
        "keywords": [],
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

HALLUCINATION_PHRASES = [
    "as of my knowledge",
    "as an ai",
    "i don't have access",
    "i'm not able to browse",
    "according to my training",
    "i cannot access the internet",
]


def is_fallback(answer: str) -> bool:
    return (
        FALLBACK_MSG.lower() in answer.lower()
        or "i could not find" in answer.lower()
        or "not in the provided" in answer.lower()
        or "no mention" in answer.lower()
        or len(answer.strip()) < 20
    )


def has_hallucination(answer: str) -> bool:
    low = answer.lower()
    return any(phrase in low for phrase in HALLUCINATION_PHRASES)


def contains_keywords(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    low = answer.lower()
    return any(k.lower() in low for k in keywords)


def ask(vectorstore, llm, question: str) -> tuple[str, float, float]:
    """Returns (answer, similarity_score, elapsed_seconds)"""
    t0 = time.time()

    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    if not results:
        return FALLBACK_MSG, 999.0, time.time() - t0

    best_score = results[0][1]
    if best_score > SIMILARITY_THRESHOLD:
        return FALLBACK_MSG, best_score, time.time() - t0

    context_parts, sources = [], []
    for doc, _ in results:
        if doc.page_content:
            context_parts.append(doc.page_content.strip())
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts)
    source_text = ", ".join(sources)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Source URLs available:
{source_text}
"""
    response = llm.invoke(prompt)
    answer = response.strip() if isinstance(response, str) else str(response).strip()
    return answer, best_score, time.time() - t0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 60)
    print("  Jenkins RAG — Answer Quality Check")
    print("=" * 60)

    # Load embeddings + index
    print(f"\nLoading FAISS index with Ollama ({OLLAMA_EMBEDDING_MODEL}) …")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except FileNotFoundError:
        print(f"\n❌  FAISS index not found at '{INDEX_PATH}/'.")
        print("    Run `python ingest.py` first, then retry.\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌  Failed to load index: {exc}")
        print("    Make sure Ollama is running and the model is pulled.\n")
        sys.exit(1)

    llm = OllamaLLM(model="mistral", temperature=0, verbose=False)
    print("Index loaded. Running tests …\n")

    results = []
    type_groups = {"A": [], "B": [], "C": []}
    total_time = 0.0

    for i, tc in enumerate(TEST_CASES, 1):
        q   = tc["question"]
        typ = tc["type"]

        print(f"[{i}/{len(TEST_CASES)}] Type {typ} — {q}")
        answer, score, elapsed = ask(vectorstore, llm, q)
        total_time += elapsed

        # ── Evaluate ──────────────────────────────────────────────────────
        fell_back   = is_fallback(answer)
        hallucinate = has_hallucination(answer)
        fast_enough = elapsed < SPEED_LIMIT_SEC
        keywords_ok = contains_keywords(answer, tc["keywords"])

        if typ == "A":
            passed = (not fell_back) and (not hallucinate) and keywords_ok and fast_enough
            verdict = "✅ PASS" if passed else "❌ FAIL"
        elif typ == "B":
            # Edge case: accepts either hint or fallback, but must not hallucinate
            passed = (not hallucinate) and fast_enough
            verdict = "✅ PASS" if passed else "❌ FAIL"
        else:  # C
            passed = fell_back and (not hallucinate) and fast_enough
            verdict = "✅ PASS" if passed else "❌ FAIL"

        row = {
            "type": typ,
            "question": q,
            "answer_snippet": answer[:120].replace("\n", " "),
            "score": score,
            "elapsed": elapsed,
            "fell_back": fell_back,
            "hallucinate": hallucinate,
            "fast_enough": fast_enough,
            "keywords_ok": keywords_ok,
            "passed": passed,
            "verdict": verdict,
        }
        results.append(row)
        type_groups[typ].append(row)

        print(f"  {verdict}  |  score={score:.3f}  |  {elapsed:.1f}s")
        print(f"  Answer: {row['answer_snippet']} …")
        if hallucinate:
            print("  ⚠️  Possible hallucination detected!")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    passed_all = sum(1 for r in results if r["passed"])
    print("=" * 60)
    print(f"  SUMMARY  —  {passed_all}/{len(results)} passed")
    print("=" * 60)

    for typ, label in [("A", "In-scope"), ("B", "Edge case"), ("C", "Out-of-scope")]:
        group = type_groups[typ]
        p = sum(1 for r in group if r["passed"])
        print(f"  Type {typ} ({label}): {p}/{len(group)} passed")

    avg_time = total_time / len(results)
    print(f"\n  ⏱  Avg response time : {avg_time:.1f}s  (limit: {SPEED_LIMIT_SEC}s)")
    print(f"  🚨  Hallucinations    : {sum(1 for r in results if r['hallucinate'])}")
    print(f"  ⚡  Slow responses    : {sum(1 for r in results if not r['fast_enough'])}")

    # ── Checklist ─────────────────────────────────────────────────────────
    print("\n  Checklist:")
    checks = [
        ("FAISS index loaded successfully",         True),
        ("Retrieval returns documents",              all(r["score"] < 999 for r in results if r["type"] != "C")),
        ("In-scope questions answered correctly",   all(r["passed"] for r in type_groups["A"])),
        ("Out-of-scope questions return fallback",  all(r["passed"] for r in type_groups["C"])),
        ("No hallucination detected",               sum(1 for r in results if r["hallucinate"]) == 0),
        (f"Response time < {SPEED_LIMIT_SEC}s",     avg_time < SPEED_LIMIT_SEC),
    ]
    for label, ok in checks:
        icon = "☑" if ok else "☐"
        print(f"    {icon}  {label}")

    print()
    if passed_all == len(results):
        print("  🎉  All tests passed!")
    else:
        print(f"  ⚠️   {len(results) - passed_all} test(s) failed — review answers above.")
    print()


if __name__ == "__main__":
    run_tests()
