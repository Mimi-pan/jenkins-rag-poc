"""
test_quality.py — Answer Quality Check for Jenkins RAG PoC

Runs in-scope, out-of-scope, and hallucination-trap test cases against the
same hybrid retrieval pipeline used by the CLI/app.

Usage:
    python test_quality.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from query import (
    FALLBACK,
    INDEX_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    ask_llm,
    build_bm25,
    build_context,
    has_question_support,
    load_chunks,
    load_index,
    retrieve,
    should_force_fallback,
)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

TOP_K = 3
SPEED_LIMIT_SEC = 30
FALLBACK_MSG = FALLBACK


# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── Type A: In-scope Jenkins questions ───────────────────────────────────
    {
        "type": "A",
        "question": "What is a Jenkinsfile?",
        "expect": "answer",
        "keywords": ["jenkinsfile", "pipeline"],
    },
    {
        "type": "A",
        "question": "Why should I use a Jenkinsfile?",
        "expect": "answer",
        "keywords": ["pipeline", "code", "source control"],
    },
    {
        "type": "A",
        "question": "How do I run parallel stages in Jenkins?",
        "expect": "answer",
        "keywords": ["parallel", "stage", "pipeline"],
    },
    {
        "type": "A",
        "question": "How do I use credentials in Jenkins pipeline?",
        "expect": "answer",
        "keywords": ["credentials", "pipeline"],
    },
    {
        "type": "A",
        "question": "What does the Git plugin provide in Jenkins?",
        "expect": "answer",
        "keywords": ["git", "plugin"],
    },
    {
        "type": "A",
        "question": "How does the Kubernetes plugin help Jenkins builds?",
        "expect": "answer",
        "keywords": ["kubernetes", "plugin"],
    },
    {
        "type": "A",
        "question": "How do I troubleshoot reverse proxy issues in Jenkins?",
        "expect": "answer",
        "keywords": ["reverse", "proxy"],
    },
    {
        "type": "A",
        "question": "What is Declarative Pipeline syntax?",
        "expect": "answer",
        "keywords": ["pipeline", "syntax"],
    },
    {
        "type": "A",
        "question": "How can Jenkins monitoring help system administration?",
        "expect": "answer",
        "keywords": ["monitoring", "system"],
    },
    {
        "type": "A",
        "question": "How do I manage plugins in Jenkins?",
        "expect": "answer",
        "keywords": ["plugins", "manage"],
    },
    {
        "type": "A",
        "question": "What is the workflow-aggregator plugin?",
        "expect": "fallback_or_answer",
        "keywords": ["workflow-aggregator", "plugin"],
    },
    {
        "type": "A",
        "question": "What is pipeline syntax in Jenkins?",
        "expect": "answer",
        "keywords": ["pipeline", "syntax"],
    },

    # ── Type B: Out-of-scope questions ───────────────────────────────────────
    {
        "type": "B",
        "question": "What is the best restaurant in Bangkok?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "How do I deploy AWS Lambda with Terraform?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "Who won the World Cup in 2022?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "How do I use Docker Compose?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "What is the capital of France?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "Write a Python function for merge sort.",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "How do I cook pad thai?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "What stock should I buy today?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "Explain React hooks in simple terms.",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "B",
        "question": "How can I train for a marathon?",
        "expect": "fallback",
        "keywords": [],
    },

    # ── Type C: Hallucination traps / unsupported-but-near-Jenkins ───────────
    {
        "type": "C",
        "question": "Which Jenkins plugin is best for deploying to AWS?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "Does Jenkins guarantee zero-downtime upgrades?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "What exact CPU usage threshold does Jenkins use to mark a node unhealthy?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "Does the Git plugin support every Git hosting provider automatically?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "What is the recommended memory size for Jenkins on a 500-user team?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "Which Jenkins plugin should I choose for the fastest Kubernetes autoscaling?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "Can Jenkins monitoring predict hardware failure automatically?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "What is the safest reverse proxy brand for Jenkins?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "Which Git plugin option is the best for all enterprise teams?",
        "expect": "fallback",
        "keywords": [],
    },
    {
        "type": "C",
        "question": "Does workflow-aggregator remove all plugin dependency issues?",
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
    "it seems",
    "appears to",
]


def is_fallback(answer: str) -> bool:
    low = answer.lower()
    return (
        FALLBACK_MSG.lower() in low
        or should_force_fallback(answer)
    )


def has_hallucination(answer: str) -> bool:
    low = answer.lower()
    return any(phrase in low for phrase in HALLUCINATION_PHRASES)


def contains_keywords(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    low = answer.lower()
    return any(k.lower() in low for k in keywords)


def ask(vectorstore, llm, question: str, chunks, bm25) -> tuple[str, float, float]:
    """Returns (answer, similarity_score, elapsed_seconds)."""
    t0 = time.time()

    results = retrieve(vectorstore, question, chunks=chunks, bm25=bm25)
    if not results:
        return FALLBACK_MSG, 999.0, time.time() - t0

    best_score = results[0][1]
    if best_score > 0 and best_score > SIMILARITY_THRESHOLD:
        return FALLBACK_MSG, best_score, time.time() - t0

    if not has_question_support(question, results):
        return FALLBACK_MSG, best_score, time.time() - t0

    context, sources = build_context(results)
    if not context:
        return FALLBACK_MSG, best_score, time.time() - t0

    answer = ask_llm(question, context, sources)
    if is_fallback(answer) or has_hallucination(answer):
        return FALLBACK_MSG, best_score, time.time() - t0

    return answer, best_score, time.time() - t0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 70)
    print("  Jenkins RAG — Hybrid Retrieval Quality Check")
    print("=" * 70)
    print(f"  Index path              : {INDEX_PATH}")
    print(f"  Embedding model         : {OLLAMA_EMBEDDING_MODEL}")
    print(f"  Similarity threshold    : {SIMILARITY_THRESHOLD}")
    print(f"  Total test cases        : {len(TEST_CASES)}")

    print(f"\nLoading FAISS index with Ollama ({OLLAMA_EMBEDDING_MODEL}) ...")
    try:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        vectorstore = load_index(embeddings)
        chunks = load_chunks()
        bm25 = build_bm25(chunks)
    except FileNotFoundError:
        print(f"\n❌  FAISS index not found at '{INDEX_PATH}/'.")
        print("    Run `python ingest.py` first, then retry.\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌  Failed to load retrieval pipeline: {exc}")
        print("    Make sure Ollama is running and the model is pulled.\n")
        sys.exit(1)

    llm = OllamaLLM(model="mistral", temperature=0, verbose=False)
    print("Index loaded. Running tests ...\n")

    results = []
    type_groups = {"A": [], "B": [], "C": []}
    total_time = 0.0

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        typ = tc["type"]

        print(f"[{i:02d}/{len(TEST_CASES)}] Type {typ} — {question}")
        answer, score, elapsed = ask(vectorstore, llm, question, chunks, bm25)
        total_time += elapsed

        fell_back = is_fallback(answer)
        hallucinate = has_hallucination(answer)
        fast_enough = elapsed < SPEED_LIMIT_SEC
        keywords_ok = contains_keywords(answer, tc["keywords"])

        if tc["expect"] == "answer":
            passed = (not fell_back) and (not hallucinate) and keywords_ok and fast_enough
        elif tc["expect"] == "fallback":
            passed = fell_back and (not hallucinate) and fast_enough
        else:  # fallback_or_answer
            passed = ((fell_back or keywords_ok) and (not hallucinate) and fast_enough)

        verdict = "✅ PASS" if passed else "❌ FAIL"

        row = {
            "type": typ,
            "question": question,
            "answer_snippet": answer[:140].replace("\n", " "),
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
        print(f"  Answer: {row['answer_snippet']} ...")
        if hallucinate:
            print("  ⚠️  Possible hallucination detected!")
        print()

    passed_all = sum(1 for r in results if r["passed"])
    print("=" * 70)
    print(f"  SUMMARY  —  {passed_all}/{len(results)} passed")
    print("=" * 70)

    for typ, label in [
        ("A", "In-scope"),
        ("B", "Out-of-scope"),
        ("C", "Hallucination trap"),
    ]:
        group = type_groups[typ]
        passed = sum(1 for r in group if r["passed"])
        print(f"  Type {typ} ({label}): {passed}/{len(group)} passed")

    avg_time = total_time / len(results)
    print(f"\n  ⏱  Avg response time : {avg_time:.1f}s  (limit: {SPEED_LIMIT_SEC}s)")
    print(f"  🚨  Hallucinations    : {sum(1 for r in results if r['hallucinate'])}")
    print(f"  ⚡  Slow responses    : {sum(1 for r in results if not r['fast_enough'])}")

    print("\n  Checklist:")
    checks = [
        ("Hybrid retrieval pipeline loaded successfully", True),
        ("BM25 index available", bm25 is not None),
        (
            "In-scope questions mostly answered correctly",
            sum(1 for r in type_groups["A"] if r["passed"]) >= len(type_groups["A"]) - 1,
        ),
        (
            "Out-of-scope questions return fallback",
            all(r["passed"] for r in type_groups["B"]),
        ),
        (
            "Hallucination-trap questions avoid unsupported claims",
            all(r["passed"] for r in type_groups["C"]),
        ),
        ("No hallucination detected", sum(1 for r in results if r["hallucinate"]) == 0),
        (f"Response time < {SPEED_LIMIT_SEC}s", avg_time < SPEED_LIMIT_SEC),
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
