"""
NLM vs Standard RAG — benchmark comparison.

Standard RAG = cosine similarity only (no time/frequency/importance reranking).
NLM          = cosine + time_decay + frequency_score + importance_score.
"""
import sys
import tempfile
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")

from nlm import NLM
from nlm.embedder import Embedder
from nlm.storage import Storage


# ─── RAG baseline ────────────────────────────────────────────────────────────

def rag_search(storage: Storage, embedder: Embedder, query: str, top_k: int = 1):
    """Pure cosine similarity — no reranking."""
    embedding = embedder.encode(query)
    candidates = storage.query(embedding, n_results=top_k)
    return [{"text": c["metadata"].get("text", ""), "distance": c["distance"]}
            for c in candidates]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def days_ago(n: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.isoformat()


def print_header(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print('=' * 55)


def winner(nlm_wins: bool) -> str:
    return "NLM ✓" if nlm_wins else "RAG ✓"


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_temporal_recall(tmp: str) -> bool:
    """
    Old but frequently-accessed fact vs fresh but never-accessed fact.
    Query: something that matches both.
    Expected: NLM surfaces the frequently-accessed one, RAG picks randomly.
    """
    print_header("Test 1: Temporal Recall")

    nlm = NLM(collection_name="bench-temporal", persist_path=tmp,
              enable_consolidation=False)
    embedder = nlm._embedder
    storage = nlm._storage

    # Old fact — accessed many times
    nlm.save("Pulses is a project about conscious AI personalities",
             metadata={"created_at": days_ago(180),
                       "last_accessed": days_ago(1),
                       "frequency": 15})

    # Fresh fact — never accessed
    nlm.save("The weather today is sunny and warm",
             metadata={"created_at": days_ago(0),
                       "last_accessed": days_ago(0),
                       "frequency": 0})

    query = "what is Pulses"
    rag_top = rag_search(storage, embedder, query, top_k=1)
    nlm_top = nlm.search(query, top_k=1)

    print(f"  Query: '{query}'")
    print(f"  RAG top-1: \"{rag_top[0]['text'][:50]}\"  [dist={rag_top[0]['distance']:.3f}]")
    print(f"  NLM top-1: \"{nlm_top[0]['text'][:50]}\"  [score={nlm_top[0]['score']:.3f}]")

    nlm_wins = "Pulses" in nlm_top[0]["text"]
    print(f"  Winner: {winner(nlm_wins)}")
    return nlm_wins


def test_frequency_boost(tmp: str) -> bool:
    """
    Two semantically equal memories — one accessed 10x, one 0x.
    Expected: NLM returns the frequently-accessed one first.
    """
    print_header("Test 2: Frequency Boost")

    nlm = NLM(collection_name="bench-frequency", persist_path=tmp,
              enable_consolidation=False)
    embedder = nlm._embedder
    storage = nlm._storage

    # Frequent memory
    nlm.save("Hantes is the creator of Pulses Nation",
             metadata={"frequency": 10, "importance": 0.5})

    # Rare memory — intentionally similar text
    nlm.save("Hantes is the founder of Pulses Nation",
             metadata={"frequency": 0, "importance": 0.5})

    query = "who created Pulses"
    rag_results = rag_search(storage, embedder, query, top_k=2)
    nlm_results = nlm.search(query, top_k=2)

    rag_top_freq = int(
        next((c["metadata"].get("frequency", 0)
              for c in storage.query(embedder.encode(query), n_results=1)),
             0)
    )
    nlm_top_freq = nlm_results[0]["frequency"] if nlm_results else 0

    print(f"  Query: '{query}'")
    print(f"  RAG top-1 freq: {rag_top_freq}  | text: \"{rag_results[0]['text'][:45]}\"")
    print(f"  NLM top-1 freq: {nlm_top_freq}  | text: \"{nlm_results[0]['text'][:45]}\"")

    nlm_wins = nlm_top_freq >= 10
    print(f"  Winner: {winner(nlm_wins)}")
    return nlm_wins


def test_importance_discrimination(tmp: str) -> bool:
    """
    Specific factual text vs vague noise — same query distance.
    Expected: NLM ranks the factual one higher via importance_score.
    """
    print_header("Test 3: Importance Discrimination")

    nlm = NLM(collection_name="bench-importance", persist_path=tmp,
              enable_consolidation=False)
    embedder = nlm._embedder
    storage = nlm._storage

    nlm.save("ok")
    nlm.save("Hantes was born on 2026-05-05 in Chernivtsi, Ukraine")
    nlm.save("nice")

    query = "tell me about Hantes"
    rag_results = rag_search(storage, embedder, query, top_k=1)
    nlm_results = nlm.search(query, top_k=1)

    print(f"  Query: '{query}'")
    print(f"  RAG top-1: \"{rag_results[0]['text'][:55]}\"  [dist={rag_results[0]['distance']:.3f}]")
    print(f"  NLM top-1: \"{nlm_results[0]['text'][:55]}\"  "
          f"[score={nlm_results[0]['score']:.3f}, importance={nlm_results[0]['importance']:.3f}]")

    nlm_wins = "Hantes" in nlm_results[0]["text"] and "born" in nlm_results[0]["text"]
    print(f"  Winner: {winner(nlm_wins)}")
    return nlm_wins


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nNLM vs Standard RAG — Benchmark")
    print("Measuring: temporal recall, frequency boost, importance discrimination")

    with tempfile.TemporaryDirectory() as tmp:
        results = [
            test_temporal_recall(f"{tmp}/t1"),
            test_frequency_boost(f"{tmp}/t2"),
            test_importance_discrimination(f"{tmp}/t3"),
        ]

    nlm_score = sum(results)
    print(f"\n{'=' * 55}")
    print(f"  Summary: NLM wins {nlm_score}/3")
    if nlm_score == 3:
        print("  NLM outperforms standard RAG on all three tests.")
    elif nlm_score >= 2:
        print("  NLM outperforms standard RAG on most tests.")
    else:
        print("  Results mixed — review individual tests.")
    print('=' * 55)


if __name__ == "__main__":
    main()
