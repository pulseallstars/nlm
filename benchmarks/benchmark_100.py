"""
NLM vs RAG — formal benchmark on 100 memories.

Methodology:
- 100 memories with varying age, frequency, and importance
- 30 queries across 3 categories
- Ground truth: human-labeled correct answer for each query
- Metric: top-1 accuracy (did the right memory rank first?)

Categories:
  - Temporal (10 queries): recent vs old facts, correct = most recent relevant
  - Frequency (10 queries): same-topic memories, correct = most-accessed
  - Importance (10 queries): factual vs vague memories, correct = factual one
"""

import sys
import tempfile
import random
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")

from nlm import NLM
from nlm.embedder import Embedder
from nlm.storage import Storage
from nlm.scoring import rerank as nlm_rerank


# ─── RAG baseline ────────────────────────────────────────────────────────────

def rag_search(storage: Storage, embedder: Embedder, query: str) -> str:
    """Pure cosine similarity — no reranking. Returns top-1 text."""
    embedding = embedder.encode(query)
    candidates = storage.query(embedding, n_results=1)
    return candidates[0]["metadata"].get("text", "") if candidates else ""


# ─── Dataset ─────────────────────────────────────────────────────────────────

def days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def build_dataset():
    """
    Returns list of (text, metadata_overrides) and list of (query, correct_text_fragment).
    """
    memories = []
    queries = []

    # ── Category 1: Temporal (10 queries) ────────────────────────────────────
    # Each query has an OLD and NEW version. Correct = NEW (more recent).
    temporal_pairs = [
        # (old_text, new_text, neutral_query, correct_fragment)
        ("Hantes uses PyTorch for training",          "Hantes switched to JAX for training",         "what ML framework is Hantes using for training",    "JAX"),
        ("Project is on version 0.1",                 "Project is on version 1.0",                   "what is the current project version number",         "1.0"),
        ("The server runs on port 8000",              "The server was moved to port 8001",            "which port does the server listen on",               "8001"),
        ("Pulse model is Mistral 7B",                 "Pulse model was upgraded to RWKV-7",           "what AI model does Pulse use",                       "RWKV"),
        ("Team has 2 members",                        "Team now has 4 members",                       "how many people are on the team",                    "4 members"),
        ("Training loss is 0.85",                     "Training loss improved to 0.57",               "what is the current training loss value",            "0.57"),
        ("Database is SQLite",                        "Database migrated to PostgreSQL",              "what database system is being used",                 "PostgreSQL"),
        ("UI runs on React",                          "UI was rewritten in vanilla JS",               "what technology is the user interface built with",   "vanilla"),
        ("Model has 1B parameters",                   "Model was scaled up to 7B parameters",        "how many parameters does the model have",            "7B"),
        ("Auth uses JWT tokens",                      "Auth switched to session-based approach",      "how does authentication work in the system",         "session"),
    ]
    for i, (old_text, new_text, query_text, correct_fragment) in enumerate(temporal_pairs):
        old_meta = {"created_at": days_ago(180), "last_accessed": days_ago(180), "frequency": 0}
        new_meta = {"created_at": days_ago(1),   "last_accessed": days_ago(1),   "frequency": 0}
        memories.append((old_text, old_meta))
        memories.append((new_text, new_meta))
        queries.append(("temporal", query_text, correct_fragment))

    # ── Category 2: Frequency (10 queries) ───────────────────────────────────
    # Two semantically similar facts, one accessed 15x, one 0x. Correct = high-freq.
    freq_pairs = [
        ("Hantes is the creator of Pulses",           "Hantes is the founder of Pulses",           "creator"),
        ("NLM stands for Neural Long Memory",         "NLM means Neural Long Memory system",       "stands for"),
        ("Pulses has 8 families of AI",               "Pulses Nation contains 8 AI families",      "has 8 families"),
        ("RWKV uses linear attention",                "RWKV model employs linear attention",        "uses linear"),
        ("ChromaDB stores embeddings on disk",        "ChromaDB persists vector data locally",      "on disk"),
        ("The venv is in .venv folder",               "Virtual environment lives in .venv",         "in .venv folder"),
        ("Training uses SFTTrainer",                  "Fine-tuning is done with SFTTrainer",        "uses SFTTrainer"),
        ("Emotion model is distilroberta",            "Emotion classification uses distilroberta",  "is distilroberta"),
        ("Consolidation threshold is 0.15",           "Merge threshold set to 0.15",                "is 0.15"),
        ("Memory half-life is 90 days",               "Decay half-life equals 90 days",             "is 90 days"),
    ]
    for high_text, low_text, correct_fragment in freq_pairs:
        high_meta = {"frequency": 15, "importance": 0.5}
        low_meta  = {"frequency": 0,  "importance": 0.5}
        memories.append((high_text, high_meta))
        memories.append((low_text,  low_meta))
        queries.append(("frequency", f"tell me about {high_text.split()[0]} {high_text.split()[1]}", correct_fragment))

    # ── Category 3: Importance (10 queries) ──────────────────────────────────
    # Factual vs vague memory on same topic. Correct = factual (specific).
    importance_pairs = [
        ("Vitalii Halak was born on 1998-03-15 in Chernivtsi, Ukraine",  "someone was born somewhere sometime",      "1998"),
        ("NLM version 1.0.0 released on 2026-04-15",                     "NLM got a new release at some point",      "1.0.0"),
        ("RWKV-7 model has 2.9 billion parameters",                      "RWKV is a large language model",           "2.9 billion"),
        ("ChromaDB collection uses cosine distance metric",               "the database stores things somehow",       "cosine"),
        ("sentence-transformers all-MiniLM-L6-v2 outputs 384 dimensions","a model outputs some embeddings",          "384"),
        ("PyPI package name is nlm, free to register",                   "the package has a name on PyPI",           "is nlm"),
        ("Pulses project started on 2026-05-05 in Chernivtsi",           "Pulses project started some time ago",     "2026-05-05"),
        ("GPU scorer uses typeform/distilbert-base-uncased-mnli model",   "there is an optional GPU mode",            "distilbert"),
        ("association_threshold default value is 0.55 cosine distance",  "associations have a threshold setting",    "0.55"),
        ("forget_smart deletes if days>180 AND freq<2 AND importance<0.3","forget_smart has multiple conditions",     "AND freq<2"),
    ]
    for factual_text, vague_text, correct_fragment in importance_pairs:
        memories.append((factual_text, {"importance": 0.8}))
        memories.append((vague_text,   {"importance": 0.1}))
        queries.append(("importance", f"what specifically about {factual_text.split()[0]} {factual_text.split()[1]}", correct_fragment))

    # ── Filler: 40 unrelated background memories ─────────────────────────────
    fillers = [
        "The cat sat on the mat yesterday morning",
        "Coffee is best enjoyed at room temperature",
        "Photosynthesis converts sunlight into glucose",
        "The Eiffel Tower is located in Paris France",
        "Water boils at 100 degrees Celsius at sea level",
        "Shakespeare wrote Hamlet in the early 1600s",
        "Mount Everest is 8849 meters tall",
        "The speed of light is 299792458 meters per second",
        "DNA stands for deoxyribonucleic acid",
        "The capital of Australia is Canberra not Sydney",
        "Bees communicate through a waggle dance",
        "The Great Wall of China is not visible from space",
        "Octopuses have three hearts",
        "Bananas are technically berries botanically",
        "The human brain has approximately 86 billion neurons",
        "Pluto was reclassified as a dwarf planet in 2006",
        "The Amazon river flows into the Atlantic Ocean",
        "Penguins live only in the Southern Hemisphere naturally",
        "Glass is made primarily from silicon dioxide sand",
        "The moon is slowly drifting away from Earth",
        "Honey never spoils if stored properly sealed",
        "Tigers are the largest wild cat species",
        "The Sahara desert expands and contracts seasonally",
        "Oxygen was discovered by Carl Wilhelm Scheele",
        "The first computer bug was an actual moth in 1947",
        "Linux was created by Linus Torvalds in 1991",
        "Python programming language was named after Monty Python",
        "The internet was originally called ARPANET",
        "Bitcoin was created by Satoshi Nakamoto in 2008",
        "The first iPhone was released in June 2007",
        "Netflix started as a DVD rental service by mail",
        "Amazon was originally an online bookstore",
        "Google was founded in a garage in Menlo Park",
        "Wikipedia launched on January 15 2001",
        "The first email was sent in 1971 by Ray Tomlinson",
        "USB was introduced in 1996 as a universal standard",
        "Wi-Fi stands for Wireless Fidelity",
        "The term bug in software comes from Grace Hopper",
        "Open source software can be freely modified and shared",
        "Version control with git was created by Linus Torvalds",
    ]
    for filler in fillers:
        memories.append((filler, {}))

    return memories, queries


# ─── Run benchmark ───────────────────────────────────────────────────────────

def run_benchmark():
    random.seed(42)
    memories, queries = build_dataset()

    print(f"\nNLM vs RAG — Formal Benchmark")
    print(f"Memories: {len(memories)} | Queries: {len(queries)}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        nlm = NLM(
            collection_name="benchmark",
            persist_path=tmp,
            enable_consolidation=False,
            enable_associations=False,
        )

        # Save all memories
        for text, meta in memories:
            nlm.save(text, metadata=meta if meta else None)

        embedder = nlm._embedder
        storage  = nlm._storage

        # Run queries
        results_by_category = {"temporal": [], "frequency": [], "importance": []}

        for category, query, correct_fragment in queries:
            # RAG: cosine only
            rag_top = rag_search(storage, embedder, query)
            rag_correct = correct_fragment.lower() in rag_top.lower()

            # NLM: full scoring
            nlm_results = nlm.search(query, top_k=1)
            nlm_top = nlm_results[0]["text"] if nlm_results else ""
            nlm_correct = correct_fragment.lower() in nlm_top.lower()

            results_by_category[category].append((rag_correct, nlm_correct))

        # ── Print results ─────────────────────────────────────────────────────
        total_rag = total_nlm = total = 0

        for category, results in results_by_category.items():
            n = len(results)
            rag_score = sum(r for r, _ in results)
            nlm_score = sum(n for _, n in results)
            total += n
            total_rag += rag_score
            total_nlm += nlm_score

            rag_pct = rag_score / n * 100
            nlm_pct = nlm_score / n * 100
            delta = nlm_pct - rag_pct

            print(f"\n  {category.upper()} ({n} queries)")
            print(f"    RAG: {rag_score}/{n}  ({rag_pct:.0f}%)")
            print(f"    NLM: {nlm_score}/{n}  ({nlm_pct:.0f}%)")
            sign = "+" if delta >= 0 else ""
            print(f"    Delta: {sign}{delta:.0f}%")

        # ── Summary ───────────────────────────────────────────────────────────
        rag_total_pct = total_rag / total * 100
        nlm_total_pct = total_nlm / total * 100
        overall_delta = nlm_total_pct - rag_total_pct

        print(f"\n{'=' * 60}")
        print(f"  OVERALL ({total} queries)")
        print(f"  RAG accuracy: {total_rag}/{total}  ({rag_total_pct:.0f}%)")
        print(f"  NLM accuracy: {total_nlm}/{total}  ({nlm_total_pct:.0f}%)")
        print(f"  NLM improvement: +{overall_delta:.0f}% over RAG")
        print("=" * 60)

        return {
            "rag_pct": rag_total_pct,
            "nlm_pct": nlm_total_pct,
            "delta": overall_delta,
            "by_category": {
                cat: {
                    "rag": sum(r for r, _ in res) / len(res) * 100,
                    "nlm": sum(n for _, n in res) / len(res) * 100,
                }
                for cat, res in results_by_category.items()
            }
        }


if __name__ == "__main__":
    run_benchmark()
