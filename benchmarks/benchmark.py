"""
NLM benchmark runner.

Loads a fixed dataset and runs every query against a fresh store in one of two
retrieval modes, reporting top-1 / MRR / recall@k / latency per category.

  rag  - cosine similarity only (semantic baseline)
  nlm  - semantic + time + frequency + importance hybrid scoring

Both modes share the exact same store and candidate pool so the only variable
is the ranker.

Examples:
    python benchmarks/benchmark.py --mode rag --save benchmarks/results/v1.1.0_rag.json
    python benchmarks/benchmark.py --mode nlm --save benchmarks/results/v1.1.0_nlm.json
    python benchmarks/benchmark.py --mode nlm \\
        --semantic 0.5 --time 0.2 --freq 0.15 --importance 0.15 \\
        --save benchmarks/results/tuned_w05.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlm.embedder import Embedder
from nlm.storage import Storage
from nlm.scoring import rerank
from nlm.utils import specificity_score


# ─── dataset → store ────────────────────────────────────────────────────────

def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def materialize_memory(mem, now):
    """Convert a dataset entry into the metadata dict the store expects.

    `created_days_ago` is anchored to `now`, and `importance` is filled in
    with the same CPU heuristic NLM uses by default — this keeps the
    benchmark reproducible without baking importance into the dataset.
    """
    created = (now - timedelta(days=int(mem["created_days_ago"]))).isoformat()
    return {
        "text": mem["text"],
        "created_at": created,
        "last_accessed": created,
        "frequency": int(mem.get("frequency", 0)),
        "importance": specificity_score(mem["text"]),
    }


def build_store(dataset, persist_path, embedder):
    store = Storage(
        collection_name="nlm_bench",
        persist_path=persist_path,
        embedding_dim=embedder.dim,
        embedding_model=embedder.model_name,
    )
    store.clear()

    now = datetime.now(timezone.utc)
    ids = [m["id"] for m in dataset["memories"]]
    texts = [m["text"] for m in dataset["memories"]]
    metas = [materialize_memory(m, now) for m in dataset["memories"]]
    embeddings = embedder.encode_many(texts)
    store.add_many(ids, embeddings, metas)
    return store


# ─── retrieval modes ────────────────────────────────────────────────────────

def search_rag(store, embedder, query, top_k, n_candidates):
    emb = embedder.encode(query)
    cands = store.query(emb, n_results=n_candidates)
    out = [
        {
            "id": c["id"],
            "text": c["metadata"].get("text", ""),
            "score": max(0.0, 1.0 - c["distance"] / 2.0),
        }
        for c in cands
    ]
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_k]


def search_nlm(store, embedder, query, top_k, n_candidates, weights, half_life):
    emb = embedder.encode(query)
    cands = store.query(emb, n_results=n_candidates)
    ranked = rerank(cands, weights, half_life)
    return ranked[:top_k]


# ─── evaluation ─────────────────────────────────────────────────────────────

def find_rank(results, gt_id):
    for i, r in enumerate(results):
        if r["id"] == gt_id:
            return i + 1
    return None


def run(dataset, mode, weights, half_life, top_k, n_candidates, persist_path):
    embedder = Embedder()
    store = build_store(dataset, persist_path, embedder)

    per_query = []
    for q in dataset["queries"]:
        t0 = time.perf_counter()
        if mode == "rag":
            results = search_rag(store, embedder, q["text"], top_k, n_candidates)
        else:
            results = search_nlm(
                store, embedder, q["text"], top_k, n_candidates, weights, half_life
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        rank = find_rank(results, q["ground_truth_id"])
        per_query.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["text"],
            "ground_truth_id": q["ground_truth_id"],
            "rank": rank,
            "top1_id": results[0]["id"] if results else None,
            "top1_text": results[0]["text"] if results else None,
            "latency_ms": round(latency_ms, 2),
        })

    return per_query


def compute_metrics(per_query, top_k):
    by_cat = defaultdict(list)
    for r in per_query:
        by_cat[r["category"]].append(r)

    recall_key = f"recall_at_{top_k}"

    def cat_metrics(rows):
        n = len(rows)
        top1 = sum(1 for r in rows if r["rank"] == 1)
        rec = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= top_k)
        mrr = sum(1.0 / r["rank"] for r in rows if r["rank"] is not None)
        lat = sum(r["latency_ms"] for r in rows)
        return {
            "n": n,
            "top1_accuracy": round(top1 / n, 4),
            recall_key: round(rec / n, 4),
            "mrr": round(mrr / n, 4),
            "avg_latency_ms": round(lat / n, 2),
        }

    return {
        "per_category": {cat: cat_metrics(rows) for cat, rows in sorted(by_cat.items())},
        "overall": cat_metrics(per_query),
    }


# ─── reporting ──────────────────────────────────────────────────────────────

def format_summary(output):
    m = output["metrics"]
    recall_key = next(k for k in m["overall"] if k.startswith("recall_at_"))
    top_k = output["top_k"]

    lines = [f"Mode: {output['mode']}"]
    if output["weights"]:
        w = output["weights"]
        lines.append(
            f"Weights: sem={w['semantic']} time={w['time']} "
            f"freq={w['frequency']} imp={w['importance']}  "
            f"half_life={output['half_life_days']}d"
        )
    lines += [
        "",
        f"{'Category':<28} {'top-1':>7} {'MRR':>7} {f'r@{top_k}':>7} {'lat ms':>8}",
        "-" * 60,
    ]
    for cat, vals in m["per_category"].items():
        lines.append(
            f"{cat:<28} {vals['top1_accuracy']:>7.2%} {vals['mrr']:>7.3f} "
            f"{vals[recall_key]:>7.2%} {vals['avg_latency_ms']:>8.1f}"
        )
    o = m["overall"]
    lines += [
        "-" * 60,
        f"{'OVERALL':<28} {o['top1_accuracy']:>7.2%} {o['mrr']:>7.3f} "
        f"{o[recall_key]:>7.2%} {o['avg_latency_ms']:>8.1f}",
    ]
    return "\n".join(lines)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="benchmarks/dataset.json")
    ap.add_argument("--mode", choices=["rag", "nlm"], default="nlm")
    ap.add_argument("--semantic", type=float, default=0.4)
    ap.add_argument("--time", type=float, default=0.2)
    ap.add_argument("--freq", type=float, default=0.2)
    ap.add_argument("--importance", type=float, default=0.2)
    ap.add_argument("--half-life", type=float, default=90.0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-candidates", type=int, default=50)
    ap.add_argument("--persist", default="./benchmarks/.bench_chroma")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)

    weights = {
        "semantic": args.semantic,
        "time": args.time,
        "frequency": args.freq,
        "importance": args.importance,
    }

    per_query = run(
        dataset, args.mode, weights, args.half_life,
        args.top_k, args.n_candidates, args.persist,
    )
    metrics = compute_metrics(per_query, args.top_k)

    output = {
        "dataset_version": dataset.get("version"),
        "dataset_path": args.dataset,
        "mode": args.mode,
        "weights": weights if args.mode == "nlm" else None,
        "half_life_days": args.half_life if args.mode == "nlm" else None,
        "top_k": args.top_k,
        "n_candidates": args.n_candidates,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "per_query": per_query,
    }

    print(format_summary(output))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
