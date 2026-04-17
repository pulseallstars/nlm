"""
Grid-search NLM weight combinations against the benchmark dataset.

Encoding happens once: all 100 memories and 30 queries are embedded a single
time, the raw candidate pool is fetched from the store once per query, and
every weight combination just re-runs the in-memory rerank. A 400-cell sweep
takes seconds instead of an hour of redundant model loads.

Examples:
    # Default sweep — 480 combos, sorted by top-1 accuracy
    python benchmarks/tune.py --save benchmarks/results/grid.csv

    # Tighter search around a promising region
    python benchmarks/tune.py \\
        --semantic 0.55,0.6,0.65 \\
        --time 0.15,0.2,0.25 \\
        --freq 0.0,0.05,0.1 \\
        --importance 0.1,0.15,0.2 \\
        --metric mrr
"""

import argparse
import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlm.embedder import Embedder
from nlm.scoring import rerank

from benchmark import (
    build_store,
    compute_metrics,
    find_rank,
    load_dataset,
)


def parse_range(s):
    return [float(x) for x in s.split(",")]


def cache_candidates(dataset, store, embedder, n_candidates):
    """Run each query once, keep raw candidates for in-memory rerank later."""
    cached = []
    for q in dataset["queries"]:
        emb = embedder.encode(q["text"])
        cands = store.query(emb, n_results=n_candidates)
        cached.append({
            "id": q["id"],
            "category": q["category"],
            "ground_truth_id": q["ground_truth_id"],
            "candidates": cands,
        })
    return cached


def evaluate(cached, weights, half_life, top_k):
    per_query = []
    for q in cached:
        ranked = rerank(q["candidates"], weights, half_life)[:top_k]
        per_query.append({
            "id": q["id"],
            "category": q["category"],
            "ground_truth_id": q["ground_truth_id"],
            "rank": find_rank(ranked, q["ground_truth_id"]),
            "latency_ms": 0.0,
        })
    return compute_metrics(per_query, top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="benchmarks/dataset.json")
    ap.add_argument("--persist", default="./benchmarks/.bench_chroma")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-candidates", type=int, default=50)
    ap.add_argument("--half-life", type=float, default=90.0)
    ap.add_argument("--semantic", type=parse_range, default="0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--time", type=parse_range, default="0.05,0.1,0.2,0.3")
    ap.add_argument("--freq", type=parse_range, default="0.0,0.05,0.1,0.2")
    ap.add_argument("--importance", type=parse_range, default="0.05,0.1,0.15,0.2")
    ap.add_argument("--metric", choices=["top1_accuracy", "mrr", "recall"], default="top1_accuracy")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--save", default=None, help="CSV path for full results")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    metric_key = f"recall_at_{args.top_k}" if args.metric == "recall" else args.metric

    print("Encoding memories and building store...")
    embedder = Embedder()
    store = build_store(dataset, args.persist, embedder)

    print(f"Encoding {len(dataset['queries'])} queries and caching candidates...")
    cached = cache_candidates(dataset, store, embedder, args.n_candidates)

    combos = list(itertools.product(args.semantic, args.time, args.freq, args.importance))
    print(f"Sweeping {len(combos)} weight combinations...\n")

    rows = []
    t0 = time.perf_counter()
    for sem, tim, frq, imp in combos:
        weights = {"semantic": sem, "time": tim, "frequency": frq, "importance": imp}
        m = evaluate(cached, weights, args.half_life, args.top_k)
        row = {
            "semantic": sem, "time": tim, "frequency": frq, "importance": imp,
            "top1_accuracy": m["overall"]["top1_accuracy"],
            "mrr": m["overall"]["mrr"],
            f"recall_at_{args.top_k}": m["overall"][f"recall_at_{args.top_k}"],
        }
        for cat, vals in m["per_category"].items():
            row[f"top1_{cat}"] = vals["top1_accuracy"]
        rows.append(row)
    elapsed = time.perf_counter() - t0

    rows.sort(key=lambda r: r[metric_key], reverse=True)

    print(f"Done in {elapsed:.1f}s. Top {args.top_n} by {metric_key}:\n")
    headers = ["sem", "time", "freq", "imp", "top1", "mrr", f"r@{args.top_k}",
               "tmp_recall", "tmp_conf", "freq_b", "imp_d", "proper", "multi"]
    print(("{:>5} " * 4 + "{:>7} {:>7} {:>7}  " + "{:>10} " * 6).format(*headers))
    print("-" * 130)
    for r in rows[: args.top_n]:
        print(
            f"{r['semantic']:>5.2f} {r['time']:>5.2f} {r['frequency']:>5.2f} {r['importance']:>5.2f}  "
            f"{r['top1_accuracy']:>6.2%} {r['mrr']:>7.3f} {r[f'recall_at_{args.top_k}']:>6.2%}  "
            f"{r.get('top1_temporal_recall', 0):>9.2%} {r.get('top1_temporal_conflict', 0):>9.2%} "
            f"{r.get('top1_frequency_boost', 0):>9.2%} {r.get('top1_importance_discrimination', 0):>9.2%} "
            f"{r.get('top1_proper_noun_precision', 0):>9.2%} {r.get('top1_multi_hop', 0):>9.2%}"
        )

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nFull grid saved to {args.save}")

    # Companion JSON next to CSV — easy machine read
    json_path = (
        Path(args.save).with_suffix(".best.json") if args.save
        else Path("benchmarks/results/best.json")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "metric": metric_key,
            "n_combinations": len(rows),
            "elapsed_seconds": round(elapsed, 2),
            "best": rows[0],
            "top_n": rows[: args.top_n],
        }, f, indent=2)
    print(f"Best config: {json_path}")


if __name__ == "__main__":
    main()
