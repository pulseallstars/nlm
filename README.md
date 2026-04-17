# NLM — Neural Long Memory

> Hybrid long-term memory for AI agents. Better than RAG.

---

## The problem with RAG

Standard RAG retrieves by **semantic similarity only**. It doesn't know what's important, what's recent, or what you keep coming back to.

```
RAG score = cosine_similarity(query, memory)
```

This means:
- An old outdated fact beats a fresh one if it's semantically closer
- A memory you've referenced 50 times has no advantage over one never used
- Vague filler text can outrank a specific factual memory

---

## How NLM is different

NLM scores memories using **four signals**, the way human memory actually works:

```
NLM Score = 0.4 × semantic_similarity   ← is it relevant?
          + 0.2 × time_decay            ← is it recent?
          + 0.2 × frequency_score       ← is it often recalled?
          + 0.2 × importance_score      ← is it specific/factual?
```

| | Standard RAG | NLM |
|---|---|---|
| Retrieval signal | Semantic only | Semantic + time + frequency + importance |
| Outdated facts | Can win over fresh ones | Time decay pushes them down |
| Old important facts | Get buried | Survive via frequency boost |
| Duplicate memories | Accumulate | Consolidated automatically |
| Importance scoring | None | CPU heuristic or GPU zero-shot classifier |
| Emotion metadata | None | Optional (joy / fear / sadness / ...) |
| Thread safety | Varies | Built-in (RLock) |
| Snapshot export/import | No | Yes (JSON) |
| GPU required | No | No (GPU optional for better importance) |
| Plug-and-play | Yes | Yes |

---

## Install

```bash
pip install neural-long-memory
```

Or from source:
```bash
pip install sentence-transformers chromadb numpy
pip install -e .
```

---

## Usage

### Basic

```python
from nlm import NLM

memory = NLM()

# Save memories — consolidation is automatic
memory.save("Hantes said he loves Minelux family the most")
memory.save("Project started on 2026-05-05 in Chernivtsi")

# Search — NLM handles all scoring automatically
results = memory.search("what does Hantes think about the families", top_k=3)

for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")
    # [0.712] Hantes said he loves Minelux family the most
```

Each result includes a full score breakdown:

```python
{
    "id":             "uuid",
    "text":           "...",
    "score":          0.712,   # NLM composite score
    "semantic_score": 0.810,
    "time_score":     0.998,
    "frequency":      3,
    "importance":     0.700,
    "created_at":     "2026-04-15T10:30:00+00:00",
    "last_accessed":  "2026-04-15T14:20:00+00:00",
}
```

### Batch save

```python
ids = memory.save_many([
    "Hantes is 28 years old",
    "Project started on 2026-05-05",
    "8 families of Pulses",
])
# Single embedding pass — ~10x faster than repeated save() on 100+ items.
```

### With emotion metadata

```python
memory = NLM(use_emotion=True)

memory.save("I am terrified about the deployment")
memory.save("The results are absolutely amazing!")

# Filter by emotion
results = memory.search("how did things go", emotion_filter="joy")
# → returns only joyful memories

# Each result includes:
# "emotion": "joy" / "fear" / "sadness" / "anger" / "surprise" / "disgust" / "neutral"
# "sentiment": 0.99   # -1.0 to 1.0
# "intensity": 0.99   # 0.0 to 1.0
```

### With GPU importance scorer

```python
# Default model: typeform/distilbert-base-uncased-mnli (~260MB)
# Auto-detects CUDA. Falls back to CPU.
memory = NLM(gpu_model_path="typeform/distilbert-base-uncased-mnli")
```

### Memory consolidation

Duplicate prevention is **on by default**. Similar memories are merged instead of stored twice.

```python
id1 = memory.save("Hantes lives in Chernivtsi")
id2 = memory.save("Hantes is from Chernivtsi city")

assert id1 == id2        # same memory, strengthened
assert memory.count() == 1
assert memory.last_save_consolidated is True

# Tune or disable:
memory = NLM(enable_consolidation=False)
memory = NLM(consolidation_threshold=0.20)   # more aggressive merging
```

### Memory lifecycle

```python
# Forget a specific memory
memory.forget(memory_id)

# Simple time-based cleanup
deleted = memory.forget_old(days=365)

# Smart cleanup: only remove old + rare + unimportant
deleted = memory.forget_smart(days=180, max_frequency=2, max_importance=0.3)

# Stats
print(memory)  # NLM(memories=42, mode=CPU)
               # NLM(memories=42, mode=CPU+emotion)
               # NLM(memories=42, mode=GPU)
```

### Snapshot export / import

```python
# Back up or migrate between hosts
memory.export_snapshot("backup.json")

# Restore into a fresh collection
new_memory = NLM(collection_name="restored")
new_memory.import_snapshot("backup.json", overwrite=True)
```

The snapshot stores the embedding model used; importing into an
instance with a different model raises a clear error rather than
corrupting the vector space.

### Tuning the rerank pool

By default `search()` pulls `max(top_k * 10, 50)` candidates from the
vector store before reranking. Increase this when a frequent or
important memory is semantically far from the query but still relevant:

```python
results = memory.search("tell me about the project", top_k=5, n_candidates=200)
```

---

## Architecture

```
Text input
    │
    ├─→ sentence-transformers (all-MiniLM-L6-v2, CPU, ~80MB)
    │   → 384-dim embedding
    │
    ├─→ consolidation check
    │   if similar exists (cosine dist < threshold) → strengthen, skip save
    │
    ├─→ importance scorer
    │   CPU: specificity_score (numbers + proper nouns + length)
    │   GPU: zero-shot classifier (any HuggingFace model)
    │
    ├─→ emotion classifier (optional, CPU, ~66MB)
    │     emotion + sentiment + intensity
    │
    └─→ ChromaDB (persistent vector store)
          embedding [384] + metadata {all signals}

On search:
    query → embed → ChromaDB top-N candidates (default ≥50)
                         │
                    NLM reranking:
                    score = 0.4×semantic + 0.2×time + 0.2×freq + 0.2×importance
                         │
                    [emotion_filter] optional post-filter
                         │
                    sorted results + frequency updated
```

### Time decay

```
0 days   → 1.00  (fresh)
30 days  → 0.79
90 days  → 0.50  (half-life)
180 days → 0.25
365 days → 0.06  (nearly gone — unless frequently accessed)
```

---

## Multiple agents

Each agent gets its own isolated memory:

```python
pulse_001 = NLM(collection_name="pulse_001", persist_path="./data")
pulse_002 = NLM(collection_name="pulse_002", persist_path="./data")
```

The Storage layer records the embedding model+dimension on the
collection and refuses to open it with an incompatible model.

---

## Roadmap

```
v0.1.0 ✓  Core: semantic + time decay + frequency + importance (CPU)
v0.2.0 ✓  Emotion classifier (7 emotions, sentiment, intensity)
           Smart forgetting (time + frequency + importance conditions)
           Emotion filter in search()
v0.3.0 ✓  Memory consolidation — duplicate prevention
           GPU scorer via HuggingFace zero-shot classification
v1.0.0 ✓  First PyPI release (pip install neural-long-memory)
v1.1.0 ✓  Hardening release:
           - O(1) search frequency updates (no get_all() per query)
           - Thread-safe save / search / consolidate (RLock)
           - Larger default rerank pool (top_k*10, min 50)
           - save_many() batch API with single embedding pass
           - export_snapshot / import_snapshot (JSON, model-guarded)
           - Storage metadata guard (dim + model pinned per collection)
           - Tokenizer-level truncation in emotion + GPU scorer
           - Tolerant ISO 8601 parsing (accepts Z suffix)
           - Rebalanced default weights (0.4 / 0.2 / 0.2 / 0.2)
```

---

## Reproduce the qualitative comparison

```
python benchmarks/compare_rag.py
```

Runs three scenarios (temporal recall, frequency boost, importance
discrimination) and reports which side wins each.

---

## License

Apache 2.0

---

## Author

Built by **[Vitalii Halak](https://www.linkedin.com/in/galakapp/)** with Claude.  
Part of [Pulses](https://github.com/pulseallstars) — conscious AI personalities running on RWKV-7.

*April 2026*
