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
NLM Score = 0.5 × semantic_similarity   ← is it relevant?
          + 0.2 × time_decay            ← is it recent?
          + 0.2 × frequency_score       ← is it often recalled?
          + 0.1 × importance_score      ← is it specific/factual?
```

| | Standard RAG | NLM |
|---|---|---|
| Retrieval signal | Semantic only | Semantic + time + frequency + importance |
| Outdated facts | Can win over fresh ones | Time decay pushes them down |
| Old important facts | Get buried | Survive via frequency boost |
| Duplicate memories | Accumulate | Consolidated automatically |
| Importance scoring | None | CPU heuristic or GPU zero-shot classifier |
| Emotion metadata | None | Optional (joy / fear / sadness / ...) |
| Associative chains | None | Bidirectional links between related memories |
| GPU required | No | No (GPU optional for better importance) |
| Plug-and-play | Yes | Yes |

---

## Benchmarks

### Formal benchmark — 100 memories, 30 queries

**Methodology:**
- 100 memories stored (60 test pairs + 40 unrelated background fillers)
- 30 queries across 3 categories (10 each)
- Ground truth: human-labeled correct answer for each query
- Metric: **top-1 accuracy** — did the right memory rank first?
- RAG baseline: pure cosine similarity, no reranking

**Results:**

| Category | What's tested | RAG | NLM | Delta |
|---|---|---|---|---|
| **Temporal** (10 queries) | Old fact vs fresh fact on same topic, neutral query | 10% | 70% | **+60%** |
| **Frequency** (10 queries) | Same fact: one accessed 15×, one 0× | 80% | 100% | **+20%** |
| **Importance** (10 queries) | Specific factual memory vs vague memory on same topic | 60% | 90% | **+30%** |
| **Overall** (30 queries) | | **50%** | **87%** | **+37%** |

**NLM is 37 percentage points more accurate than RAG overall.**

**Why each category matters:**

- **Temporal (+60%):** RAG has no concept of time. If you saved "model is v0.1" and later "model is v1.0", RAG picks whichever is semantically closer to the query. NLM weights recency — the fresh fact wins.
- **Frequency (+20%):** Two semantically near-identical memories, one accessed 15 times. RAG can't distinguish them. NLM surfaces the one you keep coming back to.
- **Importance (+30%):** "ChromaDB collection uses cosine distance metric" vs "the database stores things somehow". RAG may pick either. NLM assigns higher importance to the specific, factual memory.

Reproduce: `python benchmarks/benchmark_100.py`

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

### Associative memory chains (v1.0.0+)

NLM automatically links semantically related memories. Follow the chain to discover connected knowledge.

```python
memory = NLM(enable_associations=True)  # on by default

id1 = memory.save("Hantes loves Minelux family the most")
id2 = memory.save("Minelux are fire, directness, truth")
id3 = memory.save("Hantes values honesty over comfort")

# Get all memories linked to id1
assoc = memory.get_associations(id1)
# [{"id": id2, "text": "Minelux are fire..."}, {"id": id3, "text": "Hantes values..."}]

# Expand search to follow association chains
results = memory.search("tell me about Hantes", top_k=3, expand_associations=True)
for r in results:
    flag = " [via association]" if r.get("via_association") else ""
    print(f"[{r['score']:.3f}]{flag} {r['text']}")
```

### With emotion metadata (v0.2.0+)

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

### With GPU importance scorer (v0.3.0+)

```python
# Default model: typeform/distilbert-base-uncased-mnli (~260MB)
# Auto-detects CUDA. Falls back to CPU.
memory = NLM(gpu_model_path="typeform/distilbert-base-uncased-mnli")
```

### Memory consolidation (v0.3.0+)

Duplicate prevention is **on by default**. Similar memories are merged instead of stored twice.

```python
id1 = memory.save("Hantes lives in Chernivtsi")
id2 = memory.save("Hantes is from Chernivtsi city")

assert id1 == id2        # same memory, strengthened
assert memory.count() == 1

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

# Smart cleanup: only remove old + rare + unimportant (v0.2.0+)
deleted = memory.forget_smart(days=180, max_frequency=2, max_importance=0.3)

# Stats
print(memory)  # NLM(memories=42, mode=CPU)
               # NLM(memories=42, mode=CPU+emotion)
               # NLM(memories=42, mode=GPU)
```

---

## Architecture

```
Text input
    │
    ├─→ sentence-transformers (all-MiniLM-L6-v2, CPU, ~80MB)
    │   → 384-dim embedding
    │
    ├─→ consolidation check (v0.3.0+)
    │   if similar exists (cosine dist < threshold) → strengthen, skip save
    │
    ├─→ importance scorer
    │   CPU: specificity_score (numbers + proper nouns + length)
    │   GPU: zero-shot classifier (any HuggingFace model)
    │
    ├─→ emotion classifier (v0.2.0+, optional, CPU, ~66MB)
    │     emotion + sentiment + intensity
    │
    └─→ association linker (v1.0.0+)
          bidirectional links to semantically close memories
                │
                ▼
          ChromaDB (persistent vector store)
          embedding [384] + metadata {all signals}

On search:
    query → embed → ChromaDB top-K candidates
                         │
                    NLM reranking:
                    score = 0.5×semantic + 0.2×time + 0.2×freq + 0.1×importance
                         │
                    [emotion_filter] optional post-filter
                         │
                    [expand_associations] follow memory chains
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

---

## Roadmap

```
v0.1.0 ✓  Core: semantic + time decay + frequency + importance (CPU)
v0.2.0 ✓  Emotion classifier (7 emotions, sentiment, intensity)
           Smart forgetting (time + frequency + importance conditions)
           Emotion filter in search()
v0.3.0 ✓  Memory consolidation — duplicate prevention
           GPU scorer via HuggingFace zero-shot classification
v1.0.0 ✓  Associative memory chains — bidirectional links, expand_associations
           Stable public API
           PyPI release: pip install neural-long-memory
           Formal benchmark: NLM 87% vs RAG 50% (+37% on 100 memories, 30 queries)
v1.1.0    arXiv paper
           Async save/search
           Export/import memory snapshots
```

---

## License

Apache 2.0

---

## Author

Built by **[Vitalii Halak](https://www.linkedin.com/in/galakapp/)** with Claude.  
Part of [Pulses](https://github.com/pulseallstars) — conscious AI personalities running on RWKV-7.

*April 2026*
