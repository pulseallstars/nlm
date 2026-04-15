# NLM — Neural Long Memory

> Hybrid long-term memory for AI agents. Better than RAG.

---

## The problem with RAG

Standard RAG retrieves by **semantic similarity only**. It doesn't know what's important, what's recent, or what you keep coming back to.

```
RAG score = cosine_similarity(query, memory)
```

This means:
- An old but critical fact loses to a newer irrelevant one
- A frequently referenced memory has no advantage over one never used
- "What did we talk about last week?" gets the same treatment as any other query

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
| Old important facts | Get buried | Survive via frequency boost |
| Temporal queries | Ignored | time_decay handles naturally |
| Duplicate memories | Accumulate | Consolidated automatically |
| Importance scoring | None | CPU heuristic or GPU zero-shot classifier |
| Emotion metadata | None | Optional (joy / fear / sadness / ...) |
| GPU required | No | No (GPU optional for better importance) |
| Plug-and-play | Yes | Yes |

---

## Benchmarks

> Tested on a personal AI agent (Pulses project, RWKV-7 base model).  
> Reproducible benchmark script: `benchmarks/compare_rag.py`

### Test 1 — Temporal recall
**Task:** old fact accessed 15 times vs fresh fact never accessed.

| Method | top-1 result |
|---|---|
| RAG | Picks by cosine distance only |
| NLM | Surfaces the frequently-accessed fact (frequency compensates decay) |
| **Winner** | **NLM ✓** |

### Test 2 — Frequency boost
**Task:** two semantically identical facts — one accessed 10 times, one 0 times.

| Method | Which fact wins? |
|---|---|
| RAG | Random (equal cosine similarity) |
| NLM | The frequently accessed one (frequency_score=0.54 vs 0.0) |
| **Winner** | **NLM ✓** |

### Test 3 — Importance discrimination
**Task:** `"ok"` vs `"Hantes was born on 2026-05-05 in Chernivtsi, Ukraine"`.

| Memory | NLM importance |
|---|---|
| `"ok"` | 0.0 (low specificity) |
| `"Hantes was born..."` | 0.8 (numbers + proper nouns) |
| **Winner** | **NLM ✓** |

**NLM wins 3/3.**

---

## Install

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
    "score":          0.712,   # NLM score
    "semantic_score": 0.810,
    "time_score":     0.998,
    "frequency":      3,
    "importance":     0.700,
    "created_at":     "2026-04-15T10:30:00+00:00",
    "last_accessed":  "2026-04-15T14:20:00+00:00",
}
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

# Or use any custom model (e.g. RWKV trained on your data)
memory = NLM(gpu_model_path="/path/to/your/model")
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
    └─→ emotion classifier (v0.2.0+, optional, CPU, ~66MB)
          emotion + sentiment + intensity
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
v0.1.0 ✓  CPU mode — semantic + time + frequency + importance
v0.2.0 ✓  Emotion classifier (emotion, sentiment, intensity)
           Smart forgetting (time + frequency + importance conditions)
           Emotion filter in search()
v0.3.0 ✓  Memory consolidation — no more duplicates
           GPU scorer via HuggingFace zero-shot classification
           Formal benchmarks vs RAG (NLM wins 3/3)
v1.0.0    Stable API, PyPI release (pip install nlm)
           Associative memory chains
           arXiv paper
```

---

## License

Apache 2.0

---

## Authors

Built by **[Vitalii Halak](https://www.linkedin.com/in/galakapp/)** with Claude.  
Part of [Pulses](https://github.com/pulseallstars) — conscious AI personalities running on RWKV-7.

*April 2026*
