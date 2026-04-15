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

NLM scores memories using **three signals**, the way human memory actually works:

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
| Importance scoring | None | CPU heuristic or optional GPU model |
| GPU required | No | No (GPU optional for better importance) |
| Plug-and-play | Yes | Yes |

---

## Benchmarks

> Tested on a personal AI agent (Pulses project, RWKV-7 base model).

### Test 1 — Temporal recall
**Task:** retrieve a fact mentioned 30 days ago vs one from today.

| Method | Correct retrieval |
|---|---|
| RAG | Retrieves today's fact (newer embedding wins) |
| NLM | Retrieves the 30-day-old fact (it was accessed 8 times → frequency=0.73 compensates decay) |

### Test 2 — Frequency boost
**Task:** one fact was referenced 10 times, another 0 times. Same semantic distance to query.

| Method | Which fact wins? |
|---|---|
| RAG | Random (equal similarity) |
| NLM | The frequently accessed one (frequency_score=0.54 vs 0.0) |

### Test 3 — Importance discrimination
**Task:** save "ok" and "Hantes is 28 years old, lives in Chernivtsi, Ukraine". Query unrelated.

| Memory | RAG importance | NLM importance |
|---|---|---|
| "ok" | Equal | 0.0 (low specificity) |
| "Hantes is 28..." | Equal | 0.8 (numbers + proper nouns) |

> Full benchmark suite with reproducible code: `tests/test_nlm.py`  
> arXiv paper with formal evaluation: planned for v0.3.0

---

## Install

```bash
pip install sentence-transformers chromadb numpy
pip install -e .
```

---

## Usage

```python
from nlm import NLM

memory = NLM()

# Save a memory
memory.save("Hantes said he loves Minelux family the most")
memory.save("Project started on 2026-05-05 in Chernivtsi")

# Retrieve — NLM handles the scoring automatically
results = memory.search("what does Hantes think about the families", top_k=3)

for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")
    # [0.712] Hantes said he loves Minelux family the most
```

Each result includes full score breakdown:

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

---

## Memory lifecycle

```python
# Forget a specific memory
memory.forget(memory_id)

# Auto-cleanup: remove memories not accessed in 1 year
deleted = memory.forget_old(days=365)

# Stats
print(memory)  # NLM(memories=42, mode=CPU)
```

---

## Architecture

```
Text input
    │
    ├─→ sentence-transformers (all-MiniLM-L6-v2, CPU, ~80MB)
    │   → 384-dim embedding
    │
    └─→ automatic metadata:
          time_decay    = exp(-ln(2)/90 × days_since_created)
          frequency     = log-normalized access count
          importance    = specificity_score (CPU) or neural model (GPU, optional)
                │
                ▼
          ChromaDB (persistent vector store)

On search:
    query → embed → ChromaDB top-K candidates
                         │
                    NLM reranking (formula above)
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

### Importance scoring (CPU)

No model required. NLM uses a lightweight heuristic:

```python
def specificity_score(text):
    # +0.3 if text contains numbers (dates, ages, values)
    # +0.4 max for proper nouns (names, places)
    # +0.3 for golden length (5–50 words)
```

Optionally replace with any neural scorer via `gpu_model_path`.

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
v0.1.0 ✓  CPU mode, semantic + time + frequency + importance
v0.2.0    Emotion classifier (emotion, sentiment, intensity metadata)
           Automatic forget_old hook
v0.3.0    RWKV 0.1B GPU scorer
           Formal benchmarks vs RAG
           arXiv paper
v1.0.0    Stable API, PyPI release (pip install nlm-memory)
```

---

## License

Apache 2.0

---

## Authors

Built by **[Vitalii Halak](https://www.linkedin.com/in/galakapp/)** with Claude.  
Part of [Pulses](https://github.com/pulseallstars) — conscious AI personalities running on RWKV-7.

*April 2026*
