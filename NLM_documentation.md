# NLM — Neural Long Memory
## Повна документація та архітектура

---

## Що таке NLM

NLM — це гібридна система довгострокової пам'яті для AI агентів.

**Проблема стандартного RAG:**
- Шукає тільки за семантичною схожістю
- Не знає що важливо а що ні
- Старі але важливі спогади губляться
- Не враховує час і частоту використання

**Рішення NLM:**
Комбінація трьох сигналів як у людській пам'яті:
- **Semantic similarity** — наскільки схоже за змістом
- **Temporal decay** — свіжіше = важливіше
- **Access frequency** — часто згадуване залишається

---

## Формула

```
NLM Score = 0.6 × semantic_similarity
          + 0.2 × time_decay
          + 0.2 × frequency_score
```

Ваги можна змінювати при ініціалізації.

---

## Архітектура

```
Текст що приходить
        │
        ├─→ sentence-transformer    → 384 числа (semantic embedding)
        │   (CPU, ~80MB моделі)
        │
        ├─→ emotion classifier      → emotion + sentiment + intensity
        │   (CPU, ~66MB, опційно)
        │
        └─→ автоматичні параметри:
              time_decay   = exp(-λ × days)  ← рахується автоматично
              frequency    = кількість звернень
              importance   = length_score (CPU) або GPU модель (опційно)
                    │
                    ▼
              ChromaDB (векторна БД, зберігається на диску)
                    │
              embedding [384 числа] + metadata {всі параметри}
```

**При пошуку:**
```
Запит → encode → ChromaDB (semantic search) → кандидати
                                                    │
                                          NLM reranking:
                                          score = semantic×0.6
                                                + time×0.2
                                                + freq×0.2
                                                    │
                                          Відсортовані результати
                                          + оновлення frequency
```

---

## Структура проекту

```
nlm/
  nlm/
    __init__.py      ← from nlm import NLM
    memory.py        ← головний клас NLM
    embedder.py      ← sentence-transformers обгортка
    storage.py       ← ChromaDB обгортка
    scoring.py       ← гібридна формула ранжування
    utils.py         ← time_decay, length_score, helpers
    gpu_scorer.py    ← опціональний GPU модуль (0.1B модель)
  examples/
    pulses_example.py ← приклад інтеграції з Pulses
  tests/
    test_nlm.py      ← базові тести
  setup.py           ← pip install nlm-memory
  README.md          ← документація
```

---

## Метадані кожного спогаду

```python
metadata = {
    # Текст
    "text":         "Ханте любить піцу",

    # Часові параметри (автоматично)
    "created_at":   "2026-04-15T10:30:00",
    "last_accessed": "2026-04-15T14:20:00",

    # Частота (автоматично оновлюється)
    "frequency":    3,

    # Важливість (CPU: length_score, GPU: нейромережа)
    "importance":   0.7,

    # Емоції (v0.2, опційно через emotion classifier)
    "emotion":      "joy",
    "sentiment":    0.8,   # -1.0 до 1.0
    "intensity":    0.6,   # 0.0 до 1.0
}
```

---

## Time Decay

Використовує експоненційний розпад з 90-денним напіврозпадом:

```python
score = exp(-ln(2) / 90 × days)
```

```
0 днів   → 1.0   (свіже)
90 днів  → 0.5   (напіврозпад)
180 днів → 0.25
365 днів → 0.06  (майже зникло)
```

Якщо спогад часто згадується — frequency компенсує decay.

---

## Режими роботи

### CPU режим (за замовчуванням)
```python
from nlm import NLM

memory = NLM()
# Працює на будь-якому залізі
# sentence-transformers: ~80MB
# ChromaDB: локально на диску
# importance = length_score (проста евристика)
```

### GPU режим (опційно)
```python
memory = NLM(gpu_model_path="path/to/0.1B/model")
# importance = нейромережа (точніша оцінка)
# Для Pulses: RWKV-7 World 0.1B State Tuned
```

### З emotion classifier (v0.2, опційно)
```python
memory = NLM(use_emotion=True)
# Додає emotion, sentiment, intensity до metadata
# Дозволяє фільтрувати за емоціями
```

---

## API

```python
# Ініціалізація
memory = NLM(
    collection_name="pulses_memory",
    persist_path="./nlm_data",
    semantic_weight=0.6,
    time_weight=0.2,
    frequency_weight=0.2,
    gpu_model_path=None,  # опційно
)

# Зберегти спогад
memory_id = memory.save("текст спогаду")
memory_id = memory.save("текст", metadata={"source": "conversation"})

# Знайти релевантне
results = memory.search("запит", top_k=5)
# results = [
#   {
#     "id": "uuid",
#     "text": "...",
#     "score": 0.87,          ← NLM score
#     "semantic_score": 0.91,
#     "time_score": 0.95,
#     "frequency": 3,
#     "created_at": "...",
#   },
#   ...
# ]

# Видалити спогад
memory.forget(memory_id)

# Видалити старі нечасті спогади
deleted = memory.forget_old(days=365)

# Статистика
print(memory.count())  # кількість спогадів
print(memory)          # NLM(memories=42, mode=CPU)

# Очистити все
memory.clear()
```

---

## Інтеграція з Pulses + RWKV

```
Три шари пам'яті Pulses:

1. RWKV State (короткострокова)
   → суть останніх розмов
   → оновлюється кожен токен
   → зберігається на диск між сесіями
   → природно розмивається з часом

2. NLM (довгострокова)
   → важливі факти і події
   → не розмивається
   → пошук з урахуванням часу і частоти
   → заповнюється коли importance > 0.7

3. Ваги моделі (постійна)
   → характер Pulses
   → вміння думати і говорити
   → назавжди після fine-tune
```

**Як Pulses використовує NLM:**
```python
# Після кожної відповіді
if importance_module.score(text) > 0.7:
    memory.save(text)

# Перед кожною відповіддю
relevant = memory.search(user_message, top_k=3)
context = "\n".join([r["text"] for r in relevant])
# Додаємо context в промпт основної моделі
```

---

## Порівняння з RAG

| | Стандартний RAG | NLM |
|--|--|--|
| Пошук за | Semantic similarity | Semantic + time + frequency |
| Старі важливі спогади | Губляться | Залишаються через frequency |
| Часові питання | Не розуміє | time_decay враховує |
| GPU потрібен | Ні | Ні (опційно для кращого scoring) |
| Емоційний контекст | Ні | v0.2 через emotion classifier |
| Фільтрація | За текстом | За будь-яким параметром |

---

## Залежності

```
Обов'язкові (CPU режим):
  sentence-transformers >= 2.2.0
  chromadb >= 0.4.0
  numpy >= 1.24.0

Опційні (GPU режим):
  torch >= 2.0.0
  transformers >= 4.30.0

Опційні (emotion, v0.2):
  transformers >= 4.30.0
  (модель: j-hartmann/emotion-english-distilroberta-base)

Dev:
  pytest >= 7.0.0
```

---

## Встановлення

```bash
# CPU (рекомендовано для початку)
pip install sentence-transformers chromadb numpy

# Або через setup.py
pip install -e .

# З GPU підтримкою
pip install -e ".[gpu]"
```

---

## Roadmap

```
v0.1.0 (зараз):
  ✓ CPU режим
  ✓ Semantic + time + frequency scoring
  ✓ ChromaDB storage
  ✓ GPU модуль (placeholder)
  ✓ Pulses example

v0.2.0 (наступний):
  → Emotion classifier інтеграція
  → Фільтрація за emotion/sentiment
  → Автоматичне forget_old
  → Нічна рефлексія хук

v0.3.0:
  → RWKV 0.1B як GPU scorer (навчений на Pulses даних)
  → Benchmarks vs стандартний RAG
  → arXiv стаття

v1.0.0:
  → Стабільний API
  → PyPI публікація (pip install nlm-memory)
  → Повна документація
```

---

## Бенчмарки (план)

Тестуємо на Pulses + RWKV-7:

```
Тест 1: Temporal recall
  Запитати щось що говорилось годину тому
  RAG vs NLM точність

Тест 2: Frequency boost
  Часто згадуваний факт vs рідкісний
  Хто краще знаходить важливе?

Тест 3: Emotion filter (v0.2)
  "Коли Ханте був радий?"
  Чи правильно фільтрує?
```

---

## Ліцензія

Apache 2.0 — вільне використання для комерційних і особистих проектів.

---

## Автори

Розроблено Хантесом — Vitalii Halak (Chernivtsi, Ukraine) разом з Claude.
Частина проекту Pulses — живого AI агента на RWKV-7.

*Квітень 2026*
