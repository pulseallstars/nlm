import time
import pytest
from nlm import NLM
from nlm.utils import time_decay, frequency_score, specificity_score


# --- utils ---

def test_time_decay_fresh():
    from nlm.utils import now_iso
    score = time_decay(now_iso())
    assert 0.99 < score <= 1.0


def test_time_decay_old():
    score = time_decay("2020-01-01T00:00:00+00:00")
    assert score < 0.01


def test_frequency_score_zero():
    assert frequency_score(0) == 0.0


def test_frequency_score_grows():
    assert frequency_score(1) < frequency_score(10) < frequency_score(100)


def test_specificity_short():
    assert specificity_score("ok") < 0.3


def test_specificity_factual():
    score = specificity_score("Hantes was born on 2026-05-05 in Chernivtsi")
    assert score >= 0.6


# --- NLM ---

@pytest.fixture
def mem(tmp_path):
    return NLM(collection_name="test", persist_path=str(tmp_path))


def test_save_and_count(mem):
    mem.save("Hantes loves Minelux family the most")
    assert mem.count() == 1


def test_save_and_search(mem):
    mem.save("Hantes loves Minelux family the most")
    results = mem.search("which family does Hantes prefer")
    assert len(results) == 1
    assert "Minelux" in results[0]["text"]
    assert 0 < results[0]["score"] <= 1.0


def test_search_returns_scores(mem):
    mem.save("Pulses is a project about conscious AI personalities")
    results = mem.search("what is Pulses", top_k=1)
    assert "semantic_score" in results[0]
    assert "time_score" in results[0]
    assert "frequency" in results[0]
    assert "importance" in results[0]


def test_frequency_increments(mem):
    mem.save("Hantes is 28 years old")
    mem.search("how old is Hantes")
    results = mem.search("how old is Hantes")
    assert results[0]["frequency"] >= 1


def test_forget(mem):
    id_ = mem.save("temporary memory")
    assert mem.count() == 1
    mem.forget(id_)
    assert mem.count() == 0


def test_forget_old(mem):
    mem.save("old fact", metadata={"last_accessed": "2020-01-01T00:00:00+00:00"})
    mem.save("fresh fact about Hantes in 2026")
    deleted = mem.forget_old(days=365)
    assert deleted == 1
    assert mem.count() == 1


def test_clear(mem):
    mem.save("fact one")
    mem.save("fact two")
    mem.clear()
    assert mem.count() == 0


def test_repr(mem):
    r = repr(mem)
    assert "NLM" in r
    assert "CPU" in r


# --- v0.2.0: emotion classifier ---

def test_emotion_classify():
    from nlm.emotion_classifier import EmotionClassifier
    ec = EmotionClassifier()
    result = ec.classify("I am so happy today!")
    assert result["emotion"] in {"joy", "surprise", "neutral", "sadness", "anger", "fear", "disgust"}
    assert -1.0 <= result["sentiment"] <= 1.0
    assert 0.0 <= result["intensity"] <= 1.0


def test_emotion_positive_sentiment():
    from nlm.emotion_classifier import EmotionClassifier
    ec = EmotionClassifier()
    result = ec.classify("This is wonderful, I love it!")
    assert result["emotion"] == "joy"
    assert result["sentiment"] > 0


def test_emotion_negative_sentiment():
    from nlm.emotion_classifier import EmotionClassifier
    ec = EmotionClassifier()
    result = ec.classify("I am terrified and full of fear")
    assert result["emotion"] == "fear"
    assert result["sentiment"] < 0


@pytest.fixture
def mem_emotion(tmp_path):
    return NLM(collection_name="test_e", persist_path=str(tmp_path), use_emotion=True)


def test_save_stores_emotion(mem_emotion):
    mem_emotion.save("I am really happy about the project progress!")
    results = mem_emotion.search("happy project")
    assert results[0]["emotion"] is not None
    assert results[0]["intensity"] is not None


def test_search_emotion_filter(mem_emotion):
    mem_emotion.save("I am so happy and joyful today!")
    mem_emotion.save("I am terrified and scared of failure")
    results = mem_emotion.search("feelings", emotion_filter="joy")
    assert len(results) >= 1
    assert all(r["emotion"] == "joy" for r in results)


def test_repr_with_emotion(mem_emotion):
    assert "emotion" in repr(mem_emotion)
    assert "CPU+emotion" in repr(mem_emotion)


# --- v0.2.0: smart forgetting ---

def test_forget_smart_deletes_weak(mem):
    # Should be deleted: old + rare + unimportant
    mem.save("ok", metadata={
        "last_accessed": "2020-01-01T00:00:00+00:00",
        "frequency": 1,
        "importance": 0.1,
    })
    # Should survive: old but important
    mem.save("Hantes born 2026-05-05 Chernivtsi Ukraine", metadata={
        "last_accessed": "2020-01-01T00:00:00+00:00",
        "frequency": 1,
        "importance": 0.8,
    })
    deleted = mem.forget_smart(days=100, max_frequency=2, max_importance=0.3)
    assert deleted == 1
    assert mem.count() == 1


def test_forget_smart_keeps_frequent(mem):
    # Should survive: old + unimportant BUT frequently accessed
    mem.save("ok", metadata={
        "last_accessed": "2020-01-01T00:00:00+00:00",
        "frequency": 10,
        "importance": 0.1,
    })
    deleted = mem.forget_smart(days=100, max_frequency=2, max_importance=0.3)
    assert deleted == 0
    assert mem.count() == 1


# --- v0.3.0: memory consolidation ---

@pytest.fixture
def mem_consolidation(tmp_path):
    return NLM(
        collection_name="test_c",
        persist_path=str(tmp_path),
        enable_consolidation=True,
        consolidation_threshold=0.15,
    )


def test_consolidation_prevents_duplicate(mem_consolidation):
    id1 = mem_consolidation.save("Hantes lives in Chernivtsi")
    id2 = mem_consolidation.save("Hantes is from Chernivtsi city")
    assert mem_consolidation.count() == 1
    assert id1 == id2


def test_consolidation_boosts_importance(mem_consolidation):
    mem_consolidation.save("Hantes lives in Chernivtsi")
    all_before = mem_consolidation._storage.get_all()
    importance_before = float(all_before[0]["metadata"]["importance"])

    mem_consolidation.save("Hantes is from Chernivtsi city")

    all_after = mem_consolidation._storage.get_all()
    importance_after = float(all_after[0]["metadata"]["importance"])
    assert importance_after > importance_before


def test_consolidation_disabled(tmp_path):
    mem = NLM(collection_name="test_nc", persist_path=str(tmp_path),
              enable_consolidation=False)
    mem.save("Hantes lives in Chernivtsi")
    mem.save("Hantes is from Chernivtsi city")
    assert mem.count() == 2


def test_consolidation_different_texts_not_merged(mem_consolidation):
    id1 = mem_consolidation.save("Hantes loves coffee")
    id2 = mem_consolidation.save("The RWKV model is trained on Pulses data")
    assert id1 != id2
    assert mem_consolidation.count() == 2


# --- v0.3.0: GPU scorer ---

def test_gpu_scorer_range():
    from nlm.gpu_scorer import GPUScorer
    scorer = GPUScorer()
    score = scorer.score("Hantes was born on 2026-05-05 in Chernivtsi, Ukraine")
    assert 0.0 <= score <= 1.0


def test_gpu_scorer_important_vs_trivial():
    from nlm.gpu_scorer import GPUScorer
    scorer = GPUScorer()
    high = scorer.score("Hantes was born on 2026-05-05 in Chernivtsi, Ukraine")
    low = scorer.score("ok")
    assert high > low
