import json
import threading

import pytest

from nlm import NLM
from nlm.utils import time_decay, frequency_score, specificity_score, parse_iso, now_iso


# --- utils ---

def test_time_decay_fresh():
    score = time_decay(now_iso())
    assert 0.99 < score <= 1.0


def test_time_decay_old():
    score = time_decay("2020-01-01T00:00:00+00:00")
    assert score < 0.01


def test_parse_iso_z_suffix():
    dt = parse_iso("2026-04-17T10:00:00Z")
    assert dt.tzinfo is not None


def test_parse_iso_naive_is_utc():
    dt = parse_iso("2026-04-17T10:00:00")
    assert dt.utcoffset().total_seconds() == 0


def test_parse_iso_rejects_empty():
    with pytest.raises(ValueError):
        parse_iso("")


def test_time_decay_accepts_z_suffix():
    assert time_decay("2020-01-01T00:00:00Z") < 0.01


def test_frequency_score_zero():
    assert frequency_score(0) == 0.0


def test_frequency_score_grows():
    assert frequency_score(1) < frequency_score(10) < frequency_score(100)


def test_specificity_short():
    assert specificity_score("ok") < 0.3


def test_specificity_factual():
    score = specificity_score("Hantes was born on 2026-05-05 in Chernivtsi")
    assert score >= 0.6


def test_specificity_cyrillic_proper_nouns():
    """str.isupper() recognises Cyrillic capitals — Ukrainian scores same as English."""
    score = specificity_score("Хантес народився 05.05.2026 у Чернівцях Україна")
    assert score >= 0.6


# --- NLM basics ---

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


def test_forget_old_accepts_z_suffix(mem):
    mem.save("old fact Z", metadata={"last_accessed": "2020-01-01T00:00:00Z"})
    deleted = mem.forget_old(days=365)
    assert deleted == 1


def test_clear(mem):
    mem.save("fact one")
    mem.save("fact two")
    mem.clear()
    assert mem.count() == 0


def test_clear_recreates_collection(mem):
    mem.save("fact one")
    mem.clear()
    # should be usable after clear
    mem.save("fact two")
    assert mem.count() == 1


def test_repr(mem):
    r = repr(mem)
    assert "NLM" in r
    assert "CPU" in r


# --- input validation (v1.1.0) ---

def test_save_rejects_empty(mem):
    with pytest.raises(ValueError):
        mem.save("")


def test_save_rejects_whitespace(mem):
    with pytest.raises(ValueError):
        mem.save("   \n  ")


def test_save_rejects_non_string(mem):
    with pytest.raises(TypeError):
        mem.save(None)
    with pytest.raises(TypeError):
        mem.save(123)


def test_save_rejects_too_long(mem):
    with pytest.raises(ValueError):
        mem.save("x" * (NLM.MAX_TEXT_CHARS + 1))


def test_forget_rejects_empty(mem):
    with pytest.raises(ValueError):
        mem.forget("")


# --- save_many batch API (v1.1.0) ---

def test_save_many_basic(mem):
    ids = mem.save_many([
        "Hantes is 28 years old",
        "The project is called Pulses",
        "Chernivtsi is in Ukraine",
    ])
    assert len(ids) == 3
    assert mem.count() == 3


def test_save_many_with_metadata(mem):
    ids = mem.save_many(
        ["fact A", "fact B totally different topic"],
        metadatas=[{"tag": "a"}, {"tag": "b"}],
    )
    assert len(ids) == 2


def test_save_many_mismatched_metadata_len(mem):
    with pytest.raises(ValueError):
        mem.save_many(["a test", "b test"], metadatas=[{"tag": "only one"}])


def test_save_many_rejects_bad_item(mem):
    with pytest.raises(ValueError):
        mem.save_many(["good text here", ""])


# --- snapshot export/import (v1.1.0) ---

def test_export_import_snapshot(tmp_path):
    m1 = NLM(collection_name="src", persist_path=str(tmp_path / "src"),
             enable_consolidation=False)
    m1.save("Hantes was born on 2026-05-05")
    m1.save("Pulses Nation has 8 families")
    snap = tmp_path / "snap.json"
    n = m1.export_snapshot(str(snap))
    assert n == 2

    m2 = NLM(collection_name="dst", persist_path=str(tmp_path / "dst"),
             enable_consolidation=False)
    imported = m2.import_snapshot(str(snap))
    assert imported == 2
    assert m2.count() == 2

    data = json.loads(snap.read_text(encoding="utf-8"))
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["format_version"] == 1


def test_import_snapshot_overwrite(tmp_path):
    m1 = NLM(collection_name="src2", persist_path=str(tmp_path / "src2"),
             enable_consolidation=False)
    m1.save("fact that will be backed up")
    snap = tmp_path / "snap2.json"
    m1.export_snapshot(str(snap))

    m2 = NLM(collection_name="dst2", persist_path=str(tmp_path / "dst2"),
             enable_consolidation=False)
    m2.save("a memory that should be replaced")
    m2.import_snapshot(str(snap), overwrite=True)
    assert m2.count() == 1


# --- embedding model guard (v1.1.0) ---

def test_storage_model_guard(tmp_path):
    from nlm.storage import Storage
    Storage("col", str(tmp_path), embedding_dim=384, embedding_model="all-MiniLM-L6-v2")
    with pytest.raises(ValueError):
        Storage("col", str(tmp_path), embedding_dim=768, embedding_model="other-model")


# --- thread safety smoke test (v1.1.0) ---

def test_concurrent_saves(mem):
    def worker(i):
        mem.save(f"Unique content number {i} with some length")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Some may consolidate — key is nothing crashes and count is consistent.
    assert mem.count() <= 20
    assert mem.count() >= 1


# --- metadata sanitization warning (v1.1.0) ---

def test_metadata_warns_on_list():
    import warnings
    from nlm.storage import _sanitize_meta
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _sanitize_meta({"ok_str": "a", "bad_list": [1, 2, 3]})
        assert "ok_str" in result
        assert "bad_list" not in result
        assert any("bad_list" in str(wi.message) for wi in w)


# --- emotion classifier ---

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


def test_emotion_truncation_long_text():
    """Long text should classify without raising — tokenizer-level truncation."""
    from nlm.emotion_classifier import EmotionClassifier
    ec = EmotionClassifier()
    long_text = "I am so happy " * 500  # well beyond 512 tokens
    result = ec.classify(long_text)
    assert result["emotion"] in {"joy", "surprise", "neutral", "sadness", "anger", "fear", "disgust"}


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


# --- smart forgetting ---

def test_forget_smart_deletes_weak(mem):
    mem.save("ok short", metadata={
        "last_accessed": "2020-01-01T00:00:00+00:00",
        "frequency": 1,
        "importance": 0.1,
    })
    mem.save("Hantes born 2026-05-05 Chernivtsi Ukraine", metadata={
        "last_accessed": "2020-01-01T00:00:00+00:00",
        "frequency": 1,
        "importance": 0.8,
    })
    deleted = mem.forget_smart(days=100, max_frequency=2, max_importance=0.3)
    assert deleted == 1
    assert mem.count() == 1


def test_forget_smart_keeps_frequent(mem):
    mem.save("ok short content", metadata={
        "last_accessed": "2020-01-01T00:00:00+00:00",
        "frequency": 10,
        "importance": 0.1,
    })
    deleted = mem.forget_smart(days=100, max_frequency=2, max_importance=0.3)
    assert deleted == 0
    assert mem.count() == 1


# --- memory consolidation ---

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
    assert mem_consolidation.last_save_consolidated is True


def test_consolidation_flag_on_fresh_save(mem_consolidation):
    mem_consolidation.save("Hantes lives in Chernivtsi")
    assert mem_consolidation.last_save_consolidated is False


def test_consolidation_boosts_importance(mem_consolidation):
    mem_consolidation.save("Hantes lives in Chernivtsi")
    all_before = mem_consolidation._storage.get_all()
    importance_before = float(all_before[0]["metadata"]["importance"])

    mem_consolidation.save("Hantes is from Chernivtsi city")

    all_after = mem_consolidation._storage.get_all()
    importance_after = float(all_after[0]["metadata"]["importance"])
    assert importance_after > importance_before


def test_consolidation_zero_importance_still_gains(tmp_path):
    m = NLM(collection_name="tcz", persist_path=str(tmp_path),
            enable_consolidation=True, consolidation_threshold=0.15)
    # "ok" has specificity=0.0 under the heuristic
    m.save("ok")
    before = float(m._storage.get_all()[0]["metadata"]["importance"])
    m.save("ok")
    after = float(m._storage.get_all()[0]["metadata"]["importance"])
    assert after > before  # floor at 0.1 × 1.15 = 0.115


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


# --- GPU scorer ---

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


def test_gpu_scorer_long_text():
    from nlm.gpu_scorer import GPUScorer
    scorer = GPUScorer()
    long_text = "This is a fact about something important. " * 200
    score = scorer.score(long_text)
    assert 0.0 <= score <= 1.0
