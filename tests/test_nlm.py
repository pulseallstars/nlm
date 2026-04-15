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
