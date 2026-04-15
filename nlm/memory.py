import uuid
from datetime import datetime, timezone

from nlm.embedder import Embedder
from nlm.storage import Storage
from nlm.scoring import rerank
from nlm.utils import now_iso, specificity_score


class NLM:
    def __init__(
        self,
        collection_name: str = "nlm_memory",
        persist_path: str = "./nlm_data",
        semantic_weight: float = 0.5,
        time_weight: float = 0.2,
        frequency_weight: float = 0.2,
        importance_weight: float = 0.1,
        half_life_days: float = 90,
        gpu_model_path: str = None,
        use_emotion: bool = False,
    ):
        self._weights = {
            "semantic": semantic_weight,
            "time": time_weight,
            "frequency": frequency_weight,
            "importance": importance_weight,
        }
        self._half_life = half_life_days
        self._embedder = Embedder()
        self._storage = Storage(collection_name, persist_path)
        self._gpu_scorer = None
        self._emotion = None

        if gpu_model_path:
            try:
                from nlm.gpu_scorer import GPUScorer
                self._gpu_scorer = GPUScorer(gpu_model_path)
            except ImportError:
                pass

        if use_emotion:
            from nlm.emotion_classifier import EmotionClassifier
            self._emotion = EmotionClassifier()

    def save(self, text: str, metadata: dict = None) -> str:
        """Store a memory. Returns the memory id."""
        memory_id = str(uuid.uuid4())
        importance = (
            self._gpu_scorer.score(text)
            if self._gpu_scorer
            else specificity_score(text)
        )
        meta = {
            "text": text,
            "created_at": now_iso(),
            "last_accessed": now_iso(),
            "frequency": 0,
            "importance": importance,
        }
        if self._emotion:
            meta.update(self._emotion.classify(text))
        if metadata:
            meta.update({k: v for k, v in metadata.items()
                          if isinstance(v, (str, int, float, bool))})
        embedding = self._embedder.encode(text)
        self._storage.add(memory_id, embedding, meta)
        return memory_id

    def search(self, query: str, top_k: int = 5, emotion_filter: str = None) -> list:
        """Find most relevant memories using NLM scoring.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            emotion_filter: Optional emotion label to filter by
                            (joy / sadness / anger / fear / surprise / disgust / neutral).
        """
        embedding = self._embedder.encode(query)
        n_candidates = top_k * 6 if emotion_filter else top_k * 2
        candidates = self._storage.query(embedding, n_results=n_candidates)
        if not candidates:
            return []

        ranked = rerank(candidates, self._weights, self._half_life)

        if emotion_filter:
            ranked = [r for r in ranked if r.get("emotion") == emotion_filter]

        results = ranked[:top_k]

        # Update frequency and last_accessed for returned memories
        for r in results:
            all_items = self._storage.get_all()
            meta_map = {item["id"]: item["metadata"] for item in all_items}
            if r["id"] in meta_map:
                meta = meta_map[r["id"]]
                meta["frequency"] = int(meta.get("frequency", 0)) + 1
                meta["last_accessed"] = now_iso()
                self._storage.update_metadata(r["id"], meta)
                r["frequency"] = meta["frequency"]

        return results

    def forget(self, memory_id: str):
        """Delete a specific memory."""
        self._storage.delete(memory_id)

    def forget_old(self, days: int = 365) -> int:
        """Delete memories not accessed for `days` days. Returns count deleted."""
        now = datetime.now(timezone.utc)
        deleted = 0
        for m in self._storage.get_all():
            last = m["metadata"].get("last_accessed", m["metadata"].get("created_at", ""))
            try:
                dt = datetime.fromisoformat(last)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if (now - dt).total_seconds() / 86400 >= days:
                    self._storage.delete(m["id"])
                    deleted += 1
            except Exception:
                pass
        return deleted

    def forget_smart(
        self,
        days: int = 180,
        max_frequency: int = 2,
        max_importance: float = 0.3,
    ) -> int:
        """Delete memories that satisfy ALL three conditions:
        - not accessed for `days` days
        - accessed fewer than `max_frequency` times
        - importance below `max_importance`

        Preserves old memories that are important or frequently recalled.
        Returns count deleted.
        """
        now = datetime.now(timezone.utc)
        deleted = 0
        for m in self._storage.get_all():
            meta = m["metadata"]
            last = meta.get("last_accessed", meta.get("created_at", ""))
            try:
                dt = datetime.fromisoformat(last)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = (now - dt).total_seconds() / 86400
                frequency = int(meta.get("frequency", 0))
                importance = float(meta.get("importance", 0.5))

                if age_days >= days and frequency < max_frequency and importance < max_importance:
                    self._storage.delete(m["id"])
                    deleted += 1
            except Exception:
                pass
        return deleted

    def count(self) -> int:
        return self._storage.count()

    def clear(self):
        self._storage.clear()

    def __repr__(self):
        mode = "GPU" if self._gpu_scorer else "CPU"
        if self._emotion:
            mode += "+emotion"
        return f"NLM(memories={self.count()}, mode={mode})"
