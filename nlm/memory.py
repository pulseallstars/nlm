import uuid

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

        if gpu_model_path:
            try:
                from nlm.gpu_scorer import GPUScorer
                self._gpu_scorer = GPUScorer(gpu_model_path)
            except ImportError:
                pass

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
        if metadata:
            meta.update({k: v for k, v in metadata.items()
                          if isinstance(v, (str, int, float, bool))})
        embedding = self._embedder.encode(text)
        self._storage.add(memory_id, embedding, meta)
        return memory_id

    def search(self, query: str, top_k: int = 5) -> list:
        """Find most relevant memories using NLM scoring."""
        embedding = self._embedder.encode(query)
        candidates = self._storage.query(embedding, n_results=top_k * 2)
        if not candidates:
            return []

        results = rerank(candidates, self._weights, self._half_life)[:top_k]

        # Update frequency and last_accessed for returned memories
        for r in results:
            existing = self._storage.query(
                self._embedder.encode(r["text"]), n_results=1
            )
            if existing:
                meta = existing[0]["metadata"]
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
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        all_memories = self._storage.get_all()
        deleted = 0
        for m in all_memories:
            last = m["metadata"].get("last_accessed", m["metadata"].get("created_at", ""))
            try:
                dt = datetime.fromisoformat(last)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = (now - dt).total_seconds() / 86400
                if age_days >= days:
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
        return f"NLM(memories={self.count()}, mode={mode})"
