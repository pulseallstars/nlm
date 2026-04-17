import json
import logging
import threading
import uuid
from datetime import datetime, timezone

from nlm.embedder import Embedder
from nlm.storage import Storage
from nlm.scoring import rerank
from nlm.utils import now_iso, parse_iso, specificity_score


log = logging.getLogger("nlm")


class NLM:
    MAX_TEXT_CHARS = 100_000

    def __init__(
        self,
        collection_name: str = "nlm_memory",
        persist_path: str = "./nlm_data",
        semantic_weight: float = 0.4,
        time_weight: float = 0.2,
        frequency_weight: float = 0.2,
        importance_weight: float = 0.2,
        half_life_days: float = 90,
        gpu_model_path: str = None,
        use_emotion: bool = False,
        enable_consolidation: bool = True,
        consolidation_threshold: float = 0.15,
    ):
        self._weights = {
            "semantic": semantic_weight,
            "time": time_weight,
            "frequency": frequency_weight,
            "importance": importance_weight,
        }
        self._half_life = half_life_days
        self._embedder = Embedder()
        self._storage = Storage(
            collection_name,
            persist_path,
            embedding_dim=self._embedder.dim,
            embedding_model=self._embedder.model_name,
        )
        self._gpu_scorer = None
        self._emotion = None
        self._enable_consolidation = enable_consolidation
        self._consolidation_threshold = consolidation_threshold
        self._lock = threading.RLock()
        self.last_save_consolidated = False

        if gpu_model_path:
            from nlm.gpu_scorer import GPUScorer
            self._gpu_scorer = GPUScorer(gpu_model_path)

        if use_emotion:
            from nlm.emotion_classifier import EmotionClassifier
            self._emotion = EmotionClassifier()

    # ---------- save ----------

    def _validate_text(self, text, label="text"):
        if not isinstance(text, str):
            raise TypeError(f"{label} must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError(f"{label} must be non-empty")
        if len(text) > self.MAX_TEXT_CHARS:
            raise ValueError(f"{label} exceeds {self.MAX_TEXT_CHARS} chars")

    def _build_meta(self, text, extra=None):
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
        if extra:
            meta.update(extra)
        return meta

    def save(self, text: str, metadata: dict = None) -> str:
        """Store a memory. Returns memory id.

        If consolidation is enabled and a similar memory exists,
        strengthens the existing one and returns its id instead.
        Check ``self.last_save_consolidated`` to tell the two cases apart.
        """
        self._validate_text(text)

        with self._lock:
            self.last_save_consolidated = False
            if self._enable_consolidation and self._storage.count() > 0:
                similar = self._find_similar(text)
                if similar:
                    self._consolidate(similar["id"], similar["metadata"])
                    self.last_save_consolidated = True
                    log.info("NLM consolidated save into existing memory %s", similar["id"])
                    return similar["id"]

            memory_id = str(uuid.uuid4())
            meta = self._build_meta(text, metadata)
            embedding = self._embedder.encode(text)
            self._storage.add(memory_id, embedding, meta)
            return memory_id

    def save_many(self, texts: list, metadatas: list = None) -> list:
        """Batch save. Returns list of ids (same length as texts).

        Uses a single embedding call for all texts — ~10x faster than
        repeated save() on 100+ items. Consolidation still runs per-item.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("len(metadatas) must match len(texts)")
        for i, t in enumerate(texts):
            self._validate_text(t, label=f"texts[{i}]")

        embeddings = self._embedder.encode_many(texts)

        ids_out = []
        new_ids, new_embeds, new_metas = [], [], []

        with self._lock:
            for i, text in enumerate(texts):
                emb = embeddings[i]
                extra = metadatas[i] if metadatas else None

                if self._enable_consolidation and self._storage.count() > 0:
                    candidates = self._storage.query(emb, n_results=1)
                    if candidates and candidates[0]["distance"] < self._consolidation_threshold:
                        self._consolidate(candidates[0]["id"], candidates[0]["metadata"])
                        ids_out.append(candidates[0]["id"])
                        log.info("NLM consolidated save_many[%d] into %s", i, candidates[0]["id"])
                        continue

                memory_id = str(uuid.uuid4())
                new_ids.append(memory_id)
                new_embeds.append(emb)
                new_metas.append(self._build_meta(text, extra))
                ids_out.append(memory_id)

            if new_ids:
                self._storage.add_many(new_ids, new_embeds, new_metas)

        return ids_out

    # ---------- consolidation ----------

    def _find_similar(self, text: str):
        embedding = self._embedder.encode(text)
        candidates = self._storage.query(embedding, n_results=1)
        if candidates and candidates[0]["distance"] < self._consolidation_threshold:
            return candidates[0]
        return None

    def _consolidate(self, existing_id: str, existing_meta: dict):
        """Strengthen an existing memory instead of creating a duplicate.

        Caller must hold self._lock.
        """
        existing_meta["frequency"] = int(existing_meta.get("frequency", 0)) + 1
        # Floor at 0.1 so memories with importance=0 still gain on consolidation.
        current = max(0.1, float(existing_meta.get("importance", 0.5)))
        existing_meta["importance"] = min(1.0, current * 1.15)
        existing_meta["last_accessed"] = now_iso()
        self._storage.update_metadata(existing_id, existing_meta)

    # ---------- search ----------

    def search(
        self,
        query: str,
        top_k: int = 5,
        emotion_filter: str = None,
        n_candidates: int = None,
    ) -> list:
        """Find most relevant memories using NLM scoring.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            emotion_filter: Optional emotion label to filter by
                (joy / sadness / anger / fear / surprise / disgust / neutral).
            n_candidates: Override the rerank candidate pool. Default
                max(top_k*10, 50), or max(top_k*30, 150) when emotion_filter
                is set (so the filter still has material to pick from).
        """
        self._validate_text(query, label="query")

        embedding = self._embedder.encode(query)
        if n_candidates is None:
            pool = max(top_k * 10, 50)
            if emotion_filter:
                pool = max(pool, top_k * 30, 150)
        else:
            pool = max(1, int(n_candidates))

        candidates = self._storage.query(embedding, n_results=pool)
        if not candidates:
            return []

        ranked = rerank(candidates, self._weights, self._half_life)
        if emotion_filter:
            ranked = [r for r in ranked if r.get("emotion") == emotion_filter]

        results = ranked[:top_k]

        # Update frequency/last_accessed using candidates map — no get_all().
        meta_map = {c["id"]: c["metadata"] for c in candidates}
        with self._lock:
            for r in results:
                meta = meta_map.get(r["id"])
                if meta is None:
                    continue
                meta["frequency"] = int(meta.get("frequency", 0)) + 1
                meta["last_accessed"] = now_iso()
                self._storage.update_metadata(r["id"], meta)
                r["frequency"] = meta["frequency"]

        return results

    # ---------- forgetting ----------

    def forget(self, memory_id: str):
        """Delete a specific memory."""
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("memory_id must be a non-empty string")
        self._storage.delete(memory_id)

    def forget_old(self, days: int = 365) -> int:
        """Delete memories not accessed for `days` days. Returns count deleted."""
        now = datetime.now(timezone.utc)
        deleted = 0
        for m in self._storage.get_all():
            last = m["metadata"].get("last_accessed", m["metadata"].get("created_at", ""))
            try:
                dt = parse_iso(last)
                if (now - dt).total_seconds() / 86400 >= days:
                    self._storage.delete(m["id"])
                    deleted += 1
            except Exception as e:
                log.debug("forget_old skipped %s: %s", m.get("id"), e)
        return deleted

    def forget_smart(
        self,
        days: int = 180,
        max_frequency: int = 2,
        max_importance: float = 0.3,
    ) -> int:
        """Delete memories satisfying ALL three conditions:
        not accessed for `days` AND frequency < max_frequency AND
        importance < max_importance. Returns count deleted.
        """
        now = datetime.now(timezone.utc)
        deleted = 0
        for m in self._storage.get_all():
            meta = m["metadata"]
            last = meta.get("last_accessed", meta.get("created_at", ""))
            try:
                dt = parse_iso(last)
                age_days = (now - dt).total_seconds() / 86400
                frequency = int(meta.get("frequency", 0))
                importance = float(meta.get("importance", 0.5))
                if age_days >= days and frequency < max_frequency and importance < max_importance:
                    self._storage.delete(m["id"])
                    deleted += 1
            except Exception as e:
                log.debug("forget_smart skipped %s: %s", m.get("id"), e)
        return deleted

    # ---------- snapshot ----------

    def export_snapshot(self, path: str) -> int:
        """Dump all memories (embeddings + metadata) to a JSON file.

        Returns the number of memories written.
        """
        data = {
            "format_version": 1,
            "embedding_model": self._embedder.model_name,
            "embedding_dim": self._embedder.dim,
            "exported_at": now_iso(),
            "memories": self._storage.export_all(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return len(data["memories"])

    def import_snapshot(self, path: str, overwrite: bool = False) -> int:
        """Load memories from a snapshot created with export_snapshot.

        Raises ValueError if the snapshot was built with a different
        embedding model — vectors would be incompatible.

        Returns the number of memories imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        snap_model = data.get("embedding_model")
        if snap_model and snap_model != self._embedder.model_name:
            raise ValueError(
                f"Snapshot embedding_model='{snap_model}' does not match "
                f"current embedder '{self._embedder.model_name}'. Vectors "
                f"would be incompatible."
            )

        if overwrite:
            self.clear()

        mems = data.get("memories", [])
        if not mems:
            return 0

        ids = [m["id"] for m in mems]
        embeds = [m["embedding"] for m in mems]
        metas = [m["metadata"] for m in mems]
        with self._lock:
            self._storage.add_many(ids, embeds, metas)
        return len(ids)

    # ---------- misc ----------

    def count(self) -> int:
        return self._storage.count()

    def clear(self):
        with self._lock:
            self._storage.clear()

    def __repr__(self):
        mode = "GPU" if self._gpu_scorer else "CPU"
        if self._emotion:
            mode += "+emotion"
        return f"NLM(memories={self.count()}, mode={mode})"
