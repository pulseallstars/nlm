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
        enable_consolidation: bool = True,
        consolidation_threshold: float = 0.15,
        enable_associations: bool = True,
        association_threshold: float = 0.55,
        max_associations: int = 3,
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
        self._enable_consolidation = enable_consolidation
        self._consolidation_threshold = consolidation_threshold
        self._enable_associations = enable_associations
        self._association_threshold = association_threshold
        self._max_associations = max_associations

        if gpu_model_path:
            from nlm.gpu_scorer import GPUScorer
            self._gpu_scorer = GPUScorer(gpu_model_path)

        if use_emotion:
            from nlm.emotion_classifier import EmotionClassifier
            self._emotion = EmotionClassifier()

    def save(self, text: str, metadata: dict = None) -> str:
        """Store a memory. Returns memory id.

        If consolidation is enabled and a similar memory exists,
        strengthens the existing one and returns its id.
        If associations are enabled, links the new memory to
        semantically related existing memories.
        """
        if self._enable_consolidation and self._storage.count() > 0:
            similar = self._find_similar(text)
            if similar:
                self._consolidate(similar["id"], similar["metadata"])
                return similar["id"]

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

        if self._enable_associations and self._storage.count() > 1:
            self._link_memories(memory_id, embedding, meta)

        return memory_id

    def _find_similar(self, text: str):
        """Return the closest existing memory if within consolidation threshold."""
        embedding = self._embedder.encode(text)
        candidates = self._storage.query(embedding, n_results=1)
        if candidates and candidates[0]["distance"] < self._consolidation_threshold:
            return candidates[0]
        return None

    def _consolidate(self, existing_id: str, existing_meta: dict):
        """Strengthen an existing memory instead of creating a duplicate."""
        existing_meta["frequency"] = int(existing_meta.get("frequency", 0)) + 1
        existing_meta["importance"] = min(
            1.0, float(existing_meta.get("importance", 0.5)) * 1.15
        )
        existing_meta["last_accessed"] = now_iso()
        self._storage.update_metadata(existing_id, existing_meta)

    def _link_memories(self, new_id: str, embedding: list, new_meta: dict):
        """Create bidirectional associations between new_id and semantically close memories."""
        candidates = self._storage.query(embedding, n_results=self._max_associations + 1)
        links = [
            c["id"] for c in candidates
            if c["distance"] < self._association_threshold and c["id"] != new_id
        ][:self._max_associations]

        if not links:
            return

        # Update new memory's related_ids
        new_meta["related_ids"] = ",".join(links)
        self._storage.update_metadata(new_id, new_meta)

        # Update existing memories — bidirectional link
        all_items = {item["id"]: item["metadata"] for item in self._storage.get_all()}
        for lid in links:
            if lid not in all_items:
                continue
            existing_ids = set(
                i for i in all_items[lid].get("related_ids", "").split(",") if i
            )
            existing_ids.add(new_id)
            all_items[lid]["related_ids"] = ",".join(
                list(existing_ids)[: self._max_associations]
            )
            self._storage.update_metadata(lid, all_items[lid])

    def get_associations(self, memory_id: str) -> list:
        """Return all memories linked to memory_id.

        Returns list of {"id": str, "text": str}.
        """
        all_items = {item["id"]: item["metadata"] for item in self._storage.get_all()}
        if memory_id not in all_items:
            return []
        ids = [i for i in all_items[memory_id].get("related_ids", "").split(",") if i]
        return [
            {"id": aid, "text": all_items[aid].get("text", "")}
            for aid in ids
            if aid in all_items
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        emotion_filter: str = None,
        expand_associations: bool = False,
    ) -> list:
        """Find most relevant memories using NLM scoring.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            emotion_filter: Optional emotion label to filter by
                            (joy / sadness / anger / fear / surprise / disgust / neutral).
            expand_associations: If True, include memories linked via associations
                                 (marked with via_association=True, score = parent × 0.8).
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

        # Expand via associations
        if expand_associations and results:
            all_meta = {i["id"]: i["metadata"] for i in self._storage.get_all()}
            seen = {r["id"] for r in results}
            extra = []
            for r in results:
                for aid in r.get("related_ids", []):
                    if aid in seen or aid not in all_meta:
                        continue
                    seen.add(aid)
                    ameta = all_meta[aid]
                    extra.append({
                        "id": aid,
                        "text": ameta.get("text", ""),
                        "score": round(r["score"] * 0.8, 4),
                        "semantic_score": None,
                        "time_score": None,
                        "frequency": int(ameta.get("frequency", 0)),
                        "importance": float(ameta.get("importance", 0.5)),
                        "created_at": ameta.get("created_at", ""),
                        "last_accessed": ameta.get("last_accessed", ""),
                        "emotion": ameta.get("emotion", None),
                        "sentiment": float(ameta.get("sentiment", 0.0)) if ameta.get("sentiment") is not None else None,
                        "intensity": float(ameta.get("intensity", 0.0)) if ameta.get("intensity") is not None else None,
                        "related_ids": [i for i in ameta.get("related_ids", "").split(",") if i],
                        "via_association": True,
                    })
            # Associations are appended beyond top_k — not competing with direct results
            results = results + sorted(extra, key=lambda x: x["score"], reverse=True)

        # Update frequency and last_accessed
        all_items = self._storage.get_all()
        meta_map = {item["id"]: item["metadata"] for item in all_items}
        for r in results:
            if r["id"] in meta_map and not r.get("via_association"):
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
        """Delete memories satisfying ALL three conditions:
        not accessed for `days` days AND frequency < max_frequency AND importance < max_importance.
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
