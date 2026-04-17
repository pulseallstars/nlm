import warnings

import chromadb


_PRIMITIVE = (str, int, float, bool)


def _sanitize_meta(metadata: dict) -> dict:
    """Return metadata with only ChromaDB-compatible primitive values.

    Warns once per call if any keys were dropped so callers notice silent
    coercion of lists/dicts/None into oblivion.
    """
    safe = {}
    dropped = []
    for k, v in metadata.items():
        if isinstance(v, _PRIMITIVE):
            safe[k] = v
        else:
            dropped.append(k)
    if dropped:
        warnings.warn(
            f"NLM Storage dropped non-primitive metadata keys: {dropped}. "
            "ChromaDB only accepts str/int/float/bool.",
            UserWarning,
            stacklevel=3,
        )
    return safe


class Storage:
    def __init__(
        self,
        collection_name: str,
        persist_path: str,
        embedding_dim: int = None,
        embedding_model: str = None,
    ):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection_name = collection_name

        desired_meta = {"hnsw:space": "cosine"}
        if embedding_dim is not None:
            desired_meta["embedding_dim"] = int(embedding_dim)
        if embedding_model is not None:
            desired_meta["embedding_model"] = str(embedding_model)

        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata=desired_meta,
        )

        self._guard_embedding_compatibility(embedding_dim, embedding_model)

    def _guard_embedding_compatibility(self, dim, model):
        """Raise if an existing collection was built with a different model/dim."""
        existing = self._col.metadata or {}
        if dim is not None and "embedding_dim" in existing:
            if int(existing["embedding_dim"]) != int(dim):
                raise ValueError(
                    f"Collection '{self._collection_name}' was created with "
                    f"embedding_dim={existing['embedding_dim']} but caller "
                    f"provided {dim}. Use a different collection_name or "
                    f"clear the persist_path."
                )
        if model is not None and "embedding_model" in existing:
            if str(existing["embedding_model"]) != str(model):
                raise ValueError(
                    f"Collection '{self._collection_name}' was created with "
                    f"embedding_model='{existing['embedding_model']}' but "
                    f"caller provided '{model}'. Mixing models produces "
                    f"incompatible embeddings."
                )

    def add(self, id: str, embedding: list, metadata: dict):
        self._col.add(ids=[id], embeddings=[embedding], metadatas=[_sanitize_meta(metadata)])

    def add_many(self, ids: list, embeddings: list, metadatas: list):
        safe = [_sanitize_meta(m) for m in metadatas]
        self._col.add(ids=ids, embeddings=embeddings, metadatas=safe)

    def query(self, embedding: list, n_results: int) -> list:
        count = self._col.count()
        if count == 0:
            return []
        n = min(n_results, count)
        result = self._col.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )
        items = []
        for i, id_ in enumerate(result["ids"][0]):
            items.append({
                "id": id_,
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],  # cosine distance [0..2]
            })
        return items

    def get_by_ids(self, ids: list) -> dict:
        """Bulk fetch of metadatas for a set of ids. Returns {id: metadata}."""
        if not ids:
            return {}
        result = self._col.get(ids=ids, include=["metadatas"])
        return {id_: result["metadatas"][i] for i, id_ in enumerate(result["ids"])}

    def update_metadata(self, id: str, metadata: dict):
        self._col.update(ids=[id], metadatas=[_sanitize_meta(metadata)])

    def update_many(self, ids: list, metadatas: list):
        safe = [_sanitize_meta(m) for m in metadatas]
        self._col.update(ids=ids, metadatas=safe)

    def delete(self, id: str):
        self._col.delete(ids=[id])

    def get_all(self) -> list:
        count = self._col.count()
        if count == 0:
            return []
        result = self._col.get(include=["metadatas"])
        items = []
        for i, id_ in enumerate(result["ids"]):
            items.append({"id": id_, "metadata": result["metadatas"][i]})
        return items

    def export_all(self) -> list:
        """Full dump including embeddings — used by snapshot export."""
        count = self._col.count()
        if count == 0:
            return []
        result = self._col.get(include=["metadatas", "embeddings"])
        items = []
        for i, id_ in enumerate(result["ids"]):
            items.append({
                "id": id_,
                "metadata": result["metadatas"][i],
                "embedding": list(result["embeddings"][i]),
            })
        return items

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        """Drop the collection entirely and recreate with the same settings.

        O(1) in the client instead of loading every id into RAM.
        """
        existing_meta = dict(self._col.metadata or {})
        self._client.delete_collection(self._collection_name)
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=existing_meta or {"hnsw:space": "cosine"},
        )
