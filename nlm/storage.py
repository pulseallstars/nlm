import chromadb


class Storage:
    def __init__(self, collection_name: str, persist_path: str):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, id: str, embedding: list, metadata: dict):
        # ChromaDB metadata values must be str/int/float/bool
        safe_meta = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        self._col.add(ids=[id], embeddings=[embedding], metadatas=[safe_meta])

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

    def update_metadata(self, id: str, metadata: dict):
        safe_meta = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        self._col.update(ids=[id], metadatas=[safe_meta])

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

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        self._col.delete(ids=[item["id"] for item in self.get_all()])
