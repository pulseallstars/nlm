from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        # get_embedding_dimension is the new name (sentence-transformers ≥3.0);
        # fall back to the deprecated alias for older installations.
        try:
            self.dim = int(self._model.get_embedding_dimension())
        except AttributeError:
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_many(self, texts: list) -> list:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
