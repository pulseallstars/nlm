from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list:
        return self._model.encode(text, normalize_embeddings=True).tolist()
