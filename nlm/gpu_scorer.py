import torch
from transformers import pipeline


class GPUScorer:
    """
    Scores memory importance using zero-shot classification.

    Default model: typeform/distilbert-base-uncased-mnli (~260MB)
    Works on CPU and GPU. Auto-detects CUDA.

    When RWKV-7 0.1B is trained on Pulses data — pass its path as model_path.
    """

    IMPORTANT_LABEL = "important fact worth remembering"
    TRIVIAL_LABEL = "trivial or unimportant content"

    def __init__(self, model_path: str = "typeform/distilbert-base-uncased-mnli"):
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_path,
            device=device,
        )

    def score(self, text: str) -> float:
        """Return importance score [0..1]. Higher = more worth remembering."""
        result = self._pipe(
            text[:512],
            candidate_labels=[self.IMPORTANT_LABEL, self.TRIVIAL_LABEL],
        )
        idx = result["labels"].index(self.IMPORTANT_LABEL)
        return round(result["scores"][idx], 4)
