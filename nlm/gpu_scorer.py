# Optional GPU importance scorer (placeholder for v0.2)
# Replace with RWKV-7 World 0.1B State Tuned for Pulses


class GPUScorer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: load model here

    def score(self, text: str) -> float:
        # TODO: run inference, return importance [0..1]
        raise NotImplementedError("GPU scorer not yet implemented")
