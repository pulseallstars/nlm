from transformers import pipeline


class EmotionClassifier:
    POSITIVE = {"joy", "surprise"}
    NEGATIVE = {"sadness", "anger", "fear", "disgust"}

    def __init__(self):
        self._pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )

    def classify(self, text: str) -> dict:
        # Let the tokenizer truncate on token boundaries (max 512) rather than
        # chopping mid-word with text[:512].
        result = self._pipe(text, truncation=True, max_length=512)[0][0]
        label = result["label"]   # joy / sadness / anger / fear / surprise / disgust / neutral
        score = result["score"]   # confidence 0..1

        if label in self.POSITIVE:
            sentiment = score
        elif label in self.NEGATIVE:
            sentiment = -score
        else:
            sentiment = 0.0

        return {
            "emotion":   label,
            "sentiment": round(sentiment, 4),  # -1.0..1.0
            "intensity": round(score, 4),       # 0.0..1.0
        }
