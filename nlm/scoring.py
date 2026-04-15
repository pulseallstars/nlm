from nlm.utils import time_decay, frequency_score


def nlm_score(
    semantic: float,
    time_val: float,
    freq_val: float,
    importance: float,
    weights: dict,
) -> float:
    return (
        weights["semantic"] * semantic
        + weights["time"] * time_val
        + weights["frequency"] * freq_val
        + weights["importance"] * importance
    )


def rerank(candidates: list, weights: dict, half_life_days: float) -> list:
    """
    Rerank ChromaDB candidates using the NLM formula.
    Each candidate: {id, metadata, distance}
    Returns sorted list with nlm_score and component scores added.
    """
    results = []
    for c in candidates:
        meta = c["metadata"]

        # Cosine distance [0..2] → similarity [0..1]
        semantic = max(0.0, 1.0 - c["distance"] / 2.0)

        time_val = time_decay(meta.get("created_at", ""), half_life_days)
        freq_val = frequency_score(int(meta.get("frequency", 0)))
        importance = float(meta.get("importance", 0.5))

        score = nlm_score(semantic, time_val, freq_val, importance, weights)

        results.append({
            "id": c["id"],
            "text": meta.get("text", ""),
            "score": round(score, 4),
            "semantic_score": round(semantic, 4),
            "time_score": round(time_val, 4),
            "frequency": int(meta.get("frequency", 0)),
            "importance": round(importance, 4),
            "created_at": meta.get("created_at", ""),
            "last_accessed": meta.get("last_accessed", ""),
            "emotion": meta.get("emotion", None),
            "sentiment": float(meta.get("sentiment", 0.0)) if meta.get("sentiment") is not None else None,
            "intensity": float(meta.get("intensity", 0.0)) if meta.get("intensity") is not None else None,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
