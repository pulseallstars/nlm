import math
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def time_decay(created_at: str, half_life_days: float = 90) -> float:
    """Exponential decay: 1.0 when fresh, ~0.5 after half_life_days."""
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created
        days = delta.total_seconds() / 86400
        return math.exp(-math.log(2) / half_life_days * days)
    except Exception:
        return 0.5


def frequency_score(count: int, max_freq: int = 100) -> float:
    """Logarithmic frequency: rare=low, common=high, capped at 1.0."""
    if count <= 0:
        return 0.0
    return min(1.0, math.log(1 + count) / math.log(1 + max_freq))


def specificity_score(text: str) -> float:
    """CPU importance heuristic: numbers + proper nouns + length sweet spot."""
    if not text or not text.strip():
        return 0.0

    words = text.split()
    score = 0.0

    # Numbers indicate specific facts (dates, ages, values)
    if any(any(c.isdigit() for c in w) for w in words):
        score += 0.3

    # Proper nouns (capitalized words after the first)
    proper = sum(1 for w in words[1:] if w and w[0].isupper())
    score += min(0.4, proper * 0.1)

    # Golden length zone: not too short, not too long
    if 5 <= len(words) <= 50:
        score += 0.3

    return min(1.0, score)
