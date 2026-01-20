# scoring/composite.py
WEIGHTS = {
    "performance": 0.25,
    "display": 0.15,
    "camera": 0.20,
    "battery": 0.15,
    "build": 0.15,
    "connectivity": 0.10
}

def total_score(scores: dict):
    return round(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 2)
