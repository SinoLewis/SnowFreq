#🎯 PART 4: Persona-Aware Weighting
PERSONA_WEIGHTS = {
    "gamer": {
        "performance": 0.40,
        "display": 0.25,
        "battery": 0.20,
        "camera": 0.05,
        "build": 0.05,
        "connectivity": 0.05
    },
    "business": {
        "performance": 0.20,
        "battery": 0.25,
        "build": 0.20,
        "camera": 0.15,
        "connectivity": 0.10,
        "display": 0.10
    }
}

def persona_score(scores, persona):
    weights = PERSONA_WEIGHTS[persona]
    return round(sum(scores[k] * weights[k] for k in weights), 2)
