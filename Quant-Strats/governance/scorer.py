def score(metrics, weights):
    return sum(metrics[k] * w for k, w in weights.items() if k in metrics)
