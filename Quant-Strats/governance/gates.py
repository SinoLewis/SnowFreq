def gates(metrics):
    return metrics.get("max_drawdown", 0) > -0.3
