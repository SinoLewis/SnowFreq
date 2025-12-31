import numpy as np

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return ((equity - peak) / peak).min()
