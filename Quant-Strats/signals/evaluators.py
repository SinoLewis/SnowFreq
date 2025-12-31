import numpy as np
import pandas as pd

def hit_rate(signal: pd.Series, returns: pd.Series):
    aligned = signal.shift(1) * returns
    return (aligned > 0).mean()

def entropy(signal: pd.Series):
    probs = signal.value_counts(normalize=True)
    return -np.sum(probs * np.log(probs + 1e-9))
