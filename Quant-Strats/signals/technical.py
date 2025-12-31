import numpy as np
import pandas as pd
from .base import Signal

class RSISignal(Signal):
    def __init__(self, window=14, low=30, high=70):
        self.window = window
        self.low = low
        self.high = high

    def generate(self, df: pd.DataFrame):
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(self.window).mean()
        loss = -delta.clip(upper=0).rolling(self.window).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        return pd.Series(np.where(rsi < self.low, 1, np.where(rsi > self.high, -1, 0)), index=df.index)
