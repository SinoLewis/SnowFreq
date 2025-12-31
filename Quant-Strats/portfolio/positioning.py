import pandas as pd

def volatility_targeting(signal, returns, target_vol=0.02):
    vol = returns.rolling(20).std()
    return signal * (target_vol / (vol + 1e-8)).clip(0,1)
