# ===============================
# FULL QUANT INDICATOR APP
# ===============================

import numpy as np
import pandas as pd

import ta
import pandas_ta as pta
import vectorbt as vbt

# -------------------------------
# 1. SYNTHETIC DATA GENERATION
# -------------------------------
np.random.seed(42)

N = 500  # bars
dates = pd.date_range("2024-01-01", periods=N, freq="H")

price = 100 + np.cumsum(np.random.normal(0, 0.3, N))
high = price + np.random.uniform(0.1, 0.6, N)
low = price - np.random.uniform(0.1, 0.6, N)
open_ = price + np.random.normal(0, 0.1, N)
volume = np.random.randint(100, 2000, N)

df = pd.DataFrame({
    "open": open_,
    "high": high,
    "low": low,
    "close": price,
    "volume": volume
}, index=dates)

# ======================================================
# 2️⃣ TREND & MOMENTUM STRATEGIES
# ======================================================

# SMA / EMA Cross
df["SMA_fast"] = ta.trend.SMAIndicator(df["close"], 20).sma_indicator()
df["SMA_slow"] = ta.trend.SMAIndicator(df["close"], 50).sma_indicator()
df["EMA_fast"] = ta.trend.EMAIndicator(df["close"], 12).ema_indicator()
df["EMA_slow"] = ta.trend.EMAIndicator(df["close"], 26).ema_indicator()

df["EMA_Cross"] = np.where(df["EMA_fast"] > df["EMA_slow"], 1, -1)

# Triple MA
df["MA_10"] = ta.trend.EMAIndicator(df["close"], 10).ema_indicator()
df["MA_30"] = ta.trend.EMAIndicator(df["close"], 30).ema_indicator()
df["MA_100"] = ta.trend.EMAIndicator(df["close"], 100).ema_indicator()

df["Triple_MA"] = np.where(
    (df["MA_10"] > df["MA_30"]) & (df["MA_30"] > df["MA_100"]), 1,
    np.where((df["MA_10"] < df["MA_30"]) & (df["MA_30"] < df["MA_100"]), -1, 0)
)

# Adaptive & Advanced MAs
df["KAMA"] = pta.kama(df["close"])
df["ZLEMA"] = pta.zlma(df["close"])
df["HMA"] = pta.hma(df["close"])
# df["FRAMA"] = pta.frama(df["close"])

# Momentum
df["ROC"] = ta.momentum.ROCIndicator(df["close"], 12).roc()
df["Log_Return"] = np.log(df["close"] / df["close"].shift(1))
df["Log_Momentum"] = df["Log_Return"].rolling(20).sum()

atr = ta.volatility.AverageTrueRange(
    df["high"], df["low"], df["close"]
).average_true_range()

df["Vol_Adj_Momentum"] = df["ROC"] / atr

# ======================================================
# 3️⃣ VOLATILITY-BASED STRATEGIES
# ======================================================

# ATR Breakout
df["ATR_Upper"] = df["close"] + 2 * atr
df["ATR_Lower"] = df["close"] - 2 * atr

df["ATR_Breakout"] = np.where(
    df["close"] > df["ATR_Upper"], 1,
    np.where(df["close"] < df["ATR_Lower"], -1, 0)
)

# Bollinger Squeeze
bb = ta.volatility.BollingerBands(df["close"])
df["BB_Width"] = bb.bollinger_wband()
df["BB_Squeeze"] = df["BB_Width"] < df["BB_Width"].rolling(100).quantile(0.1)

# Donchian Channel
dc = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"])
df["Donchian_Upper"] = dc.donchian_channel_hband()
df["Donchian_Lower"] = dc.donchian_channel_lband()

df["Donchian_Breakout"] = np.where(
    df["close"] > df["Donchian_Upper"], 1,
    np.where(df["close"] < df["Donchian_Lower"], -1, 0)
)

# Advanced Volatility
# df["Parkinson_Vol"] = pta.parkinson(df["high"], df["low"])
# df["GK_Vol"] = pta.garman_klass(
#     df["open"], df["high"], df["low"], df["close"]
# )

# ======================================================
# 4️⃣ MARKET MICROSTRUCTURE (INTRADAY)
# ======================================================

# VWAP Deviation
df["VWAP"] = pta.vwap(
    df["high"], df["low"], df["close"], df["volume"]
)
df["VWAP_Dev"] = (df["close"] - df["VWAP"]) / df["VWAP"]

# Order Flow Imbalance (Proxy)
df["Signed_Vol"] = np.sign(df["close"].diff()) * df["volume"]
df["OFI"] = df["Signed_Vol"].rolling(20).sum() / df["volume"].rolling(20).sum()

# Volume-weighted Momentum
df["VW_Momentum"] = (
    df["close"].pct_change(10) *
    df["volume"] / df["volume"].rolling(10).mean()
)

# Volume Spikes
# df["Vol_Z"] = vbt.zscore(df["volume"], window=50)
# df["Volume_Spike"] = df["Vol_Z"] > 2

# Tick Imbalance Proxy
df["Up"] = df["close"] > df["close"].shift()
df["Down"] = df["close"] < df["close"].shift()
df["Tick_Imbalance"] = (
    df["Up"].rolling(20).sum() -
    df["Down"].rolling(20).sum()
) / 20

# ======================================================
# 5️⃣ REGIME FILTERS (NON-ALPHA CONTEXT)
# ============================sss==========================

# Trend Strength
df["ADX"] = ta.trend.ADXIndicator(
    df["high"], df["low"], df["close"]
).adx()
df["Trending_Regime"] = df["ADX"] > 25

# Volatility Regime
df["Rolling_Vol"] = df["close"].pct_change().rolling(30).std()
df["High_Vol_Regime"] = (
    df["Rolling_Vol"] > df["Rolling_Vol"].rolling(100).median()
)

# ======================================================
# 6️⃣ FINAL EXECUTION GATES
# ======================================================

df["ALLOW_TREND"] = (
    df["Trending_Regime"] &
    df["High_Vol_Regime"] &
    (df["EMA_Cross"] == 1)
)

df["ALLOW_MEAN_REVERT"] = (
    (~df["Trending_Regime"]) &
    (~df["High_Vol_Regime"]) &
    (df["VWAP_Dev"].abs() > 0.01)
)

# ======================================================
# 7️⃣ OUTPUT CHECK
# ======================================================
print("\n=== FINAL DATA SNAPSHOT ===")
print(df.tail(5))

print("\nSignals Summary:")
print(df[[
    "EMA_Cross",
    "ATR_Breakout",
    "Donchian_Breakout",
    "ALLOW_TREND",
    "ALLOW_MEAN_REVERT"
]].tail())
