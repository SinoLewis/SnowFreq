# 🚀 PHASE 1 — Synthetic Data & Simulated Trading
# **(Layers 1–3 simulation)**

## 1️⃣ Dependencies
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
## 2️⃣ Synthetic Market Generator (Regime + Volatility)
def generate_market_data(
    n_bars=2000,
    seed=42
):
    np.random.seed(seed)

    regimes = np.random.choice([0, 1], size=n_bars, p=[0.6, 0.4])
    vol = np.where(regimes == 0, 0.005, 0.02)

    returns = np.random.normal(0, vol)
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "close": price,
        "returns": returns,
        "regime": regimes
    })

    return df
## 3️⃣ Indicator Engine (Layer 1)
def compute_indicators(df: pd.DataFrame):
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=10).mean()
    df["ema_slow"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["volatility"] = df["returns"].rolling(20).std()

    return df.dropna()
## 4️⃣ Signal Engineering (Layer 2)
def engineer_signals(df: pd.DataFrame):
    df = df.copy()

    df["ema_signal"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    df["ema_z"] = (df["ema_signal"] - df["ema_signal"].rolling(50).mean()) / \
                  (df["ema_signal"].rolling(50).std() + 1e-9)

    df["rsi_signal"] = (df["rsi"] - 50) / 50

    df["signal"] = (
        0.5 * df["ema_z"] +
        0.5 * df["rsi_signal"]
    )

    return df.dropna()
## 5️⃣ Strategy Logic & Simulated Execution (Layer 3)

def run_strategy(df: pd.DataFrame, threshold=0.5):
    df = df.copy()

    df["position"] = np.where(
        df["signal"] > threshold, 1,
        np.where(df["signal"] < -threshold, -1, 0)
    )

    df["trade"] = df["position"].diff().fillna(0)
    df["pnl"] = df["position"].shift(1) * df["returns"]

    return df
## 6️⃣ Phase 1 Artifact Builder
class StrategyArtifact:
    strategy_id: str
    trades: pd.DataFrame
    features: pd.DataFrame
    regimes: pd.Series
    execution_times: pd.Series


def build_phase1_artifact():
    market = generate_market_data()
    indicators = compute_indicators(market)
    signals = engineer_signals(indicators)
    trades = run_strategy(signals)

    exec_times = pd.Series(
        np.random.uniform(0.0005, 0.003, len(trades)),
        index=trades.index
    )

    return StrategyArtifact(
        strategy_id="synthetic_strategy_v1",
        trades=trades,
        features=signals[["ema_z", "rsi_signal"]],
        regimes=signals["regime"],
        execution_times=exec_times
    )

# ⚙️ PHASE 2 — Scoring Engine
# **(Layers 5–6 core logic)**

## 7️⃣ Metric Computation (Layer 5 inputs)
def sharpe_ratio(pnl):
    return pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252)

def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()

## 8️⃣ Dimension Scorers (Layer 6)

def predictive_quality(hit_rate):
    return hit_rate

def risk_adjusted_score(sharpe, max_dd):
    return sharpe / (1 + abs(max_dd))

def regime_robustness(pnl, regimes):
    perf = pnl.groupby(regimes).mean()
    return 1 / (1 + perf.std())

def parameter_stability(mock_std=0.2):
    return 1 / (1 + mock_std)

def execution_efficiency(exec_times):
    return 1 / (1 + exec_times.mean())

def complexity_penalty(big_o="O(n)"):
    return 1.0 if big_o == "O(n)" else 0.5
## 9️⃣ Unified Scoring Engine
WEIGHTS = {
    "predictive": 0.25,
    "risk": 0.25,
    "robustness": 0.20,
    "stability": 0.15,
    "execution": 0.10,
    "complexity": 0.05
}
def unified_score(scores: Dict[str, float]):
    return sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
## 🔟 Gating Logic
def promotion_gate(score):
    if score >= 0.75:
        return "PROMOTE"
    elif score >= 0.6:
        return "SANDBOX"
    return "REJECT"
## 1️⃣1️⃣ Phase 2 Orchestrator
def evaluate_strategy(artifact: StrategyArtifact):
    trades = artifact.trades

    equity = trades["pnl"].cumsum()
    sharpe = sharpe_ratio(trades["pnl"])
    mdd = max_drawdown(equity)

    hit_rate = (trades["pnl"] > 0).mean()

    scores = {
        "predictive": predictive_quality(hit_rate),
        "risk": risk_adjusted_score(sharpe, mdd),
        "robustness": regime_robustness(trades["pnl"], artifact.regimes),
        "stability": parameter_stability(),
        "execution": execution_efficiency(artifact.execution_times),
        "complexity": complexity_penalty("O(n)")
    }

    final_score = unified_score(scores)
    decision = promotion_gate(final_score)

    return final_score, decision, scores
## ✅ End-to-End Run
artifact = build_phase1_artifact()
score, decision, breakdown = evaluate_strategy(artifact)

print("FINAL SCORE:", round(score, 3))
print("DECISION:", decision)
print("BREAKDOWN:", breakdown)
