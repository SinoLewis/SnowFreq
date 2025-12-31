# ==============================
# Quant-Strat: End-to-End Flow
# Layers 1 → 7
# ==============================

# ---- Imports ----
from data.validators import validate_ohlcv
from data.regimes import volatility_regime

from signals.technical import RSISignal
from signals.evaluators import signal_hit_rate, signal_entropy

from models.hmm import RegimeHMM

from portfolio.positioning import volatility_targeting
from portfolio.risk import max_drawdown

from trader.integration import run_trader

from monitoring.system import system_metrics

from governance.scorer import score_strategy
from governance.gates import apply_gates
from governance.promotion import promotion_decision


# ==============================
# Layer 1 — Data Validation & Regime
# ==============================

validate_ohlcv(df)

regime = volatility_regime(df["close"])


# ==============================
# Layer 2 — Signal Generation & Evaluation
# ==============================

signal = RSISignal().generate(df)
returns = df["close"].pct_change()

print("Hit rate:", signal_hit_rate(signal, returns))
print("Entropy:", signal_entropy(signal))


# ==============================
# Layer 3 — Regime Modeling (HMM)
# ==============================

returns_clean = returns.dropna().values.reshape(-1, 1)

hmm = RegimeHMM()
hmm.fit(returns_clean)
states = hmm.predict(returns_clean)


# ==============================
# Layer 4 — Portfolio Construction
# ==============================

positions = volatility_targeting(signal, returns)


# ==============================
# Layer 5 — Trader Execution
# ==============================

equity_curve = run_trader(df["close"], positions)


# ==============================
# Layer 6 — Risk & System Metrics
# ==============================

dd = max_drawdown(equity_curve)

print("Max Drawdown:", dd)
print("System Metrics:", system_metrics())


# ==============================
# Layer 7 — Scoring, Gating & Promotion
# ==============================

metrics = {
    "sharpe": 1.35,
    "max_drawdown": -0.18,
    "hit_rate": 0.56,
    "entropy": 0.82,
    "turnover": 2.1,
    "latency_ms": 12,
    "stability": 0.75,
}

bounds = {
    "sharpe": (0.0, 2.0),
    "max_drawdown": (-0.5, 0.0),
    "hit_rate": (0.4, 0.7),
    "entropy": (0.0, 1.5),
    "turnover": (0.0, 6.0),
    "latency_ms": (0.0, 100.0),
    "stability": (0.0, 1.0),
}

score = score_strategy(metrics, bounds)
gate_failures = apply_gates(metrics)
decision = promotion_decision(score, gate_failures)

print("Strategy Score:", score)
print("Gate Failures:", gate_failures)
print("Final Decision:", decision)
