# OG Integrated Flow (Layer 1 → 3)
validate_ohlcv(df)

regime = volatility_regime(df["close"])

signal = RSISignal().generate(df)
returns = df["close"].pct_change()

print("Hit rate:", signal_hit_rate(signal, returns))
print("Entropy:", signal_entropy(signal))

hmm = RegimeHMM()
hmm.fit(returns.dropna().values)
states = hmm.predict(returns.dropna().values)

# Layer 1–3
signal = RSISignal().generate(df)
returns = df["close"].pct_change()

# Layer 4
positions = volatility_targeting(signal, returns)

# Layer 5
equity = run_trader(df["close"], positions)

# Layer 6
dd = max_drawdown(equity)
print("Max DD:", dd)
print("System:", system_metrics())

# Layer 7: End-to-End Strategy Evaluation
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
    "sharpe": (0, 2),
    "max_drawdown": (-0.5, 0),
    "hit_rate": (0.4, 0.7),
    "entropy": (0, 1.5),
    "turnover": (0, 6),
    "latency_ms": (0, 100),
    "stability": (0, 1),
}

from governance.scorer import score_strategy
from governance.gates import apply_gates
from governance.promotion import promotion_decision

score = score_strategy(metrics, bounds)
gates = apply_gates(metrics)
decision = promotion_decision(score, gates)

print("Score:", score)
print("Gate Failures:", gates)
print("Decision:", decision)
