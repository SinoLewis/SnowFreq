Development procedures to achieve SnowFreq Quant modules.

> Main Modules:

1. Options Trading
2. Portfolio Optimizers
3. Risk Metrics

## 1. Options Trading

## 2. Portfolio Optimizers

[PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/index.html) is the main package for optimzation problems.

Main features that we can leverage from the package includes:

### A) Expected Returns

Mean-variance optimization requires knowledge of the expected returns. In practice, these are rather difficult to know with any certainty. 
Thus the best we can do is to come up with estimates, for example by extrapolating historical data,

### B) Risk Models

We need some way of quantifying asset risk. The most commonly-used risk model is the covariance matrix, which describes asset volatilities and their co-dependence. 
This is important because one of the principles of diversification is that risk can be reduced by making many uncorrelated bets (correlation is just normalised covariance).

Types of Risk Models

- sample_cov
- semicovariance
- exp_cov
- ledoit_wolf
- ledoit_wolf_constant_variance
- ledoit_wolf_single_factor
- ledoit_wolf_constant_correlation
- oracle_approximating

> NB.

Supplying expected returns can do more harm than good. If predicting stock returns were as easy as calculating the mean historical return, we’d all be rich! 
For most use-cases, I would suggest that you focus your efforts on choosing an appropriate risk model.

### C) Optimization

Mathematical optimization is a very difficult problem in general, particularly when we are dealing with complex objectives and constraints. 
However, convex optimization problems are a well-understood class of problems, which happen to be incredibly useful for finance.

- Efficient Frontier Optimizers
  - Mean-Variance 
  - Semivariance
  - CVaR
  - CDaR
- Black-Litterman Model
- Hierachal Risk parity

## 3. Risk Metrics
