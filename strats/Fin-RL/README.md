## FinRl Project Specs

### A. Portfolio Allocation

- Multi vs Single Assets Trading

### B. Portfolio Data

> Securities Category

1. Information Technology / Big Tech
2. Financials / Banks
3. Energy
4. Health Care
5. Communication Services
6. Crypto Currensies

> Factors to consider when choosing Securities

1. Related News
2. Mkt Performance

> Assets Data

- FinRL DataHandler: Stocks, Crypto
- Freq DataHandler: Crypto
- Baseline data

### B. Feature Engineering
> Features From Other Models

- NLP
- LSTM

> Data Tech Indicators

- M.A.C.D
- R.S.I

### C. Gym Env Init

- FinRl Envs
    - Crypto & Stock & PaperTrade Env
    - Portfolio Allocation Env ie. StockPortfolioEnv
    - Portfolio Optimization Env
- Anytrade Envs

### D. Portfolio Optimization

- cov_list
- M.V.O Weighted Portfolio
- Turbulence threshold
- EIIE (ensemble of identical independent evaluators)

### E. Agents Init

> DRL libraries

- StableBaselines

> Models Types

- DQN
- PPO
- A2C
- GPM: A graph convolutional network

### F. Bactest & Performance Reports

1. Prediction & Performance Metrics:
    1. Model Stats
        - Annual return
        - Annual volatility
        - Sharpe ratio
        - Max drawdown
        - Calmar ratio
        - equal_cum_ret, mean_var_cum_ret, retail_cum_ret 4rm `Weight_Initialization.ipynb`
        - Multi-Window return
            - df_daily_return: df.cumprod
            - multi_performance_score deriverd from meta_score_coef 4rm `FinRL_PortfolioAllocation_Explainable_DRL.ipynb`
        - Multi-Crypto
            - equal_returns: equal_weight_values
        - Opt
            - mvo & weights
    2. Model Performance
        - Rewards
        - actions
        - asset_memory & action_memory
        - MSE
    3. Backtest: 
        - account_value 
1. Benchmark of FinRL BT
    1. Plots of Performance Metrics of Test Models
    2. Plots of M.V.O Opt
1. Benchmark of each Asset BT performance
    1. Combined indexed Returns of Portfolio Assets
    2. Each Portfolio Assets returns
1. Benchmark new Feature Eng
1. Benchmark Envs
1. Benchmark of Hyper Params

### G. FineTuning

- Project Opt 
    - GPU procs: Replay Memory on GPU
    - Local virt env
- Top Benchmarked Objects
- Research Target Params
- Reward Calc & Actions No.s
- Hyper Params
- Weights & Biases
- Derivative Hedging
- Exploration & Expoitation

### H. Imitation Learning