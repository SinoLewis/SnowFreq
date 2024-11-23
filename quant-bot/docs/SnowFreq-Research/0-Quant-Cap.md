# Title Page

> Can Algoritmic Pattern trading produce a better portfolio against Global Indexes.

# Abstract

- There is a distinct social, modern approach towards of being self-relliant. This is without the help of the collosal titans of the industry, this being the bank & the hedge-funds around the world.

# Introduction

## What is Finance Trading?

- Finance Trading represent our system that assumes role of Portfolio Manager to algorithmic analyze timeseries market information to generate actions of our portfolio asset position sizing.

## Evolution of Finance trading.

- Financing has been part of human society since as early as 300AD in the roman empire.

### Banks & Hedge Funds Finance Trading.

- Hedge Funds granted their huge resources can gain great advantage over personal finance in the following ways

1. Complex trading systems beyond power of personal desktops.
2. Large compute power that can exponentially execute many Programming opperations.

- Besides this one may desire to challenge these titans, because these programs with time become more readily available 

### Personal Finance Trading

## Thesis statement

## Overview of Finance Trading

# Literature Review

# Methodology

## Quant Analsis Framework for SnowFreq Portfolio (ChatGPT + Investopedia + DataCamp articles)

The Big 6 modules facilitating the core aspect of SnowFreq Portfolio Investing ie. Quant Analsis for Trade signals & Final report

### 1. **Overview of the Portfolio Strategy**

The project consist of three app execution state for **Stock Selection,** **Quant Backtesting**, **Live Portfolio Monitoring**.

- Chapter 2 is the first app execution for **Stock Selection** of our Portfolio.
- Chapter 3-5 is the second app execution for  **Quant Backtesting** our Strategy Stock data.
- Chapter 6, the 3rd app Live server for trading the actual assets with real-world funding.

The foundation of my strategy is built on a **quantitative, data-driven approach** that seeks to maximize risk-adjusted returns. The portfolio is diversified across sectors and geographies, but the core of stock selection and asset allocation is based on a combination of **statistical analysis**, **factor models**, and **risk management**. 

---

### 2. **Stock Selection Process**

#### a) **Data-Driven Stock Screening**
I begin with a large universe of stocks, narrowing down candidates through rigorous screening based on fundamental, technical, and alternative data. Here’s how I approach stock selection:
   - **Fundamental Data**: Stocks are screened based on factors such as earnings growth, revenue, cash flow, and debt levels.
   - **Technical Indicators**: I use momentum indicators like **relative strength index (RSI)** and **moving averages** to assess market trends and timing.
   - **Alternative Data**: Social sentiment, web traffic, and industry-specific data sources are incorporated to give an edge in stock prediction.

#### b) **Factor Models (Fama-French)**
The stock selection process is deeply rooted in factor models that drive excess returns. I use the **Fama-French multi-factor model** to identify which stocks are positioned to outperform based on:
   - **Market (Beta)**: Exposure to broad market movements.
   - **Size (SMB)**: Small-cap stocks historically outperform large-cap stocks.
   - **Value (HML)**: High book-to-market value stocks tend to outperform growth stocks.

#### c) **Factor Tilt**:
- I overweight **small-cap stocks** with strong fundamentals and value characteristics when market conditions are favorable for risk-on strategies.
- In a more defensive environment, I shift toward **large-cap, low-beta stocks** that provide stability and steady returns.

---

### 3. **Quantitative Analysis Methods Used**

#### a) **Correlation and Covariance Analysis**
I use **correlation analysis** to assess how different stocks and asset classes move relative to each other. This helps build a well-diversified portfolio where risk is minimized through low-correlation or negative-correlation assets.

- **Example**: A stock in the technology sector might be highly correlated with other tech stocks, but less correlated with utility or consumer staple stocks. By including stocks from multiple sectors, I ensure that portfolio volatility is reduced.

#### b) **Volatility Modeling and Risk Adjustments**
I model the volatility of individual assets and the portfolio as a whole to ensure I’m staying within acceptable risk limits. Using **GARCH models**, I predict future volatility based on historical price movements, allowing for dynamic adjustments to portfolio weightings.
   - **Dynamic Rebalancing**: High-volatility assets are assigned smaller weightings, while low-volatility, stable assets receive more capital allocation. This keeps portfolio risk in check without sacrificing upside potential.

#### c) **Factor Analysis and Multi-Factor Optimization**
Beyond traditional factors (e.g., value, size), I also integrate **momentum** and **quality factors**. Here’s how:
   - **Momentum**: Stocks with strong recent performance tend to continue to outperform in the short-term. I rank stocks based on 3-month and 6-month momentum indicators and adjust weightings.
   - **Quality**: I focus on firms with low debt, high return on equity (ROE), and consistent earnings growth. Stocks with high **quality scores** get a larger weighting in the portfolio.

#### d) **Backtesting and Machine Learning**
My strategy is constantly evolving, backed by historical **backtesting** and machine learning models:
   - **Backtesting**: I test various model parameters using historical data to ensure the strategy would have performed well in different market regimes, including recessions and bull markets.
   - **Machine Learning**: I implement algorithms like **random forests** and **gradient boosting** to predict stock price movements based on historical patterns and non-linear relationships.

---

### 4. **Risk Management**

#### a) **Value at Risk (VaR)**
I calculate the portfolio's **Value at Risk (VaR)** daily, ensuring that potential losses remain within acceptable bounds at a given confidence level (e.g., 95%). This measure helps the firm quantify potential downside risks.

- **For instance**: If the VaR for the portfolio is $1M at 95%, this means there is a 5% chance that the portfolio could lose more than $1M on a given day.

#### b) **Expected Shortfall (ES)**
I complement VaR with **Expected Shortfall (ES)**, which measures the expected loss in extreme scenarios where losses exceed VaR. This provides a more accurate picture of the risks in turbulent markets.

#### c) **Hedging and Protection**
To protect the portfolio from sharp drawdowns:
   - **Options and Derivatives**: I use options for downside protection, especially for high-beta stocks. For instance, buying put options on volatile stocks can limit potential losses.
   - **Dynamic Hedging**: I dynamically adjust exposure using futures or inverse ETFs to hedge during periods of expected market decline.

#### d) **Stress Testing**
The portfolio is regularly stress-tested against macroeconomic shocks, such as interest rate hikes, geopolitical events, or market crashes. This ensures the portfolio is resilient across multiple scenarios.

---

### 5. **Capital Allocation and Diversification**

#### a) **Sector and Asset Class Limits**
- **Sector Exposure**: I set exposure limits to each sector (e.g., no more than 20% in technology) to ensure diversification and to reduce the risk of over-concentration in any single sector.
- **Asset Allocation**: I maintain strict limits on individual stock positions (e.g., no more than 5% in a single stock) to avoid concentration risk.

#### b) **Liquidity Management**
- **Liquid Stocks**: I ensure that a portion of the portfolio is allocated to highly liquid, large-cap stocks to meet any potential liquidity needs.
- **Cash Allocation**: I also maintain a certain percentage of the portfolio in cash or near-cash equivalents (e.g., money market instruments) to take advantage of short-term opportunities or to weather market downturns.

---

### 6. **Performance Monitoring and Continuous Improvement**

- **Performance Metrics**: I track risk-adjusted metrics like **Sharpe Ratio**, **Sortino Ratio**, and **Alpha** to ensure the portfolio consistently outperforms relative to the risk taken.
- **Rebalancing**: I rebalance the portfolio periodically based on performance metrics and changing market conditions, ensuring it remains aligned with the overall strategy.
- **Continuous Learning**: The model is always learning. I regularly incorporate new data, adjust model parameters, and experiment with new algorithms to stay ahead of market changes.

---

### Conclusion:
This data-driven, multi-factor, and risk-managed approach ensures the portfolio is optimized to maximize risk-adjusted returns while protecting against significant downside risks. The strategy's success lies in its combination of **quantitative rigor**, **factor-based stock selection**, and **robust risk management**, allowing it to consistently outperform while maintaining stability.

### Example of Quant Strats (ChatGPT articles)

> Quant Technical Strats Table 

- Main goal: **Monitor & Maximize Returns**

Monitor: Server Logs(Live) w/ Lumibot-like Portofolio returns & Module Output(BT)
Maximize: Quant + Technical Strat signals, AI Strat predictions

| NO. | Quant Strategy               | Input Parameters         | Formula or Concept                                             |
| --- | ---------------------------- | ------------------------ | -------------------------------------------------------------- |
| 1   | Value at Risk (VaR)           | Returns                  | VaR α = Quantile α (Portfolio Returns)                         |
| 2   | Expected Shortfall (ES)       | Loss                     | ES α = E[Loss ∣ Loss > VaR α]                                  |
| 3   | Maximum Drawdown (MDD)        | Peak & Trough Value      | MDD = (Peak Value − Trough Value) / Peak Value                 |
| 4   | Fama-French Model             | Size, Value, Market      | R_p - R_f = α + β1(R_m - R_f) + β2(SMB) + β3(HML) + ε          |
| 5   | Factor Tilt                   | Factor Scores            | Adjust portfolio weightings toward desired factors (e.g., Value, Momentum) |
| 6   | GARCH Model                   | Returns, Volatility      | σ²_t = α_0 + α_1 ε²_(t-1) + β_1 σ²_(t-1)                      |
| 7   | Correlation Analysis          | Asset Returns            | Correlation = Cov(r_x, r_y) / (σ_x * σ_y)                      |
| 8   | Volatility Analysis           | Asset Returns            | Volatility = Standard Deviation of Returns, σ = sqrt(Var(r))    |
| 9   | Mean-Variance Optimization (MVO) | Returns, Risk, Covariance | Minimize portfolio variance: σ²_p = Σ Σ w_i w_j Cov(r_i, r_j) |
| 10  | Black-Litterman Model         | Market Equilibrium, Investor Views | Combines market-implied returns with investor views for optimal asset allocation |
| 11  | Risk Parity                   | Asset Weights, Volatility | Allocate based on equal risk contribution from each asset       |
| 12  | Sharpe Ratio                  | Returns, Risk-Free Rate, Volatility | Sharpe Ratio = (R_p - R_f) / σ_p                       |
| 13  | Sortino Ratio                 | Returns, Risk-Free Rate, Downside Deviation | Sortino Ratio = (R_p - R_f) / σ_d                   |
| 14  | Black-Scholes Model           | Stock Price, Strike Price, Volatility, Time to Maturity | Black-Scholes PDE: ∂V/∂t + 0.5σ²S² ∂²V/∂S² + rS∂V/∂S - rV = 0 |
| 15  | Delta Hedging                 | Option Price Sensitivity | Δ = ∂V/∂S, Adjusting asset holdings to neutralize option risk  |

> Quant Strats time-delta ranges

- Main goal: **Calibrate & Optimize Time-Delta Range**

Here’s a high-efficiency, industry-standard table disclosing the time-delta ranges used for various Quant Strategies. 
These time-delta ranges are selected based on best practices in quantitative finance, ensuring accurate and reliable performance evaluations for each strategy.

| NO. | Quant Strategy            | Time-Delta Range                        | Reasoning and Usage                                              |
| --- | ------------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| 1   | Value at Risk (VaR)        | 1 day to 1 month                        | Typically computed for short-term horizons to measure potential daily/weekly losses in portfolios. |
| 2   | Expected Shortfall (ES)    | 1 day to 1 month                        | Used in tandem with VaR to assess tail risk for short-term extreme losses. |
| 3   | Volatility Analysis        | 1 day to 6 months                       | Typically applied to shorter time frames to capture recent volatility patterns, often in daily or monthly intervals. |
| 4   | GARCH Model                | 1 day to 6 months                       | GARCH is generally applied to short-term volatility modeling, but historical data is necessary for reliable parameter estimation. |
| 5   | Correlation Analysis       | 1 month to 5 years                      | Correlation matrices are often computed over a variety of periods; short-term for tactical adjustments, and long-term for strategic asset allocation. |
| 6   | Maximum Drawdown (MDD)     | 1 month to 1 year                       | MDD is usually observed over medium to long-term periods to capture significant downturns. |
| 7   | Factor Tilt                | 6 months to 5 years                     | Factor premiums like **Value** or **Momentum** require medium to long-term horizons to manifest reliably. |
| 8   | Mean-Variance Optimization | 1 year to 3 years                       | For optimal portfolio weights, MVO relies on historical returns and covariances, often calculated over medium to long-term horizons. |
| 9   | Fama-French Model          | 1 year to 5 years                       | Factors like **Size** and **Value** typically require long-term historical data for robust analysis. |

- Freqtrade offers the available dataframe to derive calcs for Quant Strats. Below object describes it.

### Q/A of Quant Analysis

> What Is the Difference Between Alpha and Beta in Finance?

Alpha measures how much an investment outperforms or underperforms a benchmark. Beta is a measurement of an investment’s volatility and is one measurement of an investment’s risk.


## Portfolio Investing (Investopedia + DataCamp articles)

### What Is a Financial Portfolio? (Investopedia articles)

## SnowFreq App SDLC (Overall Project App)

### 1. Bkg. Information 

The project is primarily based on python from the **freqtrade** & **lumibot** packages. We have our Technical mkt indicators that generate trade action signals while being monitored by Quant Technical indicators. The Quant signal will determine how we monitor & manage the portfolio. 

### 2. Sys Design & Wireframes

> I) **Stock Selection**
    
    A[OHCLV Timeframe Data] --> B[Freqtrade Strats ranks Stock Selection]
    B --> C[Weighted Asset Portfolio ]

> II) **Quant Backtesting**
    
    A[OHCLV Timeframe Data] --> B[Freqtrade Strats generates trade signals]
    B --> C[Quant Tech signals of Cummulative Portfolio Performance ]
    C --> D[Risk Mngt ]
    D --> E[Portfolio Capital & Returns ]

> III) **Live Portfolio Monitoring Mngt**
    
    A[OHCLV Timeframe Data] --> B[Freqtrade Strats generates trade signals]
    B --> C[Quant Tech signals of Cummulative Portfolio Performance ]
    C --> D[Portfolio Capital & Returns ]
    D --> E[Live Quant Overide TradeBot API ]

### 3. Code Implementation

> I) **Stock Selection**

1. Define test list of Quote assets  
2. Obtain Timeframe data for the list
3. Backtest run on Tech Strats on each list into strat-name + time-unique output folder
    1. RSI, ADX & Supertrend
4. Rank the Test(s) specified run based on Portfolio returns profits
5. Stress test different test Quant scenerarios with their ranks
6. Define a TradeBot config (Freqtrade) for final approved Asset Portfolio

> II) **Quant Backtesting**

1. Quantitative Analysis Methods Used
2. Risk Management
3. Portfolio Capital Allocation and Diversification & Returns

> III) **Live Portfolio Monitoring Mngt**

1. Same as Quant Backtesting but now an active server that monitors asset's OHLCV ticker data record.
2. Active Monitor dashboard chart of Quant Bot signals
3. Live Quant Trade sigs mngt Overide TradeBot API to our active server based Quant Bot signals

### 4. Testing

Designing an industry-standard testing framework for a **Quantitative Strategy (Quant Strat) application** involves ensuring the robustness, accuracy, and reliability of the quantitative strategies implemented, like **Value at Risk (VaR)**, **Expected Shortfall (ES)**, **Volatility Analysis**, and **GARCH Model**. The framework should not only validate the individual components but also ensure that the application behaves reliably in production environments. Here’s an outline of an effective testing framework for a Quant Strat app:

---

### **1. Key Components of the Testing Framework**

1. **Unit Testing**: 
   - **Purpose**: Validate the correctness of individual functions and modules, such as calculations for VaR, ES, and GARCH models.
   - **Frameworks**: Use industry-standard frameworks like `unittest` or `pytest` in Python, which support clear structure, modularity, and coverage reporting.
   - **Tests**:
     - Assert mathematical accuracy by comparing function outputs with known values.
     - Test edge cases (e.g., zero returns, very high volatility).
     - Include randomized inputs to test robustness.

2. **Integration Testing**:
   - **Purpose**: Ensure that the different components (e.g., data ingestion, calculation modules, and result reporting) work together correctly.
   - **Frameworks**: `pytest` combined with setup and teardown methods to simulate realistic data flows.
   - **Tests**:
     - Verify that the data pipeline correctly feeds historical price data into the strategy functions.
     - Ensure end-to-end calculations (e.g., VaR or ES) operate without data discrepancies.
     - Test integration of external libraries (e.g., `arch` for GARCH) with appropriate data handling.

3. **Data Validation Testing**:
   - **Purpose**: Ensure data integrity, consistency, and quality, which is essential for reliable quantitative analysis.
   - **Frameworks**: Use custom Python scripts or data validation libraries like `pandera` for DataFrame validation.
   - **Tests**:
     - Confirm data completeness (e.g., no missing values in critical columns like returns).
     - Validate column data types and ranges (e.g., dates are in chronological order).
     - Detect and handle outliers or erroneous data entries (e.g., extreme return values).

4. **Performance Testing**:
   - **Purpose**: Measure the application’s response time, memory consumption, and compute efficiency, especially for computationally intensive strategies like GARCH.
   - **Frameworks**: Use tools like `pytest-benchmark` for Python and `timeit` to measure performance of individual functions.
   - **Tests**:
     - Run stress tests on large datasets to ensure calculations for VaR, ES, and GARCH models meet latency requirements.
     - Measure resource utilization (CPU and memory) for scalability insights.
     - Establish performance baselines and detect regressions in algorithm efficiency over time.

5. **Backtesting and Simulation Testing**:
   - **Purpose**: Validate strategy performance on historical data and ensure risk metrics perform as expected.
   - **Frameworks**: Use backtesting libraries like `backtrader` or custom scripts to run simulations.
   - **Tests**:
     - Run historical backtests with expected outputs for comparison.
     - Simulate various market conditions (bullish, bearish, volatile) to test strategy resilience.
     - Check if risk metrics (e.g., VaR, ES) are accurately reported throughout the backtest.

6. **Statistical and Model Validation Testing**:
   - **Purpose**: Ensure that quantitative models produce statistically reliable and theoretically consistent results.
   - **Frameworks**: Use `scipy.stats` and `statsmodels` for statistical tests.
   - **Tests**:
     - Run statistical tests (e.g., Shapiro-Wilk test) on returns data to validate assumptions like normality for VaR.
     - Apply chi-square tests to ensure model fit in GARCH models.
     - Validate model outputs against benchmarks or expected distributions.

7. **Continuous Integration (CI) Testing**:
   - **Purpose**: Ensure the codebase remains stable and functional as new features or changes are introduced.
   - **Frameworks**: Use CI tools like **GitHub Actions**, **Jenkins**, or **Travis CI** to automate test execution upon code changes.
   - **Tests**:
     - Run the entire test suite automatically for every code push.
     - Generate coverage reports and monitor for declines in test coverage.
     - Run environment-specific tests to confirm cross-compatibility.

---

### **2. Structuring the Testing Framework**

- **Directory Structure**:
  ```
  /tests
  ├── unit_tests
  │   ├── test_var.py         # Unit tests for VaR calculations
  │   ├── test_es.py          # Unit tests for ES calculations
  │   └── test_garch.py       # Unit tests for GARCH model
  ├── integration_tests
  │   ├── test_data_pipeline.py
  │   └── test_data_handling.py
  ├── data_validation_tests
  │   └── test_data_quality.py
  ├── performance_tests
  │   └── test_performance.py
  ├── backtesting_tests
  │   └── test_backtesting.py
  └── ci_tests
      └── test_ci_pipeline.py
  ```

- **Configuration File**:
  - Use a configuration file (`pytest.ini` or `tox.ini`) to define parameters like logging verbosity, test file paths, and any environment-specific variables.

---

### **3. Sample Test Code for Each Component**

1. **Unit Test for VaR Calculation**:
   ```python
   import pytest
   import pandas as pd
   from optimize_reports import calculate_var

   def test_var_calculation():
       data = pd.DataFrame({'profit_ratio': [-0.02, 0.01, -0.015, 0.03, -0.005]})
       confidence_level = 0.95
       var = calculate_var(data, confidence_level=confidence_level)
       assert var < 0, "VaR should be a negative value indicating a loss."
   ```

2. **Integration Test for Data Handling**:
   ```python
   def test_data_pipeline_integration():
       df = load_data()  # Custom function to load a sample dataset
       assert 'profit_ratio' in df.columns, "Data should have 'profit_ratio' column"
       assert df.isnull().sum().sum() == 0, "Data should not have missing values"
   ```

3. **Performance Test for GARCH Calculation**:
   ```python
   import timeit
   from optimize_reports import calculate_garch_volatility

   def test_garch_performance(benchmark_data):
       execution_time = timeit.timeit(lambda: calculate_garch_volatility(benchmark_data), number=10)
       assert execution_time < 1.0, "GARCH model calculation should be optimized to run under 1 second."
   ```

4. **Statistical Test on Returns Normality for VaR Assumptions**:
   ```python
   from scipy.stats import shapiro

   def test_returns_normality():
       df = load_data()
       stat, p = shapiro(df['profit_ratio'])
       assert p > 0.05, "Returns data should be normally distributed for VaR assumptions"
   ```

5. **CI Test with Mock Data**:
   ```python
   def test_ci_integration():
       df = generate_mock_data()
       assert df.shape[0] > 100, "Mock data should have more than 100 rows for reliable CI testing"
   ```

---

### **4. Best Practices for the Testing Framework**

- **Coverage**: Aim for **90%+ test coverage** across modules, especially for core functions in quantitative strategies.
- **Parameterization**: Use **parameterized tests** (e.g., `@pytest.mark.parametrize`) to test functions with multiple configurations and confidence levels.
- **Mocking and Fixtures**: Use `pytest` fixtures and mocking libraries to simulate data inputs and streamline the setup process for complex tests.
- **Detailed Reporting**: Generate detailed reports (e.g., `pytest-html`) and track metrics like **pass/fail rates**, **test execution times**, and **performance benchmarks**.
- **CI Integration**: Integrate tests into a **CI pipeline** that triggers on every commit to ensure code stability and prevent regressions.
- **Documentation**: Maintain clear and concise documentation for each test case, including expected inputs, outputs, and edge cases.

---

### **Conclusion**

This testing framework ensures that each component of your Quant Strat app is rigorously tested for reliability, accuracy, and performance. With a combination of unit, integration, performance, backtesting, and statistical tests, you’ll be well-equipped to handle real-world requirements and ensure that your Quant strategies provide accurate, consistent, and actionable insights in production. 

### 5. Deployment

### 6. Maintenance

## SnowFreq Final Statistical analysis & Report briefing

# Results

> With these features applied we generate a **Historical** and **Live** Performance of its Trade Postionings metrics & graphs.

- We should perform both backtest & live runs at the same time to ensure their timedelta's are closest.

1. Use Freqtrade for crypto while Lumibot for stocks for both Historical & Live data.
2. Use Tradeview to observe for metric & graph data co-relation. But it has no dynamic Timedelta data testing
3. Store and rank each strategy backtest & live performance.

> Comparison of Portfolio price for **Historical** and **Live** execution against Global Indexes
:SMP, Berkshire, HKSE

# Discussion

> How does Timerange & Timedelta variables affect trading performance

> Performance of Technical Indicators strategies & Quant Portfolio assets Engineering

# Conclusion
# References
