from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions, black_litterman, risk_models 
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.black_litterman import BlackLittermanModel

def mvo_optimizer(df):
  mu = mean_historical_return(df)
  S = CovarianceShrinkage(df).ledoit_wolf()
  
  ef = EfficientFrontier(mu, S)
  weights = ef.max_sharpe()
  
  cleaned_weights = ef.clean_weights()
  #ef.save_weights_to_file("weights.txt")  # saves to file
  print(cleaned_weights)
  
  # Shorting
  #ef.efficient_return(target_return=0.2, market_neutral=True)
  
  # Dealing with many negligible weights 
  ef.add_objective(objective_functions.L2_reg, gamma=0.1)
  w = ef.clean_weights()
  print('fewer negligible weights than before \n')
  print(w)
  
  # convert these weights into an actual allocation 
  latest_prices = get_latest_prices(df)
  da = DiscreteAllocation(w, latest_prices, total_portfolio_value=20000)
  allocation, leftover = da.lp_portfolio()
  print(allocation)

# the (daily) prices of the market portfolio, e.g SPY.
def blm_optimizer(df, market_prices):
  """
  cov_matrix is a NxN sample covariance matrix
  mcaps is a dict of market caps
  market_prices is a series of S&P500 prices
  """
  cov_matrix = risk_models.sample_cov(df) 
  # prior as the “default” estimate, in the absence of any information 
  delta = black_litterman.market_implied_risk_aversion(market_prices)
  prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)
  # Views: users can either provide absolute or relative views.
  viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
  bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)

  rets = bl.bl_returns()
  ef = EfficientFrontier(rets, cov_matrix)
  
  # OR use return-implied weights
  delta = black_litterman.market_implied_risk_aversion(market_prices)
  bl.bl_weights(delta)
  weights = bl.clean_weights()
  print('BlackLittermanModel weights \n')
  print(w)
   
  latest_prices = get_latest_prices(df)
  da = DiscreteAllocation(w, latest_prices, total_portfolio_value=20000)
  allocation, leftover = da.lp_portfolio()
  print(allocation)
    
'''
NB: Improving performance
Let’s say you have conducted backtests and the results aren’t spectacular. What should you try?

1. Try the Hierarchical Risk Parity model (see Other Optimizers) – which seems to robustly outperform mean-variance optimization out of sample.
2. Use the Black-Litterman model to construct a more stable model of expected returns. Alternatively, just drop the expected returns altogether! There is a large body of research that suggests that minimum variance portfolios (ef.min_volatility()) consistently outperform maximum Sharpe ratio portfolios out-of-sample (even when measured by Sharpe ratio), because of the difficulty of forecasting expected returns.
3. Try different risk models: shrinkage models are known to have better numerical properties compared with the sample covariance matrix.
4. Add some new objective terms or constraints. Tune the L2 regularisation parameter to see how diversification affects the performance.
'''