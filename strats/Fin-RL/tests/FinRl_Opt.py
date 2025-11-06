# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# %matplotlib inline

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
# from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

import sys
sys.path.append("../FinRL")

import itertools

"""<a id='1.4'></a>
## 2.4. Create Folders
"""

import os
from finrl import config
from finrl import config_tickers

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)

import os

# Directory containing your feather files
data_dir = r"binance-22-25"  # <-- change this

# List of assets you want to include
assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']  # <-- customize this

# Desired timeframe(s)
tf = '1d'
# Container for all loaded DataFrames
df_list = []
# Add Vix prices Series
vix_df = pd.read_csv(r"vix_daily.csv")
vix_col = vix_df['close'].rename('vix')

for asset in assets:
      file_path = os.path.join(data_dir, f"{asset}-{tf}.feather")
      if os.path.exists(file_path):
          # Load feather file
          df = pd.read_feather(file_path)

          # Add ticker_name and timeframe info
          df["tic"] = asset

          # Keep only consistent OHCLV columns + metadata
          cols = ["date", "open", "high", "low", "close", "volume", "tic"]
          df = df[[col for col in cols if col in df.columns]]
          # Add vix_col to each asset
          df = df.join(vix_col, how='left')

          df_list.append(df)
      else:
          print(f"⚠️ File not found: {file_path}")
# Combine all into a single composite DataFrame
if df_list:
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
else:
    df = pd.DataFrame()
df["date"] = df["date"].dt.strftime('%d-%m-%Y')

# --- Final check ---
print("✅ Composite DataFrame created!")
print(df.head(10))
print(df.tail(10))
print(f"\nTotal records: {len(df):,}")

df.sort_values(['date','tic']).head()

TRAIN_START_DATE = df['date'].iloc[0]
TRAIN_END_DATE = '2023-03-11'
TRADE_START_DATE = '2023-03-12'
TRADE_END_DATE = df['date'].iloc[-1]


fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)

# DEBUG: Resolved miss-align error
# --- Ensure date column is datetime ---
# Handles cases like '13-01-2021' or '2021-01-13'
processed["date"] = pd.to_datetime(processed["date"], errors="coerce", dayfirst=True).dt.normalize()

# Drop rows with invalid (NaT) dates if any
processed = processed.dropna(subset=["date"])

# --- Build all combinations of dates and tickers ---
list_ticker = processed["tic"].unique().tolist()
list_date = pd.date_range(processed["date"].min(), processed["date"].max(), freq='D')

# --- Create full date-ticker frame ---
combination = list(itertools.product(list_date, list_ticker))
processed_full = pd.DataFrame(combination, columns=["date", "tic"])

# --- Merge safely ---
processed_full = processed_full.merge(processed, on=["date", "tic"], how="left")

# --- Sort and fill missing ---
processed_full = processed_full.sort_values(["date", "tic"]).reset_index(drop=True)
processed_full = processed_full.fillna(0)

print("✅ Processed full dataframe created successfully!")
print(processed_full.head(10))

"""## Add covariance matrix as states"""

# add covariance matrix as states
df=processed_full.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values
  cov_list.append(covs)


df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
processed_cov = df.merge(df_cov, on='date')
processed_full = processed_cov.sort_values(['date','tic']).reset_index(drop=True)

train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)

stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']
feature_dimension = len(tech_indicator_list)
print(f"Feature Dimension: {feature_dimension}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicator_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-1

}

e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# # Weight Init by Sampling
# retail_train = StockPortfolioEnv.sample_from_env(i=0, env=e_train_gym, weights=train['macd'])
# print("🚩 Retail Weights Init: ", retail_train)

agent = DRLAgent(env = env_train)

A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c',
                                total_timesteps=40000)

agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.001,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=40000)

## Back-Testing
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)

df_daily_return_a2c, df_actions_a2c = DRLAgent.DRL_prediction(model=trained_a2c,
                        environment = e_trade_gym)
df_daily_return_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo,
                        environment = e_trade_gym)