"""
FinRl_Opt.py — Modularized Single-File FinRL Implementation
Author: Scott West
Description:
A modular FinRL script for crypto portfolio optimization using Stable-Baselines3.
"""

# ============================================================
# 1. Imports
# ============================================================
import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config, config_tickers

# ============================================================
# 2. Directory Setup
# ============================================================
def create_directories():
    """Ensure all necessary directories exist."""
    dirs = [
        config.DATA_SAVE_DIR,
        config.TRAINED_MODEL_DIR,
        config.TENSORBOARD_LOG_DIR,
        config.RESULTS_DIR
    ]
    for d in dirs:
        os.makedirs(f"./{d}", exist_ok=True)
    print("✅ Required directories are ready.")

# ============================================================
# 3. Data Loading
# ============================================================
def load_market_data(data_dir: str, assets: list, tf: str, vix_path: str):
    """Load OHLCV data and append VIX indicator."""
    vix_df = pd.read_csv(vix_path)
    vix_col = vix_df['close'].rename('vix')
    df_list = []

    for asset in assets:
        file_path = os.path.join(data_dir, f"{asset}-{tf}.feather")
        if os.path.exists(file_path):
            df = pd.read_feather(file_path)
            df["tic"] = asset
            cols = ["date", "open", "high", "low", "close", "volume", "tic"]
            df = df[[col for col in cols if col in df.columns]]
            df = df.join(vix_col, how='left')
            df_list.append(df)
        else:
            print(f"⚠️ File not found: {file_path}")

    if not df_list:
        raise FileNotFoundError("No asset files were loaded.")

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime('%d-%m-%Y')

    print("✅ Composite DataFrame created!")
    print(df.head(5))
    print(f"Total records: {len(df):,}")
    return df

# ============================================================
# 4. Feature Engineering
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering and fill missing dates/tickers."""
    fe = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)
    processed = fe.preprocess_data(df)

    processed["date"] = pd.to_datetime(processed["date"], errors="coerce", dayfirst=True).dt.normalize()
    processed = processed.dropna(subset=["date"])

    list_ticker = processed["tic"].unique().tolist()
    list_date = pd.date_range(processed["date"].min(), processed["date"].max(), freq='D')

    combination = list(itertools.product(list_date, list_ticker))
    processed_full = pd.DataFrame(combination, columns=["date", "tic"])
    processed_full = processed_full.merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full.sort_values(["date", "tic"]).reset_index(drop=True).fillna(0)

    print("✅ Feature-engineered dataframe ready!")
    return processed_full

# ============================================================
# 5. Covariance State Construction
# ============================================================
def add_covariance_states(processed_full: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Add rolling covariance matrices and returns to the dataset."""
    df = processed_full.sort_values(['date','tic'], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list, return_list = [], []
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        cov_list.append(return_lookback.cov().values)

    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
    processed_cov = df.merge(df_cov, on='date')
    processed_full = processed_cov.sort_values(['date','tic']).reset_index(drop=True)
    print("✅ Covariance states added.")
    return processed_full

# ============================================================
# 6. Environment Creation
# ============================================================
def make_env(df: pd.DataFrame, start: str, end: str, tech_indicator_list: list):
    """Initialize training or trading environment."""
    data = data_split(df, start, end)
    stock_dimension = len(data.tic.unique())
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "transaction_cost_pct": 0,
        "state_space": stock_dimension,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-1
    }
    env = StockPortfolioEnv(df=data, **env_kwargs)
    return env, stock_dimension

# ============================================================
# 7. DRL Model Training
# ============================================================
def train_agents(env_train):
    """Train both A2C and PPO agents."""
    agent = DRLAgent(env=env_train)

    # A2C
    A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}
    model_a2c = agent.get_model("a2c", model_kwargs=A2C_PARAMS)
    trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=40000)

    # PPO
    PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.001, "batch_size": 128}
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=40000)

    print("✅ Agents trained successfully.")
    return trained_a2c, trained_ppo

# ============================================================
# 8. Backtesting
# ============================================================
def backtest_agents(trained_a2c, trained_ppo, e_trade_gym):
    """Run DRL predictions for A2C and PPO."""
    df_daily_return_a2c, df_actions_a2c = DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)
    df_daily_return_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    print("✅ Backtesting completed.")
    return df_daily_return_a2c, df_actions_a2c, df_daily_return_ppo, df_actions_ppo

# ============================================================
# 9. Main Execution
# ============================================================
# Args Mod: data_dir, assets, tf, timerange, split_ratio
def main():
    create_directories()

    # User configuration
    data_dir = "binance-22-25"
    assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
    tf = '1d'
    vix_path = "vix_daily.csv"

    # Load data
    df = load_market_data(data_dir, assets, tf, vix_path)

    TRAIN_START_DATE = df['date'].iloc[0]
    TRAIN_END_DATE = '2023-03-11'
    TRADE_START_DATE = '2023-03-12'
    TRADE_END_DATE = df['date'].iloc[-1]
    # preprocess Tech Indicators
    processed_full = engineer_features(df)
    processed_full = add_covariance_states(processed_full)

    base_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
    tech_indicator_list = [col for col in processed_full.columns if col not in base_cols]
    # tech_indicator_list = ['macd', 'rsi_30', 'cci_30', 'dx_30']

    # Environments
    e_train_gym, stock_dim = make_env(processed_full, TRAIN_START_DATE, TRAIN_END_DATE, tech_indicator_list)
    env_train, _ = e_train_gym.get_sb_env()

    trained_a2c, trained_ppo = train_agents(env_train)

    e_trade_gym, _ = make_env(processed_full, TRADE_START_DATE, TRADE_END_DATE, tech_indicator_list)
    backtest_agents(trained_a2c, trained_ppo, e_trade_gym)

if __name__ == "__main__":
    main()
