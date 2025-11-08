from FinRl_Mods.feature_engineering import process_features, load_portfolio_data
from FinRl_Mods.env_initializer import init_stock_environment
from FinRl_Mods.agent_initializer import train_agent
import pandas as pd
import datetime
from finrl.plot import backtest_stats, get_baseline
from finrl.agents.stablebaselines3.models import DRLAgent

# --- Config ---
DATA_DIR = "binance-22-25"
ASSETS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]

# --- Step 1: Load data ---
df = load_portfolio_data(DATA_DIR, ASSETS)

# --- Step 2: Feature Engineering ---
processed = process_features(df)

# --- Step 3: Environment Init ---
TRAIN_START = processed["date"].iloc[0]
TRAIN_END = "2023-03-11"
TEST_START = "2023-03-12"
TEST_END = processed["date"].iloc[-1]

# NB: 2-3 min
e_train_gym, e_trade_gym = init_stock_environment(processed, TRAIN_START, TRAIN_END, TEST_START, TEST_END)

# --- Step 4: Train Agent ---
trained_model = train_agent(e_train_gym, model_type="ppo", total_timesteps=50_000)

# --- Step 5: Backtest ---
df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)

perf_stats_all_ppo = backtest_stats(account_value=df_account_value_ppo)
perf_stats_all_ppo = pd.DataFrame(perf_stats_all_ppo)
perf_stats_all_ppo.to_csv(
    "./" + config.RESULTS_DIR + "/perf_stats_all_ppo_" + now + ".csv"
)
# TODO: Get Data localy
# print("==============Get Baseline Stats===========")
# baseline_df = get_baseline(ticker="^DJI", start=TEST_START_DATE, end=TEST_END_DATE)
# stats = backtest_stats(baseline_df, value_col_name="close")
# print(f"✅ Backtest & baseline stats complete")

df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
result = df_result_ppo
result.columns = ["ppo"]

print("result: ", result)
result.to_csv("result.csv")
