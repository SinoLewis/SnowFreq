from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

def init_stock_environment(processed_df, train_start, train_end, test_start, test_end):
    """Initialize Gym environments for training and trading."""
    train_data = data_split(processed_df, train_start, train_end)
    trade_data = data_split(processed_df, test_start, test_end)

    base_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
    indicator_list = [col for col in train_data.columns if col not in base_cols]

    stock_dim = len(train_data.tic.unique())
    num_stock_shares = [0] * stock_dim
    state_space = 1 + 2 * stock_dim + len(indicator_list) * stock_dim

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "num_stock_shares": num_stock_shares,
        "tech_indicator_list": indicator_list,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    e_trade_gym = StockTradingEnv(df=trade_data, **env_kwargs)

    print("✅ Gym environments initialized.")
    return env_train, e_trade_gym
