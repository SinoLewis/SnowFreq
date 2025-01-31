from datetime import datetime 
import pickle
from pprint import pprint
import pandas as pd

test_dir = "/freqtrade/user_data/tests"

with open(f'{test_dir}/bt_data_preprocessed_tmp_content.pkl', 'rb') as f:
    df = pickle.load(f)

pprint(df['BTC/USDT'].columns)
pprint(df)

with open(f'{test_dir}/bt_data_preprocessed_content.pkl', 'rb') as f:
    df = pickle.load(f)

pprint(df['BTC/USDT'].columns)
pprint(df)

# df = df_loaded['results']

# date_col = pd.to_datetime(df['close_timestamp'], unit='ms').dt.date
# pprint(date_col)

with open(f'{test_dir}/strat_df.pkl', 'rb') as f:
    df_strat = pickle.load(f)
    # pprint(df_strat.columns)

# Quant Strats DEV

import numpy as np
import pandas as pd
# from arch import arch_model  # For GARCH model, ensure arch package is installed

# Each strat will compute on simple input vars. unlike input df for freq strats
def calculate_var(df, confidence_level=0.95, returns_col='profit_ratio'):
    """
    Calculate Value at Risk (VaR) for a given DataFrame and confidence level.
    
    Parameters:
    - df (DataFrame): The ticker timeseries data.
    - confidence_level (float): Confidence level for VaR calculation (default is 95%).
    - returns_col (str): Column in the DataFrame containing returns data.
    
    Returns:
    - VaR (float): Value at Risk at the specified confidence level.
    """
    # Compute VaR using the quantile of returns at (1 - confidence level)
    VaR = df[returns_col].quantile(1 - confidence_level)
    return VaR

def calculate_es(df, confidence_level=0.95, returns_col='profit_ratio'):
    """
    Calculate Expected Shortfall (ES) for a given DataFrame and confidence level.
    
    Parameters:
    - df (DataFrame): The ticker timeseries data.
    - confidence_level (float): Confidence level for ES calculation (default is 95%).
    - returns_col (str): Column in the DataFrame containing returns data.
    
    Returns:
    - ES (float): Expected Shortfall at the specified confidence level.
    """
    # First, calculate the VaR
    VaR = calculate_var(df, confidence_level, returns_col)
    
    # Filter returns less than VaR (i.e., losses worse than VaR)
    losses_beyond_VaR = df[df[returns_col] < VaR][returns_col]
    
    # Calculate Expected Shortfall as the mean of the losses beyond VaR
    ES = losses_beyond_VaR.mean()
    return ES

def calculate_volatility(df, returns_col='profit_ratio'):
    """
    Calculate Volatility based on returns from the DataFrame.
    
    Parameters:
    - df (DataFrame): The ticker timeseries data.
    - returns_col (str): Column in the DataFrame containing returns data.
    
    Returns:
    - volatility (float): Standard deviation of the returns.
    """
    # Calculate the standard deviation of returns to measure volatility
    volatility = df[returns_col].std()
    return volatility

def calculate_garch_volatility(df, returns_col='profit_ratio', p=1, q=1):
    """
    Calculate volatility using a GARCH(p, q) model on returns data.
    
    Parameters:
    - df (DataFrame): The ticker timeseries data.
    - returns_col (str): Column in the DataFrame containing returns data.
    - p (int): Lag order of the GARCH model for the autoregressive term (default is 1).
    - q (int): Lag order of the GARCH model for the moving average term (default is 1).
    
    Returns:
    - garch_volatility (float): Forecasted volatility from the GARCH model.
    """
    # Fit a GARCH(p, q) model to the returns
    returns = df[returns_col].dropna()  # Remove any NaN values
    model = arch_model(returns, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp="off")
    
    # Forecast the volatility for the next period
    forecast = model_fit.forecast(horizon=1)
    garch_volatility = np.sqrt(forecast.variance.values[-1, :][0])  # Get the volatility forecast for the next period
    return garch_volatility

# Freq strats w/ input df

def calculate_sortino(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    Calculate sortino
    :param trades: DataFrame containing trades (requires columns profit_abs)
    :return: sortino
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period

    down_stdev = np.std(trades.loc[trades["profit_abs"] < 0, "profit_abs"] / starting_balance)

    if down_stdev != 0 and not np.isnan(down_stdev):
        sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
    else:
        # Define high (negative) sortino ratio to be clear that this is NOT optimal.
        sortino_ratio = -100

    # print(expected_returns_mean, down_stdev, sortino_ratio)
    return sortino_ratio

def calculate_sharpe(
    trades: pd.DataFrame, min_date: datetime, max_date: datetime, starting_balance: float
) -> float:
    """
    Calculate sharpe
    :param trades: DataFrame containing trades (requires column profit_abs)
    :return: sharpe
    """
    if (len(trades) == 0) or (min_date is None) or (max_date is None) or (min_date == max_date):
        return 0

    total_profit = trades["profit_abs"] / starting_balance
    days_period = max(1, (max_date - min_date).days)

    expected_returns_mean = total_profit.sum() / days_period
    up_stdev = np.std(total_profit)

    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * np.sqrt(365)
    else:
        # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        sharp_ratio = -100

    # print(expected_returns_mean, up_stdev, sharp_ratio)
    return sharp_ratio

def calculate_max_drawdown(
    trades: pd.DataFrame,
    *,
    date_col: str = "close_date",
    value_col: str = "profit_abs",
    starting_balance: float = 0,
    relative: bool = False,
) -> DrawDownResult:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_abs')
    :param starting_balance: Portfolio starting balance - properly calculate relative drawdown.
    :return: DrawDownResult object
             with absolute max drawdown, high and low time and high and low value,
             and the relative account drawdown
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError("Trade dataframe empty.")
    profit_results = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df = _calc_drawdown_series(
        profit_results, date_col=date_col, value_col=value_col, starting_balance=starting_balance
    )

    idxmin = (
        max_drawdown_df["drawdown_relative"].idxmax()
        if relative
        else max_drawdown_df["drawdown"].idxmin()
    )
    if idxmin == 0:
        raise ValueError("No losing trade, therefore no drawdown.")
    high_date = profit_results.loc[max_drawdown_df.iloc[:idxmin]["high_value"].idxmax(), date_col]
    low_date = profit_results.loc[idxmin, date_col]
    high_val = max_drawdown_df.loc[
        max_drawdown_df.iloc[:idxmin]["high_value"].idxmax(), "cumulative"
    ]
    low_val = max_drawdown_df.loc[idxmin, "cumulative"]
    max_drawdown_rel = max_drawdown_df.loc[idxmin, "drawdown_relative"]

    return DrawDownResult(
        drawdown_abs=abs(max_drawdown_df.loc[idxmin, "drawdown"]),
        high_date=high_date,
        low_date=low_date,
        high_value=high_val,
        low_value=low_val,
        relative_account_drawdown=max_drawdown_rel,
    )
