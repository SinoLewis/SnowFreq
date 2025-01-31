import pandas as pd

def calculate_ok(row):
    """
    Calculates the Ok values (fracChange, fracHigh, fracLow) for a given row of OHLCV data.

    Args:
        row: A pandas Series representing a single row of the OHLCV data.

    Returns:
        A tuple containing the calculated fracChange, fracHigh, and fracLow values.
    """
    open_price = row['open']
    close_price = row['close']
    high_price = row['high']
    low_price = row['low']

    frac_change = (close_price - open_price) / open_price
    frac_high = (high_price - open_price) / open_price
    frac_low = (open_price - low_price) / open_price

    return frac_change, frac_high, frac_low

# Load the OHLCV data from the feather file
df = pd.read_feather('/freqtrade/user_data/data/binance/BAT_USDT-1h.feather') 

# Apply the calculate_ok function to each row of the DataFrame
df[['fracChange', 'fracHigh', 'fracLow']] = df.apply(calculate_ok, axis=1, result_type='expand')

# Print the calculated Ok values for each row
print(df[['fracChange', 'fracHigh', 'fracLow']]) 