import itertools
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import os

def load_portfolio_data(data_dir, assets, tf="1d"):
    """Load OHLCV data from multiple assets and merge into one composite DataFrame."""
    df_list = []
    for asset in assets:
        file_path = os.path.join(data_dir, f"{asset}-{tf}.feather")
        if os.path.exists(file_path):
            df = pd.read_feather(file_path)
            df["tic"] = asset
            cols = ["date", "open", "high", "low", "close", "volume", "tic"]
            df = df[[col for col in cols if col in df.columns]]
            df_list.append(df)
        else:
            print(f"⚠️ Missing file: {file_path}")

    if not df_list:
        raise ValueError("No valid data files found.")

    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    print(f"✅ Loaded {len(df):,} records from {len(assets)} assets.")
    return df

def process_features(df, start_date=None, end_date=None):
    """Generate technical indicators and prepare train/test data."""
    fe = FeatureEngineer(
        use_technical_indicator=True,
        use_turbulence=False,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce", dayfirst=True)
    processed = processed.dropna(subset=["date"])

    tickers = processed["tic"].unique()
    all_dates = pd.date_range(processed["date"].min(), processed["date"].max(), freq="D")
    combo = pd.DataFrame(list(itertools.product(all_dates, tickers)), columns=["date", "tic"])

    processed_full = combo.merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full.fillna(0).sort_values(["date", "tic"]).reset_index(drop=True)

    print("✅ Processed features ready.")
    return processed_full
