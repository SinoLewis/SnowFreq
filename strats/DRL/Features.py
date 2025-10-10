import pandas as pd

def debug_dataframe(df: pd.DataFrame, name: str = "DataFrame", max_rows: int = 5):
    """
    Prints detailed debug info about a pandas DataFrame.

    Args:
        df (pd.DataFrame): The dataframe to inspect
        name (str): Optional name for logging
        max_rows (int): Number of rows to preview from head/tail
    """
    print(f"\n{'='*40}")
    print(f"[DEBUG] Inspecting {name}")
    print(f"{'='*40}")

    # Shape and basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Dtypes + Non-null counts
    print("[INFO] DataFrame info():")
    print(df.info(memory_usage="deep"))

    # Nulls
    nulls = df.isnull().sum()
    print("\n[INFO] Null counts:")
    print(nulls[nulls > 0] if nulls.sum() > 0 else "No nulls found")

    # Stats
    print("\n[INFO] Summary statistics:")
    print(df.describe(include="all").transpose())

    # Head and tail
    print(f"\n[INFO] First {max_rows} rows:")
    print(df.head(max_rows))
    print(f"\n[INFO] Last {max_rows} rows:")
    print(df.tail(max_rows))

    print(f"\n[DEBUG] Finished inspecting {name}")
    print(f"{'='*40}\n")