import numpy as np
import pandas as pd
import talib


class PrimitiveSignal:
    """
    Primitive (Atomic) Alpha Signal Generator
    -----------------------------------------
    Initialized with OHLCV data and exposes
    individual signal functions.
    """

    def __init__(self, ohlcv: pd.DataFrame):
        """
        Parameters
        ----------
        ohlcv : pd.DataFrame
            Must contain columns:
            ['open', 'high', 'low', 'close', 'volume']
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(ohlcv.columns):
            raise ValueError(f"OHLCV data must contain {required_cols}")

        self.df = ohlcv.copy()

        self.open = self.df["open"].values
        self.high = self.df["high"].values
        self.low = self.df["low"].values
        self.close = self.df["close"].values
        self.volume = self.df["volume"].values

    # --------------------------------------------------
    # 1. RSI Momentum Signal
    # --------------------------------------------------
    def rsi_signal(self, period: int = 14) -> pd.Series:
        """
        RSI normalized to [-1, 1]
        """
        rsi = talib.RSI(self.close, timeperiod=period)
        signal = (rsi - 50) / 50
        return pd.Series(signal, index=self.df.index, name="rsi_signal")

    # --------------------------------------------------
    # 2. MACD Trend Signal
    # --------------------------------------------------
    def macd_signal(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9
    ) -> pd.Series:
        """
        MACD histogram normalized by price
        """
        macd, macd_signal, macd_hist = talib.MACD(
            self.close,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal_period
        )

        normalized = macd_hist / self.close
        return pd.Series(normalized, index=self.df.index, name="macd_signal")

    # --------------------------------------------------
    # 3. Bollinger Band Z-Score
    # --------------------------------------------------
    def bollinger_zscore(
        self,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Distance from middle band in standard deviations
        """
        upper, middle, lower = talib.BBANDS(
            self.close,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev
        )

        zscore = (self.close - middle) / (upper - middle)
        return pd.Series(zscore, index=self.df.index, name="bb_zscore")

    # --------------------------------------------------
    # 4. ATR Volatility Signal
    # --------------------------------------------------
    def atr_volatility(self, period: int = 14) -> pd.Series:
        """
        ATR normalized by price
        """
        atr = talib.ATR(
            self.high,
            self.low,
            self.close,
            timeperiod=period
        )

        normalized = atr / self.close
        return pd.Series(normalized, index=self.df.index, name="atr_volatility")

    # --------------------------------------------------
    # 5. Volume Z-Score Signal
    # --------------------------------------------------
    def volume_zscore(self, period: int = 20) -> pd.Series:
        """
        Detects abnormal volume activity
        """
        vol_series = pd.Series(self.volume, index=self.df.index)

        mean = vol_series.rolling(period).mean()
        std = vol_series.rolling(period).std()

        zscore = (vol_series - mean) / std
        return pd.Series(zscore, index=self.df.index, name="volume_zscore")

    # --------------------------------------------------
    # 6. EMA Trend Strength (Bonus)
    # --------------------------------------------------
    def ema_trend(self, short: int = 20, long: int = 50) -> pd.Series:
        """
        EMA trend strength normalized by price
        """
        ema_short = talib.EMA(self.close, timeperiod=short)
        ema_long = talib.EMA(self.close, timeperiod=long)

        trend = (ema_short - ema_long) / self.close
        return pd.Series(trend, index=self.df.index, name="ema_trend")

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_bars: int = 1000,
    start_price: float = 100.0,
    freq: str = "1D",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data for signal testing.

    Parameters
    ----------
    n_bars : int
        Number of candles
    start_price : float
        Initial price
    freq : str
        Pandas frequency string (e.g., '1D', '1H')
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        OHLCV dataframe
    """
    np.random.seed(seed)

    # --- Time index ---
    index = pd.date_range(
        start="2020-01-01",
        periods=n_bars,
        freq=freq
    )

    # --- Log returns with drift + volatility ---
    drift = 0.0002
    volatility = 0.01

    log_returns = np.random.normal(
        loc=drift,
        scale=volatility,
        size=n_bars
    )

    close = start_price * np.exp(np.cumsum(log_returns))

    # --- Open prices ---
    open_ = np.roll(close, 1)
    open_[0] = start_price

    # --- High / Low with realistic candle ranges ---
    spread = np.random.uniform(0.001, 0.02, size=n_bars)

    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)

    # --- Volume (log-normal with regime noise) ---
    volume = np.random.lognormal(
        mean=12,
        sigma=0.3,
        size=n_bars
    ).astype(int)

    # --- Assemble DataFrame ---
    ohlcv_df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=index)

    return ohlcv_df

ohlcv_df = generate_synthetic_ohlcv(
    n_bars=1500,
    start_price=100.0,
    freq="1D"
)


signals = PrimitiveSignal(ohlcv_df)

df_signals = pd.concat([
    signals.rsi_signal(),
    signals.macd_signal(),
    signals.bollinger_zscore(),
    signals.atr_volatility(),
    signals.volume_zscore(),
    signals.ema_trend()
], axis=1)
print(df_signals.tail(15))
