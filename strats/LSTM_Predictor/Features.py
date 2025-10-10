from sklearn.preprocessing import MinMaxScaler
import numpy as np
from vmdpy import VMD

def derived_features(prices, alpha=5000, tau=0, K=5, DC=0, init=1, tol=1e-7):
    """
    Decompose prices using VMD and scale both modes and prices.
    Returns a feature matrix where each row = [mode1, mode2, ..., modeK, scaled_price].
    """
    # Decompose
    modes, u_hat, omega = VMD(prices, alpha, tau, K, DC, init, tol)
    # Reshape to (T, K) so each timestep has K features
    mode_series = modes.T  

    # Scale modes
    scaler_modes = MinMaxScaler()
    mode_series_scaled = scaler_modes.fit_transform(mode_series)

    # Scale price
    scaler_price = MinMaxScaler()
    prices_scaled = scaler_price.fit_transform(prices.values.reshape(-1, 1))

    # Combine: features = [modes..., scaled price]
    features = np.hstack([mode_series_scaled, prices_scaled])

    return features


def sliding_windows(data, seq_length):
    """
    Create LSTM input-output pairs from sequential data.
    """
    x, y = [], []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length, -1]  # predict price (last feature)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)
