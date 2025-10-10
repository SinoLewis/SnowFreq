import pandas as pd
import numpy as np
from models.LSTM_TF import LSTMRegressor
from Features import derived_features, sliding_windows 
from datetime import datetime

class Regressor_TF():
    def __init__(self, seq_length=28, input_size=6, num_classes=1, hidden_size=200, num_layers=2, batch_size=32, num_epochs=2000, learning_rate=1e-3):
        ##### Model  Parameters  ######################
        self.seq_length =  seq_length
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.lstm =  LSTMRegressor(self.seq_length, self.input_size, self.learning_rate, output_size=self.num_classes).model
        ##### Train  Parameters  ######################
        self.batch_size=batch_size
        self.num_epochs = num_epochs

    def train_lstm_tf(self, dataset, num_epochs=2000, batch_size=32):
        history = self.lstm.fit(
            dataset["train"]["X"], dataset["train"]["y"],
            # validation_data=(testX, testX),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=2
        )

        # Evaluation
        evals = self.lstm.evaluate(dataset["test"]["X"], dataset["test"]["y"], verbose=0)

        # Predictions
        preds = self.lstm.predict(dataset["test"]["X"], batch_size=batch_size, verbose=0)
        return evals, preds
        print(f"Test Results - MAE: {results[1]}, MSE: {results[2]}, MAPE: {results[3]}")
    def predict(self, X_input, batch_size=32):
        preds = self.lstm.predict(X_input, batch_size=batch_size, verbose=0)
        return preds

    def main(self, prices_scaled):
        print(f"[INFO] Sample of prices_scaled:")
        print(prices_scaled[:5])

        # Prepare dataset
        x, y = sliding_windows(prices_scaled, self.seq_length)

        n_samples = len(y)
        train_end = int(n_samples * 0.6)
        val_end   = int(n_samples * 0.8)   # remaining 20% for test

        dataset = {
            "train": {
                "X": np.array(x[:train_end]),
                "y": np.array(y[:train_end]).reshape(-1, 1).astype(np.float32),
            },
            "val": {
                "X": np.array(x[train_end:val_end]),
                "y": np.array(y[train_end:val_end]).reshape(-1, 1).astype(np.float32),
            },
            "test": {
                "X": np.array(x[val_end:]),
                "y": np.array(y[val_end:]).reshape(-1, 1).astype(np.float32),
            }
        }

        # Train model with validation & Prediction set
        evals, preds = self.train_lstm_tf(dataset, self.num_epochs, self.batch_size)
        print(f"[INFO] Evals Data: {evals}")
        # === Dynamic filename setup ===
        execdate = datetime.now().strftime("%Y%m%d")   # e.g., 20251003
        domain = "cryptonews"                           # or dynamically parsed from filename
        # Build dynamic filename
        filename = f"{execdate}-{domain}.csv"
        preds_df = pd.DataFrame(preds)
        preds_df.to_csv(filename, index=False)

        print(f"[INFO] Predictions shape: {preds.shape}")
        print(f"[INFO] First 5 predictions:\n{preds[:5].flatten()}")
        print(f"[INFO] First 5 ground truth:\n{dataset['test']['y'][:5].flatten()}")


    def main_old(self, prices_scaled):
            print(f"[INFO] Sample of prices_scaled:")
            print(prices_scaled[:5])
            # x, y = sliding_windows(mode_series_scaled, seq_length=28)
            x, y = sliding_windows(prices_scaled, self.seq_length)

            train_size = int(len(y) * 0.67)
            # test_size = len(y) - train_size

            dataset = {
                "train": {
                    "X": np.array(x[:train_size]),
                    "y": np.array(y[:train_size]).reshape(-1, 1).astype(np.float32),
                },
                "test": {
                    "X": np.array(x[train_size:]),
                    "y": np.array(y[train_size:]).reshape(-1, 1).astype(np.float32),
                }
            }

            print(f"[INFO] sliding_windows() returned:")
            print(f"       x.shape = {x.shape}  # (num_samples, seq_length, features)")
            print(f"       y.shape = {y.shape}  # (num_samples, 1 or seq_length depending on target prep)")
            
            print(f"[INFO] After tensor_inputs():")
            print(dataset["train"]["X"].shape)  # (train_samples, seq_length, features)
            print(dataset["train"]["y"].shape)  # (train_samples, 1)
            print(dataset["test"]["X"].shape)   # (test_samples, seq_length, features)
            print(dataset["test"]["y"].shape)   # (test_samples, 1)

            print("\n[INFO] Model Architecture:\n")
            print(self.lstm)   # This will print a full summary of layers

            predictions = self.train_lstm_tf(dataset, self.num_epochs, self.batch_size)
            preds_df = pd.DataFrame(predictions)
            # === Dynamic filename setup ===
            execdate = datetime.now().strftime("%Y%m%d")   # e.g., 20251003
            domain = "crypto-ohlcv"                           # or dynamically parsed from filename

            # Build dynamic filename
            filename = f"{execdate}-{domain}.csv"
            preds_df.to_csv(filename, index=False)

if __name__ == "__main__":

    data_file = '../data/BTC_USDT-1d.feather'
    df_ohclv = pd.read_feather(data_file)

    app = Regressor_TF(
        seq_length=28, input_size=6, num_classes=1, hidden_size=200, num_layers=2, 
        batch_size=32, num_epochs=20, learning_rate=1e-3)
    prices_scaled = derived_features(
        prices=df_ohclv['close'], 
        alpha=5000, tau=0, K=5, DC=0, init=1, tol=1e-7)
    app.main(prices_scaled)
