import tensorflow as tf
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense

class LSTMClassifier(Model):
    def __init__(self, input_shape, num_classes=2):
        super(LSTMClassifier, self).__init__()

        # First LSTM block
        self.lstm1 = LSTM(128, return_sequences=True, input_shape=input_shape)
        self.drop1 = Dropout(0.2)
        self.bn1 = BatchNormalization()

        # Second LSTM block
        self.lstm2 = LSTM(128, return_sequences=True)
        self.drop2 = Dropout(0.1)
        self.bn2 = BatchNormalization()

        # Third LSTM block
        self.lstm3 = LSTM(128)
        self.drop3 = Dropout(0.2)
        self.bn3 = BatchNormalization()

        # Dense layers
        self.fc1 = Dense(32, activation="relu")
        self.drop4 = Dropout(0.2)
        self.fc2 = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.drop1(x, training=training)
        x = self.bn1(x, training=training)

        x = self.lstm2(x)
        x = self.drop2(x, training=training)
        x = self.bn2(x, training=training)

        x = self.lstm3(x)
        x = self.drop3(x, training=training)
        x = self.bn3(x, training=training)

        x = self.fc1(x)
        x = self.drop4(x, training=training)
        return self.fc2(x)


class LSTMRegressor:
    def __init__(self, seq_len, input_size, learning_rate, output_size=1):
        """
        seq_len: number of timesteps in each input sequence
        input_size: number of features at each timestep
        output_size: number of outputs (1 = single step, >1 = multi-step or classification classes)
        """
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.seq_len, self.input_size), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        # Final output
        model.add(Dense(self.output_size, activation='softmax' if self.output_size > 1 else 'linear'))

        model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=losses.MeanSquaredError(),  # primary loss
        metrics=[
            metrics.MeanAbsoluteError(name="MAE"),
            metrics.MeanSquaredError(name="MSE"),
            metrics.MeanAbsolutePercentageError(name="MAPE")
        ]
        )
        return model