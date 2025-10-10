import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQN(tf.keras.Model):
    def __init__(self, input_shape, output_size, learning_rate=1e-3):
        super(DQN, self).__init__()
        
        # Flatten input (state vector)
        self.flatten = layers.Flatten(input_shape=input_shape)
        
        # Q-Network layers (paper specs: 1000 → 600 → output_size)
        self.fc1 = layers.Dense(1000, activation="relu")
        self.fc2 = layers.Dense(600, activation="relu")
        self.out = layers.Dense(output_size, activation=None)  # Q-values (no activation)
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        """Forward pass for computing Q-values"""
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        q_values = self.out(x)
        return q_values

def DQN_CNN(input_shape, output_size):
    inputs = layers.Input(shape=input_shape)   # e.g. (30, 5) -> 30 timesteps, 5 features

    # Feature extractor using Conv1D
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Flatten()(x)  # flatten sequence to vector

    # Fully connected layers (as per paper)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(600, activation='relu')(x)

    # Q-value outputs (one per action)
    outputs = layers.Dense(output_size, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_dqn(input_shape, output_size):
    model = models.Sequential([
        layers.Input(shape=input_shape),     # Input layer
        layers.Flatten(),                    # Flatten input
        layers.Dense(1000, activation='relu'), # First hidden layer
        layers.Dense(600, activation='relu'),  # Second hidden layer
        layers.Dense(output_size, activation='linear')  # Output Q-values
    ])
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['mae'],
    )
    return model

if __name__ == "__main__":

    # Example usage
    input_shape = (10, 10)   # Example state shape
    output_size = 4          # Example action space
    dqn = build_dqn(input_shape, output_size)
    dqn.summary()

    # CNN
    model = DQN_CNN(input_shape=(30, 5), output_size=10)
    model.summary()
