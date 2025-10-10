import numpy as np
import random, time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        # Environment details
        self.state_size = state_size      # e.g. num_features
        self.action_size = action_size    # 3 actions: Neutral, Buy, Sell

        # Hyperparameters
        self.gamma = 0.99                 # Discount factor
        self.epsilon = 1.0                # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 5

        # Replay memory
        self.memory = deque(maxlen=50_000)

        # Main & target networks
        self.model = self._build_model_rnn()
        self.target_model = self._build_model()
        self.update_target_network()

        # Counter for updating target net
        self.target_update_counter = 0

    def _build_model_rnn(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.state_size), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))

        print(model.summary())
        return model

    def _build_model_cnn(self):
        """
        Simple MLP for trading features.
        You could replace with LSTM if using sequences.
        """
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        # Predict Q-values for current states
        q_values = self.model.predict(states, verbose=0)
        # Predict Q-values for next states (using target network)
        q_next = self.target_model.predict(next_states, verbose=0)

        # Update Q-values
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next[i])
            q_values[i][actions[i]] = target

        self.model.fit(states, q_values, epochs=1, verbose=0, batch_size=self.batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.update_target_network()
