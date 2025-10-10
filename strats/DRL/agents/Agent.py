import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import time
import random

# Agent class
class DQNAgent:
    DISCOUNT = 0.99
    REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
    MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
    UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
    WINDOW_SIZE = 10

    def __init__(self, state_shape):
        self.state_shape = state_shape
        # Main model
        self.model = self.create_model()
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    def create_model(self):
        model = Sequential()

        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.WINDOW_SIZE, self.state_shape[1])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64, activation='relu'))

        model.add(Dense(self.state_shape[1], activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):

        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # No division by 255 (use raw scaled/normalized features instead)
        current_states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X, y = [], []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[] if terminal_state else None
        )

        # Update target network by steps, not just episodes
        self.target_update_counter += 1
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
