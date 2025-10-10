import numpy as np
import pandas as pd
from collections import defaultdict

class TradingEnvQLearning:
    def __init__(self, df: pd.DataFrame, bins=10, initial_balance=1000):
        """
        Tabular Q-learning environment for trading.

        :param df: DataFrame with OHLCV + features (must include 'Close', 'Volume', optional RSI etc.)
        :param bins: number of discretization bins for features
        :param initial_balance: starting cash balance
        """
        self.df = df.reset_index(drop=True)
        self.bins = bins
        self.initial_balance = initial_balance

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = [0, 1, 2]

        # Internal state
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0   # number of shares (0 = no position)
        self.entry_price = None
        self.done = False

        return self._get_state()

    def step(self, action):
        """
        Take an action and move env one step forward.
        Returns (next_state, reward, done, info)
        """
        reward = 0
        current_price = self.df.loc[self.current_step, "close"]

        # --- Trading logic ---
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                reward = -1  # small penalty for entering
        elif action == 2:  # Sell
            if self.position == 1:  # only if we have a position
                profit = current_price - self.entry_price
                reward = profit * 10  # scale reward
                self.balance += profit
                self.position = 0
                self.entry_price = None
        else:  # Hold
            reward = -0.1  # small time penalty

        # --- Step forward ---
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        """
        Discretized state (price change %, volume ratio, RSI bucket if available).
        """
        if self.current_step == 0:
            return (0, 0, 0)

        price_now = self.df.loc[self.current_step, "close"]
        price_prev = self.df.loc[self.current_step - 1, "close"]
        vol_now = self.df.loc[self.current_step, "volume"]
        vol_prev = self.df.loc[self.current_step - 1, "volume"]

        # Features
        price_change = (price_now - price_prev) / price_prev
        vol_change = (vol_now - vol_prev) / (vol_prev + 1e-6)
        rsi_val = self.df.loc[self.current_step, "RSI"] if "RSI" in self.df.columns else 50

        # Discretize
        price_bucket = int(np.digitize(price_change, np.linspace(-0.05, 0.05, self.bins)))
        vol_bucket = int(np.digitize(vol_change, np.linspace(-1, 1, self.bins)))
        rsi_bucket = int(np.digitize(rsi_val, np.linspace(0, 100, self.bins)))

        return (price_bucket, vol_bucket, rsi_bucket)

def train(env, q_table):
    ep_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0

        done = False
        while not done:
            # Epsilon-greedy
            if np.random.random() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = np.random.randint(0, len(env.action_space))

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Q-update
            max_future_q = np.max(q_table[next_state])
            current_q = q_table[state][action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state][action] = new_q

            state = next_state

        epsilon *= EPS_DECAY
        ep_rewards.append(episode_reward)

    return ep_rewards

EPISODES = 5000
epsilon = 1.0
EPS_DECAY = 0.999
LEARNING_RATE = 0.1
DISCOUNT = 0.95

if __name__ == "__main__":

    # Load OHLCV data
    df = pd.read_feather("../data/BTC_USDT-1d.feather")  # must have Close, Volume, optional RSI column

    env = TradingEnvQLearning(df, bins=10)
    q_table = defaultdict(lambda: [np.random.uniform(-1, 0) for _ in range(len(env.action_space))])

    ep_rewards = train(env, q_table)
    print(ep_rewards)