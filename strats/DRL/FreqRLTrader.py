import pandas as pd
import json


def preprocess_drl_inputs(sent_df, price_df, save_path="data/drl_inputs.csv"):
    
    # Merge on datetime (inner join ensures alignment)
    merged = pd.merge_asof(
        sent_df.sort_values("datetime"),
        price_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",   # align closest timestamps
        tolerance=pd.Timedelta("5min")  # tolerate small lag between news & price
    )

    # Drop NA after merge
    merged.dropna(inplace=True)

    # Construct DRL features
    merged["return"] = merged["price_true"].pct_change()  # actual returns as reward signal
    features = merged[["sentiment_score_pos", "sentiment_score_neu", "sentiment_score_neg", "price_pred"]]
    labels = merged["price_true"]
    print(merged)
    # Save for DRL training
    merged.to_csv(save_path, index=False)
    print(f"✅ DRL preprocessed dataset saved: {save_path}")

    return features, labels, merged

import numpy as np
from collections import deque

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def create_sequences(data, target_col, seq_len=60):
    """
    Create sequential window data for supervised learning.
    
    Args:
        data (pd.DataFrame or np.ndarray): Input features including target.
        target_col (str or int): Column name (if df) or index (if np.ndarray) for the target variable.
        seq_len (int): Number of timesteps per sequence.

    Returns:
        X (np.ndarray): Feature sequences of shape (N, seq_len, num_features).
        y (np.ndarray): Targets of shape (N,).
    """
    sequential_data = []
    prev_days = deque(maxlen=seq_len)

    # Handle dataframe vs ndarray
    if hasattr(data, "values"):  
        data = data.values  

    target_idx = target_col if isinstance(target_col, int) else None
    
    for row in data:
        # add features except target
        if target_idx is not None:
            features = np.delete(row, target_idx)
            target = row[target_idx]
        else:
            raise ValueError("When passing numpy array, target_col must be int (index).")

        prev_days.append(features)
        
        if len(prev_days) == seq_len:
            sequential_data.append([np.array(prev_days), target])

    # Split into X, y
    X, y = [], []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)

config_file = 'data/config.json'
with open(config_file, "r") as f:
    config = json.load(f)

sentiment_file = 'data/sentiment_preds.csv'
price_file = 'data/price_preds.csv'
sent_df = pd.read_csv(sentiment_file, parse_dates=["datetime"])
price_df = pd.read_csv(price_file, parse_dates=["datetime"])
features, labels, merged = preprocess_drl_inputs(sent_df, price_df)
merged["datetime"] = merged["datetime"].astype("int64")

# print(features.shape, labels.shape, merged.shape)
# print(features.columns, merged.columns)

X, y = create_sequences(merged, target_col=8, seq_len=3)
# X, y = sliding_windows(merged, seq_length=3)

print("X shape:", X.shape)   # (N, 3, 5) -> 5 features (open, high, low, close, volume)
print("y shape:", y.shape)   # (N,)
print("Sample X[0]:\n", X[0])
print("Sample y[0]:", y[0])

from base.Base3ActionRlEnv import Base3ActionRLEnv
# from DRLAgent import DQNAgent
from CNNAgent import CNNAgent
# # Instantiate env (you may need to pass kwargs like dataframe, config, etc.)
env = Base3ActionRLEnv(df=merged, prices=labels, config=config, reward_kwargs=config['freqai']['rl_config']['model_reward_parameters'])
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = CNNAgent(state_size, action_size)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'DRL_CNN'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, action_size)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)