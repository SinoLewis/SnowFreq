# === Standard library imports ===
import json

# === Third-party imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# from gym_anytrading import gym_anytrading
# import gymnasium as gym
# import gym_anytrading
from base.stocks_env import StocksEnv
# from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
# from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

# === Local imports ===
from agents.Agent import DQNAgent
from Features import debug_dataframe

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
    merged["datetime"] = merged["datetime"].astype("int64")
    merged = merged.rename(columns={"price_true": "Close"})
    # print(merged)
    # Save for DRL training
    merged.to_csv(save_path, index=False)
    print(f"✅ DRL preprocessed dataset saved: {save_path}")

    return features, labels, merged

def train_drl(env, agent):
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset(seed=2025)[0]

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            action = env.action_space.sample()
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()
                print("info:", info)

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            step += 1
    if done:
        print("info:", info)

def predict_drl(env, agent):
    """
    Evaluate a trained DRL agent on unseen data.
    """
    # Load trained weights model_path="models/dqn_trader.h5"
    # agent.model.load_weights(model_path)
    # print(f"✅ Loaded trained model from: {model_path}")

    state = env.reset(seed=2025)[0]
    done = False
    total_reward = 0
    actions, prices, rewards, pred_actions = [], [], [], []

    while not done:
        # Deterministic policy (no exploration)
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        actions.append(action)
        rewards.append(reward)
        prices.append(info.get("price", np.nan))
        
        q_vals = agent.get_qs(new_state)
        act = int(np.argmax(q_vals))
        pred_actions.append(act)
        state = new_state

    print(f"✅ Info: {info}")
    print(f"✅ Total Reward from Test Run: {total_reward:.4f}")

    # Optional visualization
    env.unwrapped.render_all()
    plt.show()
    # len(preds) = number of times the loop runs before done == True.
    return pd.DataFrame({"action": actions, "reward": rewards, "price": prices, "pred_actions": pred_actions})

def main():
    config_file = 'data/config.json'
    with open(config_file, "r") as f:
        config = json.load(f)

    sentiment_file = 'data/sentiment_preds.csv'
    price_file = 'data/price_preds.csv'
    sent_df = pd.read_csv(sentiment_file, parse_dates=["datetime"])
    price_df = pd.read_csv(price_file, parse_dates=["datetime"])
    features, labels, merged = preprocess_drl_inputs(sent_df, price_df)
    
    # debug_dataframe(merged)

    env = StocksEnv(df=merged, frame_bound=(50, 100), window_size=10)
    observation_shape = env.observation_space.shape  # This is the shape of the price data array
    agent = DQNAgent(observation_shape)
    print("\n[INFO] ENV Architecture:\n")
    print("DATA Shape: ", merged.shape)
    print("ENV Shape: ",  observation_shape)
    print("\n[INFO] Model Architecture:\n")
    print(agent.model.summary())   # This will print a full summary of layers
    
    # train_drl(env, agent)
    preds = predict_drl(env, agent)
    print("✅ Predictions Actions", preds)

    # plt.cla()
    # env.unwrapped.render_all()
    # plt.show()

    return agent

# Environment settings
EPISODES = 20_000
#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

if __name__ == "__main__":
    main()
