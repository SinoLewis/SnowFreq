from base.stocks_env import StocksEnv
from stable_baselines3 import DQN
import pandas as pd
from colorama import Fore 

# TODO: Move to Features.py
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

# Create environment
sentiment_file = 'data/sentiment_preds.csv'
price_file = 'data/price_preds.csv'
sent_df = pd.read_csv(sentiment_file, parse_dates=["datetime"])
price_df = pd.read_csv(price_file, parse_dates=["datetime"])
features, labels, merged = preprocess_drl_inputs(sent_df, price_df)
env = StocksEnv(df=merged, frame_bound=(50, 100), window_size=10)

# Initialize DQN agent
model = DQN(
    "MlpPolicy", 
    env,
    learning_rate=1e-4,
    buffer_size=1000_000,
    learning_starts=100,
    batch_size=32,
    gamma=0.99,
    target_update_interval=10_000,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=10_000, log_interval=4)

# Save model
model.save("dqn_rl_trader")

episodes = range(100)
for episode in episodes: 
    # Test the trained agent
    print(Fore.LIGHTBLUE_EX + "\n\n\n⚡️ I'm at the pre-reset" + Fore.RESET)
    obs, _ = env.reset()
    done = False
    score = 0 
    print(Fore.LIGHTYELLOW_EX +  f"🚀 Starting step loop {done}"+ Fore.RESET)
    while not done: 
        action, _state = model.predict(obs, deterministic=True)
        print(Fore.LIGHTYELLOW_EX +f"🏋🏽‍♀️ Actions selected: {action}, step about to be taken"+ Fore.RESET)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(Fore.LIGHTGREEN_EX + f"🤖 Step taken: {reward,done}"+ Fore.RESET)
        score+=reward
        print(Fore.LIGHTMAGENTA_EX + '🏆 Reward:{} Score:{}'.format(episode, score, done)+ Fore.RESET)
    print(Fore.LIGHTRED_EX + 'Episode DONE \n\n\n' + Fore.RESET) 
print("Attempting to close")
env.close()
print("Closed")      