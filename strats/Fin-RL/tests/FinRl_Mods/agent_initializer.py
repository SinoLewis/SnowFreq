from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl import config

def train_agent(env_train, model_type="ppo", total_timesteps=50_000):
    """Initialize, train, and return DRL model."""
    agent = DRLAgent(env=env_train)
    model_params = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }

    model = agent.get_model(model_type, model_kwargs=model_params)
    log_path = f"{config.RESULTS_DIR}/{model_type}"
    model.set_logger(configure(log_path, ["stdout", "csv", "tensorboard"]))
    trained_model = agent.train_model(model=model, tb_log_name=model_type, total_timesteps=total_timesteps)
    print(f"✅ {model_type.upper()} model trained successfully.")
    return trained_model
