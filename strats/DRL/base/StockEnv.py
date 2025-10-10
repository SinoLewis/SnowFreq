import numpy as np
import torch.optim as optim
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

# 3. Stock Trading Environment
# ---------------------------
class StockTradingEnv(gym.Env):
    """
    Gymnasium-compatible environment matching the paper:
    - Discrete actions: 0=Buy,1=BuyMore,2=Sell,3=SellMore
    - Observation: user-provided vector of indicators + predicted prices + sentiment
    - Reward: Profit (positive) or 3x loss (negative)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_series: np.ndarray,
        feature_matrix: np.ndarray,
        initial_cash: float = 1_000.0,
        max_position: float = 1.0,
        transaction_cost: float = 0.0,
        gamma: float = 0.99,
    ):
        assert len(price_series) == feature_matrix.shape[0]
        self.prices = price_series  # e.g., closing price per day
        self.features = feature_matrix  # shape (T, D)
        self.n_steps = len(price_series)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0  # fraction of portfolio invested: -1 (short) to +1 (long)
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.gamma = gamma

        self.current_step = 0

        # Observation space is the feature vector plus current position and cash fraction
        obs_dim = self.features.shape[1] + 2  # add position and cash ratio
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._reset_internal()

    def _reset_internal(self):
        self.cash = self.initial_cash
        self.position = 0.0
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.last_price = self.prices[0]

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        feat = self.features[self.current_step]
        position = np.array([self.position], dtype=np.float32)
        cash_ratio = np.array([self.cash / (self.portfolio_value + 1e-8)], dtype=np.float32)
        obs = np.concatenate([feat.astype(np.float32), position, cash_ratio], axis=0)
        return obs

    def step(self, action: int):
        """
        Action mapping (2 actions):
        0 = Buy  (increase position by fixed increment)
        1 = Sell (decrease position by fixed increment)
        """

        price = self.prices[self.current_step]
        prev_portfolio = self._compute_portfolio_value(price)

        # Define position change amounts
        delta = 0.1 if action == 0 else -0.1  # single increment size

        # Apply position change with clipping
        new_position = np.clip(self.position + delta, -self.max_position, self.max_position)

        # Simulate transaction cost as proportion of traded amount
        trade_size = abs(new_position - self.position)
        cost = trade_size * self.transaction_cost * self.portfolio_value

        self.position = new_position

        # Update portfolio value based on new price and position
        self.portfolio_value = self._compute_portfolio_value(price) - cost

        # Reward: profit (positive) or 3x loss
        profit = self.portfolio_value - prev_portfolio
        reward = profit if profit >= 0 else 3 * profit

        # Advance step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        next_obs = (
            self._get_observation()
            if not done
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "price": price,
        }

        return next_obs, reward, done, False, info


    def _compute_portfolio_value(self, price):
        # Position is in [-1,1]: multiply by portfolio to get exposure
        return self.cash + self.position * (price - self.last_price)  # simple PnL; could improve with actual holdings

    def render(self, mode="human"):
        print(
            f"Step {self.current_step} | Price {self.prices[self.current_step]:.2f} | Pos {self.position:.2f} | Portfolio {self.portfolio_value:.2f}"
        )

