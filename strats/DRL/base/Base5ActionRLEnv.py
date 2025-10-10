import logging
from enum import Enum

from gymnasium import spaces

from base.BaseEnvironment import BaseEnvironment, Positions


logger = logging.getLogger(__name__)


class Actions(Enum):
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4


class Base5ActionRLEnv(BaseEnvironment):
    """
    Base class for a 5 action environment
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = Actions

    def set_action_space(self):
        self.action_space = spaces.Discrete(len(Actions))

    def step(self, action: int):
        """
        Logic for a single step (incrementing one candle in time)
        by the agent
        :param: action: int = the action type that the agent plans
            to take for the current step.
        :returns:
            observation = current state of environment
            step_reward = the reward from `calculate_reward()`
            _done = if the agent "died" or if the candles finished
            info = dict passed back to openai gym lib
        """
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._update_unrealized_total_profit()
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward
        self.tensorboard_log(self.actions._member_names_[action], category="actions")

        trade_type = None
        if self.is_tradesignal(action):
            if action == Actions.Neutral.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
                self._last_trade_tick = None
            elif action == Actions.Long_enter.value:
                self._position = Positions.Long
                trade_type = "enter_long"
                self._last_trade_tick = self._current_tick
            elif action == Actions.Short_enter.value:
                self._position = Positions.Short
                trade_type = "enter_short"
                self._last_trade_tick = self._current_tick
            elif action == Actions.Long_exit.value:
                self._update_total_profit()
                self._position = Positions.Neutral
                trade_type = "exit_long"
                self._last_trade_tick = None
            elif action == Actions.Short_exit.value:
                self._update_total_profit()
                self._position = Positions.Neutral
                trade_type = "exit_short"
                self._last_trade_tick = None
            else:
                print("case not defined")

            if trade_type is not None:
                self.trade_history.append(
                    {
                        "price": self.current_price(),
                        "index": self._current_tick,
                        "type": trade_type,
                        "profit": self.get_unrealized_profit(),
                    }
                )

        if (
            self._total_profit < self.max_drawdown
            or self._total_unrealized_profit < self.max_drawdown
        ):
            self._done = True

        self._position_history.append(self._position)

        info = dict(
            tick=self._current_tick,
            action=action,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            trade_duration=self.get_trade_duration(),
            current_profit_pct=self.get_unrealized_profit(),
        )

        observation = self._get_observation()
        # user can play with time if they want
        truncated = False

        self._update_history(info)

        return observation, step_reward, self._done, truncated, info

    def is_tradesignal(self, action: int) -> bool:
        """
        Determine if the signal is a trade signal
        e.g.: agent wants a Actions.Long_exit while it is in a Positions.short
        """
        return not (
            (action == Actions.Neutral.value and self._position == Positions.Neutral)
            or (action == Actions.Neutral.value and self._position == Positions.Short)
            or (action == Actions.Neutral.value and self._position == Positions.Long)
            or (action == Actions.Short_enter.value and self._position == Positions.Short)
            or (action == Actions.Short_enter.value and self._position == Positions.Long)
            or (action == Actions.Short_exit.value and self._position == Positions.Long)
            or (action == Actions.Short_exit.value and self._position == Positions.Neutral)
            or (action == Actions.Long_enter.value and self._position == Positions.Long)
            or (action == Actions.Long_enter.value and self._position == Positions.Short)
            or (action == Actions.Long_exit.value and self._position == Positions.Short)
            or (action == Actions.Long_exit.value and self._position == Positions.Neutral)
        )

    def _is_valid(self, action: int) -> bool:
        # trade signal
        """
        Determine if the signal is valid.
        e.g.: agent wants a Actions.Long_exit while it is in a Positions.short
        """
        # Agent should only try to exit if it is in position
        if action in (Actions.Short_exit.value, Actions.Long_exit.value):
            if self._position not in (Positions.Short, Positions.Long):
                return False

        # Agent should only try to enter if it is not in position
        if action in (Actions.Short_enter.value, Actions.Long_enter.value):
            if self._position != Positions.Neutral:
                return False

        return True

    def calculate_reward(self, action: int) -> float:
        # first, penalize if the action is not valid
        if not self._is_valid(action):
            return -2
        pnl = self.get_unrealized_profit()

        factor = 100

        pair = self.pair.replace(':', '')

        # you can use feature values from dataframe
        # Assumes the shifted RSI indicator has been generated in the strategy.
        rsi_now = self.raw_features[f"%-rsi-period_10_shift-1_{pair}_"
                        f"{self.config['timeframe']}"].iloc[self._current_tick]

        # reward agent for entering trades
        if (action in (Actions.Long_enter.value, Actions.Short_enter.value)
                and self._position == Positions.Neutral):
            if rsi_now < 40:
                factor = 40 / rsi_now
            else:
                factor = 1
            return 25 * factor

        # discourage agent from not entering trades
        if action == Actions.Neutral.value and self._position == Positions.Neutral:
            return -1
        max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
        trade_duration = self._current_tick - self._last_trade_tick
        if trade_duration <= max_trade_duration:
            factor *= 1.5
        elif trade_duration > max_trade_duration:
            factor *= 0.5
        # discourage sitting in position
        if self._position in (Positions.Short, Positions.Long) and \
        action == Actions.Neutral.value:
            return -1 * trade_duration / max_trade_duration
        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            if pnl > self.profit_aim * self.rr:
                factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(pnl * factor)
        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            if pnl > self.profit_aim * self.rr:
                factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(pnl * factor)
        return 0.