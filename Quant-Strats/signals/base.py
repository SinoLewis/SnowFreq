from abc import ABC, abstractmethod
import pandas as pd

class Signal(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        pass
