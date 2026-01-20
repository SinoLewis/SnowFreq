from abc import ABC, abstractmethod

class QuantModel(ABC):
    @abstractmethod
    def fit(self, X, y=None): pass
    @abstractmethod
    def predict(self, X): pass
