from sklearn.ensemble import RandomForestClassifier
import numpy as np

class DirectionClassifier:
    def __init__(self):
        self.model = RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return np.where(self.model.predict_proba(X)[:,1] > 0.55, 1, 0)
