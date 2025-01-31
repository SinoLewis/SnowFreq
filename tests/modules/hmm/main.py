import numpy as np

from hmmlearn import hmm

np.random.seed(42)


model = hmm.GaussianHMM(n_components=3, init_params="st", covariance_type="full")

model.startprob_ = np.array([0.6, 0.3, 0.1])

model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])

# model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.means_ = np.array([[0.0, -10.0], [5.0, -3.0], [5.0, 10.0]])

model.covars_ = np.tile(np.identity(2), (3, 1, 1))

X, Z = model.sample(100)
print(np.tile(np.identity(2), (3, 1, 1)))
print(np.tile(np.identity(3), (2, 1, 1)))
print(np.tile(np.identity(3), (3, 1, 1)))

model.fit(X)

# print(X)
# print(model.n_features)

Z2 = model.predict(X)
# print(Z2)