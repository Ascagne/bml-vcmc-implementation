import numpy as np

rng = np.random.default_rng(123)

N = 10_000
proba_feature_active = np.array([1., 0.2, 0.3, 0.5, 0.01])
d = proba_feature_active.shape[0]

X = rng.binomial(1, p=proba_feature_active, size=(N, d))
y = np.array((-3*X[:, 0]+1.2*X[:, 1]-0.5*X[:, 2] +
             0.8*X[:, 3]+3*X[:, 4]) > 0, dtype="i")
