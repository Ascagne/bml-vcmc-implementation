import numpy as np
from numba import njit
from src.functions import big_expect_q


@njit
def gradient_W(W, S, mu, betas, k, X, y, sigma=1):
    """Compute the gradient of the relaxed variational objective wrt matrix Wk
    """
    K, nsamples, d = betas.shape
    N = X.shape[0]
    G = (1/(sigma**2))*S[k]@W[k].T
    for l in range(K):
        if l != k:
            G += (1/(sigma**2)) * \
                np.outer(mu[k], mu[l])@W[l].T + W[l]@np.outer(mu[l], mu[k])
    for n in range(N):
        G += np.outer(big_expect_q(W, betas, X, y, k, n), X[n])
    G += np.linalg.inv(W[k])/K
    return G


def gradient_descent_W(W_0, learning_rate, S, mu, betas, X, y, niter=10, tolerance=1e-06):
    W = W_0
    K, nsamples, d = betas.shape
    diff = np.zeros((K, d, d))
    for i in range(niter):
        print("iter:", i)
        for k in range(K):
            diff[k] = -learning_rate * gradient_W(
                W=W,
                S=S,
                mu=mu,
                betas=betas,
                k=k,
                X=X,
                y=y
            )
        # if np.all(np.abs(diff) <= tolerance):
        #     break
        W += diff
    return W
