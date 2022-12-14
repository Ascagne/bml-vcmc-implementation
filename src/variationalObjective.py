import numpy as np
from numba import njit


@njit
def variational_obj(W, betas, X, y, sigma=1):
    K, nsamples, d = betas.shape
    N = X.shape[0]
    mu = np.zeros((ncores, d))
    for k in range(K):
        for j in range(nsamples):
            mu[k, :] += betas[k, j]
    # print(mu.shape)
    S = np.zeros((ncores, d, d))
    for k in range(K):
        for i in range(nsamples):
            S[k] += np.outer(betas[k, i], betas[k, i])
    S /= nsamples
    # print(S.shape)
    L = 0
    for k in range(K):
        L -= (1/(2*sigma**2))*tr_prod(S[k], np.dot(W[k].T, W[k]))
        for l in range(K):
            if l != k:
                L -= (2/(2*sigma**2)) * \
                    tr_prod(np.dot(np.outer(mu[k], mu[l]), W[l].T), W[k])
    for n in range(N):
        if y[n] == 0:
            L += expect_log_1mphi_n(W, betas, X, n)
        else:
            L += expect_log_phi_n(W, betas, X, n)
    for k in range(K):
        L += (1/K)*np.log(np.linalg.det(W[k]))
    return L
