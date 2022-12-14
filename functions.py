import numpy as np
import math
from numba import njit

@njit
def tr_prod(A,B):
    return np.trace(np.dot(A,B))

@njit
def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

@njit
def phi_deriv(x):
    return np.exp(-(x**2)/2)/np.sqrt(2*np.pi)

@njit
def inside_phi_n(W,betas,X,i,n):
    K,N,d=betas.shape
    res=0
    for k in range(K):
        res+=tr_prod(W[k],np.outer(betas[k,i],X[n]))
    return res

@njit
def expect_log_phi_n(W,betas,X,n):
    K,N,d=betas.shape
    res=0
    for i in range(N):
        res+=np.log(phi(inside_phi_n(W,betas,X,i,n)))
    return res/N

@njit
def expect_log_1mphi_n(W,betas,X,n):
    K,N,d=betas.shape
    res=0
    for i in range(N):
        res+=np.log(1-phi(inside_phi_n(W,betas,X,i,n)))
    return res/N

@njit
def big_scalar(W,betas,X,y,n,i):
    inside_phi=inside_phi_n(W,betas,X,i,n)
    if abs(inside_phi)>5:
        return 0
    elif y[n]==phi(inside_phi):
        return 0
    else:
        return (y[n]-phi(inside_phi))*phi_deriv(inside_phi)/(phi(inside_phi)*(1-phi(inside_phi)))

@njit
def big_expect_q(W,betas,X,y,k,n):
    K,nsamples,d=betas.shape
    N=X.shape[0]
    res=np.zeros(d)
    for i in range(nsamples):
        res+=big_scalar(W,betas,X,y,n,i)*betas[k,i]
    return res/nsamples