import numpy as np
from scipy.stats import truncnorm

def gibbs_sampler(X,y,nsim=10000,burnin=2000,sigma=1):
    rng=np.random.default_rng()
    n,d=X.shape
    thetas=np.zeros((nsim-burnin,d))
    z=np.zeros(n)
    Sigma=np.linalg.inv(1/(sigma**2)*np.eye(d)+np.dot(X.T,X))
    fac_mu_theta=np.dot(Sigma,X.T)
    mu_z=np.zeros(n)
    theta=np.zeros(d)
    for i in range(nsim):
        mu_z=np.dot(X,theta)
        z[y==0]=truncnorm.rvs(-np.inf,-mu_z[y==0],loc=mu_z[y==0],scale=1)
        z[y==1]=truncnorm.rvs(-mu_z[y==1],np.inf,loc=mu_z[y==1],scale=1)
        mu_theta=np.dot(fac_mu_theta,z)
        theta=rng.multivariate_normal(mu_theta,Sigma)
        if i>=burnin:
            thetas[i-burnin,:]=theta
    return thetas 

def cmc(X,y,ncores=10,nsim=5000,burnin=1000):
    n,d=X.shape
    nsamples=nsim-burnin
    sub_thetas=np.zeros((ncores,nsamples,d))
    shard=int(n/ncores)
    for i in range(ncores):
        sub_thetas[i,:,:]=gibbs_sampler(X[i*shard:(i+1)*shard,:],y[i*shard:(i+1)*shard],nsim,burnin)
    return sub_thetas