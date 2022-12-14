import numpy as np


def uniform_agg(sub_thetas):
    """Perform uniform aggregation (CMC)

    Args:
        sub_thetas (np.ndarray): samples from subposterior distributions. Of shape (ncores,nsamples,d)

    Returns:
        Aggregated samples: np.ndarray of shape (d,)
    """
    return sub_thetas.mean(axis=(0, 1))


def gaussian_agg(sub_thetas):
    """Perform gaussian aggregation (CMC)

    Args:
        sub_thetas (np.ndarray): samples from subposterior distributions. Of shape (ncores,nsamples,d)

    Returns:
        Aggregated samples: np.ndarray of shape (d,)
    """
    ncores, nsamples, d = sub_thetas.shape
    estimate = np.zeros((nsamples, d))
    sum_w = np.zeros(d)
    for i in range(ncores):
        w = 1/np.var(sub_thetas[i, :, :], axis=0, ddof=1)
        sum_w += w
        estimate += w*sub_thetas[i, :, :]
    estimate = estimate/sum_w
    return estimate.mean(axis=0)
