import torch
import numpy as np

def get_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = torch.expand_dims(torch.sum(torch.power(X, 2), axis=1), 1)
    dists = x2 + torch.transpose(x2) - 2 * torch.dot(X, torch.transpose(X))
    return dists

def get_shape(x):
    dims = float(x.shape(x)[1])
    N    = float(x.shape(x)[0])
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = get_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims / 2.0)*torch.log(2*np.pi*var)
    lprobs = torch.logsumexp(-dists2, axis=1) - torch.log(N) - normconst
    h = -torch.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x, 4*var)
    return val + np.log(0.25)*dims / 2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims / 2.0) * (np.log(2 * np.pi * var) + 1)

