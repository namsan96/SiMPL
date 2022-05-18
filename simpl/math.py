import numpy as np
import torch.distributions as torch_dist
import scipy


def clipped_kl(a, b, clip=20):
    kls = torch_dist.kl_divergence(a, b)
    scales =  kls.detach().clamp(0, clip) / kls.detach()
    return kls*scales

def inverse_softplus(x):
    return float(np.log(np.exp(x) - 1))

def inverse_sigmoid(x):
    return float(-np.log(1/x - 1))

def discount(x, gamma):
    # y[n] = 1/a[0]  X  ( b[0] X x[n] - a[1]  X y[n-1] )
    # g[n]=               1   X r[n]  + gamma X g[n-1]
    return scipy.signal.lfilter(b=[1], a=[1, -gamma], x=x[..., ::-1])[..., ::-1].copy()
