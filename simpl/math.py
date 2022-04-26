import numpy as np
import torch.distributions as torch_dist


def clipped_kl(a, b, clip=20):
    kls = torch_dist.kl_divergence(a, b)
    scales =  kls.detach().clamp(0, clip) / kls.detach()
    return kls*scales

def inverse_softplus(x):
    return float(np.log(np.exp(x) - 1))

def inverse_sigmoid(x):
    return float(-np.log(1/x - 1))
