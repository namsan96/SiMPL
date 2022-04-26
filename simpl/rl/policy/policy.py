from contextlib import contextmanager

import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch_truncnorm import TruncatedNormal

from simpl.nn import MLP, ToDeviceMixin
from simpl.math import inverse_softplus, inverse_sigmoid


class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()


class StochasticNNPolicy(ToDeviceMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.explore = None

    def dist(self, batch_state):
        raise NotImplementedError 

    def act(self, state):
        if self.explore is None:
            raise RuntimeError('explore is not set')

        batch_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            training = self.training
            self.eval()
            dist = self.dist(batch_state)
            self.train(training)

        if self.explore is True:
            batch_action = dist.sample()
        else:
            batch_action = dist.mean
        return batch_action.squeeze(0).cpu().numpy()
           
    @contextmanager
    def no_expl(self):
        explore = self.explore
        self.explore = False
        yield
        self.explore = explore

    @contextmanager
    def expl(self):
        explore = self.explore
        self.explore = True
        yield
        self.explore = explore


class NormalMLPPolicy(StochasticNNPolicy):
    def __init__(self, state_dim, action_dim, hidden_dim, n_hidden,
                 activation='relu', min_scale=0.001, max_scale=None, init_scale=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.mlp = MLP([state_dim] + [hidden_dim]*n_hidden + [2*action_dim], activation)

        self.min_scale = min_scale
        self.max_scale = max_scale
        
        if max_scale is None:
            self.pre_init_scale = inverse_softplus(init_scale)
        else:
            self.pre_init_scale = inverse_sigmoid(init_scale / max_scale)
 
    def dist(self, batch_state):
        loc, pre_scale = self.mlp(batch_state).chunk(2, dim=-1)
        if self.max_scale is None:
            scale = self.min_scale + F.softplus(self.pre_init_scale + pre_scale)
        else:
            scale = self.min_scale + self.max_scale*torch.sigmoid(self.pre_init_scale + pre_scale)
        dist = torch_dist.Normal(loc, scale)
        return torch_dist.Independent(dist, 1)


class TruncatedNormalMLPPolicy(StochasticNNPolicy):
    def __init__(self, state_dim, action_dim, action_low, action_high,
                 hidden_dim, n_hidden, activation='relu',
                 min_scale=0.001, max_scale=None, init_scale=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.mlp = MLP([state_dim] + [hidden_dim]*n_hidden + [2*action_dim], activation)

        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32))
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        if max_scale is None:
            self.pre_init_scale = inverse_softplus(init_scale)
        else:
            self.pre_init_scale = inverse_sigmoid(init_scale / max_scale)
 
    def dist(self, batch_state):
        pre_loc, pre_scale = self.mlp(batch_state).chunk(2, dim=-1)
        loc = self.action_low + (self.action_high - self.action_low) * torch.sigmoid(pre_loc)
        if self.max_scale is None:
            scale = self.min_scale + F.softplus(self.pre_init_scale + pre_scale)
        else:
            scale = self.min_scale + self.max_scale*torch.sigmoid(self.pre_init_scale + pre_scale)
        dist = TruncatedNormal(loc, scale, self.action_low, self.action_high)
        return torch_dist.Independent(dist, 1)
