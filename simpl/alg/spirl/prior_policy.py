import copy

import torch
import torch.distributions as torch_dist
import torch.nn.functional as F

from simpl.nn import MLP
from simpl.rl.policy import StochasticNNPolicy


def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)
    

class PriorResidualNormalMLPPolicy(StochasticNNPolicy):
    def __init__(self, prior_policy, state_dim, action_dim, hidden_dim, n_hidden,
                 prior_state_dim=None, policy_exclude_dim=None, activation='relu', min_scale=0.001):
        super().__init__()
        self.action_dim = action_dim
        self.mlp = MLP([state_dim] + [hidden_dim]*n_hidden + [2*action_dim], activation)
        
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
        self.prior_state_dim = prior_state_dim
        self.policy_exclude_dim = policy_exclude_dim
        self.min_scale = min_scale

    def dist(self, batch_state):
        self.prior_policy.eval()
        
        batch_prior_state = batch_state
        batch_policy_state = batch_state
        if self.prior_state_dim is not None:
            batch_prior_state = batch_state[..., :self.prior_state_dim]
        if self.policy_exclude_dim is not None:
            batch_policy_state = batch_state[..., self.policy_exclude_dim:]
            
        prior_locs, prior_log_scales = self.prior_policy.dist_param(batch_prior_state)
        prior_pre_scales = inverse_softplus(prior_log_scales.exp())
        
        res_locs, res_pre_scales = self.mlp(batch_policy_state).chunk(2, dim=-1)

        dist = torch_dist.Normal(
            prior_locs + res_locs,
            self.min_scale + F.softplus(prior_pre_scales + res_pre_scales)
        )
        return torch_dist.Independent(dist, 1)
