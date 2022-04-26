import torch
import torch.distributions as torch_dist
import torch.nn as nn

from simpl.rl import StochasticNNPolicy, ContextPolicyMixin


class SpirlMLP(nn.Module):
    def __init__(self, dims, activation='relu'):
        super().__init__()
        
        layers = [
            nn.Linear(dims[0], dims[1]),
            nn.LeakyReLU(0.2, inplace=True)
        ]
                            
        prev_dim = dims[1]
        for dim in dims[2:-1]:
            layers.append(nn.Linear(prev_dim, dim, bias=False))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(prev_dim, dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpirlLowPolicy(ContextPolicyMixin, StochasticNNPolicy):
    def __init__(self, state_dim, z_dim, action_dim, hidden_dim, n_hidden, prior_state_dim=None):
        super().__init__()
        self.net = SpirlMLP([state_dim + z_dim] + [hidden_dim]*n_hidden + [action_dim])
        self.log_sigma = nn.Parameter(-50*torch.ones(action_dim))
        self.z_dim = z_dim
        self.prior_state_dim = prior_state_dim
                
    def dist(self, batch_state_z):
        if self.prior_state_dim is not None:
            batch_state_z = torch.cat([
                batch_state_z[..., :self.prior_state_dim],
                batch_state_z[..., -self.z_dim:]
            ], dim=-1)
        loc = self.net(batch_state_z)
        scale = self.log_sigma.clamp(-10, 2).exp()[None, :].expand(len(loc), -1)
        dist = torch_dist.Normal(loc, scale)
        return torch_dist.Independent(dist, 1)


class SpirlPriorPolicy(StochasticNNPolicy):    
    def __init__(self, state_dim, z_dim, hidden_dim, n_hidden, prior_state_dim=None):
        super().__init__()
        self.net = SpirlMLP([state_dim] + [hidden_dim]*n_hidden + [2*z_dim])
        self.prior_state_dim = prior_state_dim

    def dist(self, batch_state):
        if self.prior_state_dim is not None:
            batch_state = batch_state[..., :self.prior_state_dim]
        loc, log_scale = self.net(batch_state).chunk(2, dim=-1)
        dist = torch_dist.Normal(loc, log_scale.clamp(-10, 2).exp())
        return torch_dist.Independent(dist, 1)
    
    def dist_param(self, batch_state):
        if self.prior_state_dim is not None:
            batch_state = batch_state[..., :self.prior_state_dim]
        loc, log_scale = self.net(batch_state).chunk(2, dim=-1)
        return loc, log_scale.clamp(-10, 2)
