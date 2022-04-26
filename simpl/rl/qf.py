import torch
import torch.nn as nn

from simpl.nn import MLP, ToDeviceMixin


class MLPQF(ToDeviceMixin, nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_hidden, activation='relu'):
        super().__init__()
        self.net = MLP([state_dim + action_dim] + [hidden_dim]*n_hidden + [1], activation)

    def forward(self, batch_state, batch_action):
        concat = torch.cat([batch_state, batch_action], dim=-1)
        return self.net(concat).squeeze(-1)
    

class ContextMLPQF(MLPQF):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim, n_hidden, activation='relu'):
        super().__init__(state_dim+z_dim, action_dim, hidden_dim, n_hidden, activation)
        
    def forward(self, batch_state, batch_action, batch_z):
        concat = torch.cat([batch_state, batch_action, batch_z], dim=-1)
        return self.net(concat).squeeze(-1)
