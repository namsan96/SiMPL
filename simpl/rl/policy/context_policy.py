from contextlib import contextmanager

import numpy as np
import torch

from .policy import NormalMLPPolicy, TruncatedNormalMLPPolicy


class ContextPolicyMixin:
    z_dim = NotImplemented
    z = None

    @contextmanager
    def condition(self, z):
        if type(z) != np.ndarray or z.shape != (self.z_dim, ):
            raise ValueError(f'z should be np.array with shape {self.z_dim}, but given : {z}')
        prev_z = self.z
        self.z = z
        yield
        self.z = prev_z

    def act(self, state):
        if self.z is None:
            raise RuntimeError('z is not set')
        state_z = np.concatenate([state, self.z], axis=0)
        return super(ContextPolicyMixin, self).act(state_z)

    def dist(self, batch_state_z):
        return super(ContextPolicyMixin, self).dist(batch_state_z)

    def dist_with_z(self, batch_state, batch_z):
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.dist(batch_state_z)

    
class ContextNormalMLPPolicy(ContextPolicyMixin, NormalMLPPolicy):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim, n_hidden,
                 activation='relu', min_scale=0.001, max_scale=None, init_scale=0.1):
        super().__init__(
            state_dim + z_dim, action_dim, hidden_dim, n_hidden, activation,
            min_scale, max_scale, init_scale
        )
        self.z_dim = z_dim

        
class ContextTruncatedNormalMLPPolicy(ContextPolicyMixin, TruncatedNormalMLPPolicy):
    def __init__(self, state_dim, action_dim, z_dim, action_low, action_high,
                 hidden_dim, n_hidden, activation='relu',
                 min_scale=0.001, max_scale=None, init_scale=0.1):
        super().__init__(
            state_dim + z_dim, action_dim, action_low, action_high,
            hidden_dim, n_hidden, activation,
            min_scale, max_scale, init_scale
        )
        self.z_dim = z_dim
