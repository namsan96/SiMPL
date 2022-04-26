from simpl.rl import ContextPolicyMixin
from simpl.alg.spirl import PriorResidualNormalMLPPolicy


class ContextPriorResidualNormalMLPPolicy(ContextPolicyMixin, PriorResidualNormalMLPPolicy):
    def __init__(self, prior_policy, state_dim, action_dim, z_dim, hidden_dim, n_hidden,
                 prior_state_dim=None, policy_exclude_dim=None, activation='relu'):
        if prior_state_dim is None:
            prior_state_dim = state_dim
        super().__init__(
            prior_policy, state_dim + z_dim, action_dim, hidden_dim, n_hidden, 
            prior_state_dim, policy_exclude_dim, activation
        )
        self.z_dim = z_dim
