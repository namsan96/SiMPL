import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F

from simpl.math import clipped_kl, inverse_softplus
from simpl.nn import ToDeviceMixin


class ConstrainedSAC(ToDeviceMixin, nn.Module):
    def __init__(self, policy, prior_policy, qfs, buffer,
                 discount=0.99, tau=0.005, policy_lr=3e-4, qf_lr=3e-4,
                 auto_alpha=True, init_alpha=0.1, alpha_lr=3e-4, target_kl=1,
                 kl_clip=20, increasing_alpha=False):
        super().__init__()
        
        self.policy = policy
        self.prior_policy = prior_policy
        self.qfs = nn.ModuleList(qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in qfs])
        self.buffer = buffer

        self.discount = discount
        self.tau = tau

        self.policy_optim = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=qf_lr) for qf in qfs]

        self.auto_alpha = auto_alpha
        pre_init_alpha = inverse_softplus(init_alpha)
        if auto_alpha is True:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=alpha_lr)
            self.target_kl = target_kl
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.kl_clip = kl_clip
        self.increasing_alpha = increasing_alpha
        
    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
    
    def to(self, device):
        self.policy.to(device)
        return super().to(device)

    def step(self, batch_size):
        stat = {}
        batch = self.buffer.sample(batch_size).to(self.device)

        # qfs
        with torch.no_grad():
            target_qs = self.compute_target_q(batch)

        qf_losses = []
        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            qs = qf(batch.states, batch.actions)
            qf_loss = (qs - target_qs).pow(2).mean()

            qf_optim.zero_grad()
            qf_loss.backward()
            qf_optim.step()

            qf_losses.append(qf_loss)
        self.update_target_qfs()
        
        stat['qf_loss'] = torch.stack(qf_losses).mean()

        # policy
        dists = self.policy.dist(batch.states)
        policy_actions = dists.rsample()
        with torch.no_grad():
            prior_dists = self.prior_policy.dist(batch.states)
        kl = torch_dist.kl_divergence(dists, prior_dists).mean(0)
        min_qs = torch.min(*[qf(batch.states, policy_actions) for qf in self.qfs])

        policy_loss = - min_qs.mean(0) + self.alpha * kl

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        stat['policy_loss'] = policy_loss
        stat['kl'] = kl
        stat['mean_policy_scale'] = dists.base_dist.scale.abs().mean()

        # alpha
        if self.auto_alpha is True:
            alpha_loss = (self.alpha * (self.target_kl - kl.detach())).mean()
            if self.increasing_alpha is True:
                alpha_loss = alpha_loss.clamp(-np.inf, 0)
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            stat['alpha_loss'] = alpha_loss
            stat['alpha'] = self.alpha
    
        return stat

    def compute_target_q(self, batch):
        dists = self.policy.dist(batch.next_states)
        actions = dists.sample()
        
        with torch.no_grad():
            prior_dists = self.prior_policy.dist(batch.next_states)
        kls = clipped_kl(dists, prior_dists, clip=self.kl_clip)
        min_qs = torch.min(*[target_qf(batch.next_states, actions) for target_qf in self.target_qfs])
        soft_qs = min_qs - self.alpha*kls
        return batch.rewards + (1 - batch.dones)*self.discount*soft_qs

    def update_target_qfs(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            for param, target_param in zip(qf.parameters(), target_qf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

