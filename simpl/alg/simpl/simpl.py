import copy
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F

from simpl.collector import Batch
from simpl.math import clipped_kl, inverse_softplus
from simpl.nn import ToDeviceMixin, TensorBatch

        
class LearnableKLReg(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        init_pre_value = inverse_softplus(init_value)
        self.param = nn.Parameter(torch.tensor(init_pre_value, dtype=torch.float32))
    
    def forward(self):
        return F.softplus(self.param)
        

class KLRegTrainer:
    def __init__(self, reg, target_kl, lr=3e-4, allow_decrease=True):
        self.reg = reg
        self.target_kl = target_kl
        self.allow_decrease = allow_decrease
        self.optim = torch.optim.Adam(reg.parameters(), lr=lr)
        
    def step(self, regs, kls):
        loss = (regs * (self.target_kl - kls.detach())).mean()

        self.optim.zero_grad()
        loss.backward()
        if self.allow_decrease is False:
            for p in filter(lambda p: p.grad is not None, self.reg.parameters()):
                p.grad.data.clamp_(min=-np.inf, max=0)
        self.optim.step()


class Simpl(ToDeviceMixin, nn.Module):
    def __init__(self, policy, prior_policy, qfs, encoder, enc_buffers, buffers,
                 init_enc_prior_reg, init_enc_post_reg, init_policy_prior_reg, init_policy_post_reg,
                 target_enc_prior_kl, target_enc_post_kl, target_policy_prior_kl, target_policy_post_kl,
                 kl_clip=20, policy_lr=3e-4, qf_lr=3e-4, enc_reg_lr=3e-4, policy_reg_lr=3e-4,
                 gamma=0.99, tau=0.005):
        super().__init__()
        
        self.policy = policy
        self.prior_policy = prior_policy
        self.qfs = nn.ModuleList(qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in qfs])
        self.encoder = encoder

        self.enc_buffers = enc_buffers
        self.buffers = buffers

        self.gamma = gamma
        self.tau = tau

        self.policy_optim = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=qf_lr) for qf in qfs]
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=qf_lr)
        
        # KL regularization coefficients
        self.enc_prior_reg = LearnableKLReg(init_enc_prior_reg)
        self.enc_post_regs = nn.ModuleList([LearnableKLReg(init_enc_post_reg) for _ in range(len(buffers))])
        self.policy_prior_reg = LearnableKLReg(init_policy_prior_reg)
        self.policy_post_reg = LearnableKLReg(init_policy_post_reg)
        
        self.enc_prior_reg_trainer = KLRegTrainer(
            self.enc_prior_reg, target_enc_prior_kl,
            enc_reg_lr, allow_decrease=False
        )
        self.enc_post_reg_trainer = KLRegTrainer(
            self.enc_post_regs, target_enc_post_kl,
            enc_reg_lr, allow_decrease=False
        )
        self.policy_prior_reg_trainer = KLRegTrainer(
            self.policy_prior_reg, target_policy_prior_kl,
            policy_reg_lr, allow_decrease=False
        )
        self.policy_post_reg_trainer = KLRegTrainer(
            self.policy_post_reg, target_policy_post_kl,
            policy_reg_lr, allow_decrease=False
        )
        
        self.kl_clip = kl_clip

    def to(self, device):
        self.encoder = self.encoder.to(device)
        return super().to(device)
    
    def make_task_batch(self, n_task, enc_size, enc_regs):
        task_indices = np.random.randint(len(self.buffers), size=n_task)
        enc_batches = []
        batch_transitions = []
        for task_idx in task_indices:
            enc_batch = self.enc_buffers[task_idx].sample(enc_size).to(self.device)
            transitions = enc_batch.as_transitions()
            enc_batches.append(enc_batch)
            batch_transitions.append(transitions)
        batch_transitions = torch.stack(batch_transitions, dim=0)
        e_dists = self.encoder.dist(batch_transitions)
        
        task_batch = TensorBatch(
            task_indices=task_indices,
            empties=torch.tensor([not enc_batch.rewards.sum().bool() for enc_batch in enc_batches]),
            e_dists=e_dists,
            enc_kls=torch_dist.kl_divergence(e_dists, self.encoder.prior_dist),
            enc_regs=torch.stack([enc_regs[task_idx] for task_idx in task_indices])
        )
        return task_batch
    
    def make_meta_batch(self, task_batch, batch_size, policy_regs):
        batches = []
        for task_idx in task_batch.task_indices:
            batch = self.buffers[task_idx].sample(batch_size).to(self.device)
            batches.append(batch)
        
        es = task_batch.e_dists.rsample((batch_size, )).swapaxes(0, 1).reshape(len(batches)*batch_size, -1)
        states = torch.cat([batch.states for batch in batches], dim=0)
        actions = torch.cat([batch.actions for batch in batches], dim=0)
        rewards = torch.cat([batch.rewards for batch in batches], dim=0)
        dones = torch.cat([batch.dones for batch in batches], dim=0)
        next_states = torch.cat([batch.next_states for batch in batches], dim=0)
        
        state_es = torch.cat([states, es], dim=-1)
        next_state_es = torch.cat([next_states, es], dim=-1)
        
        meta_batch = TensorBatch(
            states=state_es,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_states=next_state_es,
            policy_regs=torch.cat([
                policy_regs[task_idx][None].expand(batch_size)
                for task_idx in task_batch.task_indices
            ], dim=0)
        )
        return meta_batch

    def step(self, n_prior_batch, n_post_batch, batch_size, prior_enc_size, post_enc_size):
        stat = {}
        
        # sample batch from multi-task buffer
        prior_task_batch = self.make_task_batch(
            n_prior_batch, prior_enc_size,
            [self.enc_prior_reg()]*len(self.buffers)
        )
        post_task_batch = self.make_task_batch(
            n_post_batch, post_enc_size,
            [post_enc_reg() for post_enc_reg in self.enc_post_regs]
        )
        task_batch = prior_task_batch + post_task_batch
        
        prior_meta_batch = self.make_meta_batch(
            prior_task_batch, batch_size,
            [self.policy_prior_reg()]*len(self.buffers)
        )
        post_meta_batch = self.make_meta_batch(
            post_task_batch, batch_size,
            [self.policy_post_reg()]*len(self.buffers)
        )
        meta_batch = prior_meta_batch + post_meta_batch
        
        stat['prior_task_batch'] = {
            'mean_e_scale': prior_task_batch.e_dists.base_dist.scale.mean((0, 1)),
            'n_empty_batch': prior_task_batch.empties.sum()
        }
        stat['post_task_batch'] = {
            'mean_e_scale': post_task_batch.e_dists.base_dist.scale.mean((0, 1)),
            'n_empty_batch': post_task_batch.empties.sum()
        }
        
        # update qf & encoder
        with torch.no_grad():
            target_qs = self.compute_target_q(meta_batch)
        
        qf_losses = []
        for qf in self.qfs:
            qs = qf(meta_batch.states, meta_batch.actions)
            qf_loss = (qs - target_qs).pow(2).mean(0)
            qf_losses.append(qf_loss)
        qf_loss = torch.stack(qf_losses).mean(0)
        
        enc_reg_loss = (task_batch.enc_regs.detach() * task_batch.enc_kls).mean(0)
        
        qf_enc_loss = qf_loss + enc_reg_loss
        
        for qf_optim in self.qf_optims:
            qf_optim.zero_grad()
        self.encoder_optim.zero_grad()
        qf_enc_loss.backward()
        for qf_optim in self.qf_optims:
            qf_optim.step()
        self.encoder_optim.step()
        self.update_target_qfs()
        
        stat['qf_loss'] = qf_loss
        stat['enc_reg_loss'] = enc_reg_loss
        stat['qf_enc_loss'] = qf_enc_loss
        
        # update encoder regularizer
        self.enc_prior_reg_trainer.step(self.enc_prior_reg(), prior_task_batch.enc_kls)
        self.enc_post_reg_trainer.step(post_task_batch.enc_regs, post_task_batch.enc_kls)
        stat['enc_prior_kl'] = prior_task_batch.enc_kls.mean(0)
        stat['enc_post_kl'] = post_task_batch.enc_kls.mean(0)
        stat['enc_prior_reg'] = self.enc_prior_reg()
        stat['avg_enc_post_reg'] = torch.stack([enc_post_reg() for enc_post_reg in self.enc_post_regs]).mean(0)
        
        # update policy
        dists = self.policy.dist(meta_batch.states.detach())
        sampled_actions = dists.rsample()
        qs = torch.min(*[qf(meta_batch.states.detach(), sampled_actions) for qf in self.qfs])
        rl_loss = - qs.mean(0)
        
        with torch.no_grad():
            prior_dists = self.prior_policy.dist(meta_batch.states)
        kls = torch_dist.kl_divergence(dists, prior_dists)
        policy_reg_loss = (meta_batch.policy_regs.detach() * kls).mean(0)
        
        policy_loss = rl_loss + policy_reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        stat['policy_loss'] = policy_loss
        stat['rl_loss'] = rl_loss
        stat['policy_reg_loss'] = policy_reg_loss
        stat['mean_policy_scale'] = dists.base_dist.scale.mean()
        
        # update policy regularizer
        prior_kls = kls[:n_prior_batch*batch_size]
        post_kls = kls[-n_post_batch*batch_size:]
        self.policy_prior_reg_trainer.step(self.policy_prior_reg(), prior_kls)
        self.policy_post_reg_trainer.step(self.policy_post_reg(), post_kls)
        stat['policy_prior_kl'] = prior_kls.mean(0)
        stat['policy_post_kl'] = post_kls.mean(0)
        stat['policy_prior_reg'] = self.policy_prior_reg()
        stat['policy_post_reg'] = self.policy_post_reg()
        
        return stat

    def compute_target_q(self, batch):
        dists = self.policy.dist(batch.next_states)
        sampled_actions = dists.sample()
        log_probs = dists.log_prob(sampled_actions)
        min_qs = torch.min(*[target_qf(batch.next_states, sampled_actions) for target_qf in self.target_qfs])
        
        prior_dists = self.prior_policy.dist(batch.next_states)
        kls = clipped_kl(dists, prior_dists, clip=self.kl_clip)
        soft_qs = min_qs - batch.policy_regs*kls
        return batch.rewards + (1 - batch.dones)*self.gamma*soft_qs
    
    def update_target_qfs(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            for param, target_param in zip(qf.parameters(), target_qf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
