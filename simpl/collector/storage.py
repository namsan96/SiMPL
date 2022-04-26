from collections import deque, OrderedDict

import numpy as np
import torch


class Episode:
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def __repr__(self):
        return f'Episode(cum_reward={sum(self.rewards)}, length={len(self)})'

    def __len__(self):
        return len(self.actions)
    
    def add_step(self, action, next_state, reward, done, info):
        self.actions.append(action)
        self.states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def as_batch(self):
        all_states = np.array(self.states)
        states = torch.tensor(all_states[:-1], dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32).unsqueeze(-1)
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(all_states[1:], dtype=torch.float32)
        return Batch(states, actions, rewards, dones, next_states)


class Batch:
    def __init__(self, states, actions, rewards, dones, next_states, transitions=None):
        self.data = OrderedDict([
            ('states', states),
            ('actions', actions),
            ('rewards', rewards),
            ('dones', dones),
            ('next_states', next_states)
        ])
        self.transitions = transitions

    def __repr__(self):
        return f'Batch(size={len(self.transitions)})'

    def to(self, device):
        self.device = device
        self.data = {
            k: v.to(device)
            for k, v in self.data.items()
        }
        if self.transitions is not None:
            self.transitions = self.transitions.to(device)
        return self

    def as_transitions(self):
        if self.transitions is None:
            self.transitions = torch.cat(list(self.data.values()), dim=-1)
        return self.transitions
        
    @property
    def states(self):
        return self.data['states']
    
    @property
    def actions(self):
        return self.data['actions']
    
    @property
    def rewards(self):
        return self.data['rewards'].squeeze(-1)

    @property
    def dones(self):
        return self.data['dones'].squeeze(-1)

    @property
    def next_states(self):
        return self.data['next_states']


class Buffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        
        self.ptr = 0
        self.size = 0
        
        self.episodes = deque()
        self.episode_ptrs = deque()
        self.transitions = torch.empty(max_size, 2*state_dim + action_dim + 2)
        
        dims = OrderedDict([
            ('state', state_dim),
            ('action', action_dim),
            ('reward', 1),
            ('done', 1),
            ('next_state', state_dim),
        ])
        self.layout = dict()
        prev_i = 0
        for k, v in dims.items():
            next_i = prev_i + v
            self.layout[k] = slice(prev_i, next_i)
            prev_i = next_i
        
        self.device = None
  
    def __repr__(self):
        return f'Buffer(max_size={self.max_size}, self.size={self.size})'

    def to(self, device):
        self.device = device
        return self

    @property
    def states(self):
        return self.transitions[:, self.layout['state']]
    
    @property
    def actions(self):
        return self.transitions[:, self.layout['action']]
    
    @property
    def rewards(self):
        return self.transitions[:, self.layout['reward']]

    @property
    def dones(self):
        return self.transitions[:, self.layout['done']]

    @property
    def next_states(self):
        return self.transitions[:, self.layout['next_state']]

    def enqueue(self, episode):
        while len(self.episodes) > 0:
            old_episode = self.episodes[0]
            ptr = self.episode_ptrs[0]
            dist = (ptr - self.ptr) % self.max_size
            if dist < len(episode):
                self.episodes.popleft()
                self.episode_ptrs.popleft()
            else:
                break
        self.episodes.append(episode)
        self.episode_ptrs.append(self.ptr)
        
        transitions = torch.as_tensor(np.concatenate([
            episode.states[:-1], episode.actions,
            np.array(episode.rewards)[:, None], np.array(episode.dones)[:, None],
            episode.states[1:]
        ], axis=-1))
        
        if self.ptr + len(episode) <= self.max_size:
            self.transitions[self.ptr:self.ptr+len(episode)] = transitions
        elif self.ptr + len(episode) < 2*self.max_size:
            self.transitions[self.ptr:] = transitions[:self.max_size-self.ptr]
            self.transitions[:len(episode)-self.max_size+self.ptr] = transitions[self.max_size-self.ptr:]
        else:
            raise NotImplementedError
        self.ptr = (self.ptr + len(episode)) % self.max_size
        self.size = min(self.size + len(episode), self.max_size)

    def sample(self, n):
        indices = torch.randint(self.size, size=[n], device=self.device)
        transitions = self.transitions[indices]
        return Batch(*[transitions[:, i] for i in self.layout.values()], transitions)
