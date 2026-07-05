import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from advanced_network import DuelingDQNNetwork

class AdvancedDQNAgent:
    def __init__(
        self,
        obs_dim=10,
        n_actions=6,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9993484,
        device=None,
        loss_fn="huber",
        grad_clip_norm=1.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.n_actions = n_actions

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.grad_clip_norm = grad_clip_norm

        self.q_network = DuelingDQNNetwork(obs_dim, n_actions).to(self.device)
        self.target_network = DuelingDQNNetwork(obs_dim, n_actions).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss() if loss_fn == "huber" else nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax(dim=1).item())

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_actions = self.q_network(next_states).argmax(dim=1).unsqueeze(1)
            next_q = self.target_network(next_states).gather(1, next_state_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)