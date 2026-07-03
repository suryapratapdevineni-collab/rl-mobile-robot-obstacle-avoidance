import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_network import DQNNetwork


class DQNAgent:
    
    def __init__(
        self,
        obs_dim=10,
        n_actions=6,
        lr=3e-4,               # UPDATED: Slower learning rate (from 5e-4) for stable learning across 1000 episodes
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9975,  # UPDATED: Matches the 1600-episode horizon schedule
        device=None,
    ):
        

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.gamma = gamma
        self.n_actions = n_actions

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = DQNNetwork(
            obs_dim,
            n_actions
        ).to(self.device)

        self.target_network = DQNNetwork(
            obs_dim,
            n_actions
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=lr
        )

        self.loss_fn = nn.SmoothL1Loss()

    ##################################################

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.q_network(state)

        return int(torch.argmax(q).item())

    ##################################################

    def train_step(self, batch):

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.target_network(next_states)

            max_next_q = next_q.max(1)[0]

            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(
            current_q,
            target
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            5.0
        )

        self.optimizer.step()

        return loss.item()

    ##################################################

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    ##################################################

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )