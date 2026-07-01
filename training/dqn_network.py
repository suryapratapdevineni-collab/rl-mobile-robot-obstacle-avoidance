import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, obs_dim=10, n_actions=6):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(obs_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)