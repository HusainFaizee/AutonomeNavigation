import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon

        self.action_space = env.action_space
        self.state_size = 2  # assuming (x, y)
        self.action_size = len(self.action_space)

        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, state):
        return torch.FloatTensor(state)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        state_tensor = self.state_to_tensor(state)
        q_values = self.model(state_tensor)

        return self.action_space[torch.argmax(q_values).item()]

    def update(self, state, action, reward, next_state):
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        action_idx = self.action_space.index(action)

        target = q_values.clone().detach()
        target[action_idx] = reward + self.gamma * torch.max(next_q_values).item()

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()