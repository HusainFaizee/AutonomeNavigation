import numpy as np
import random


class SARSAAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.action_space = env.action_space
        self.q_table = {}

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        q_values = self.get_q(state)
        return self.action_space[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, next_action):
        q_values = self.get_q(state)
        next_q_values = self.get_q(next_state)

        action_idx = self.action_space.index(action)
        next_action_idx = self.action_space.index(next_action)

        td_target = reward + self.gamma * next_q_values[next_action_idx]
        td_error = td_target - q_values[action_idx]

        q_values[action_idx] += self.alpha * td_error