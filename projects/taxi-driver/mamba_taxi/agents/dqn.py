import numpy as np
from collections import deque
import random
from .base import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        super().__init__(n_actions, n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)

        self.weights1 = np.random.randn(n_states, 128) * 0.01
        self.bias1 = np.zeros(128)
        self.weights2 = np.random.randn(128, 64) * 0.01
        self.bias2 = np.zeros(64)
        self.weights3 = np.random.randn(64, n_actions) * 0.01
        self.bias3 = np.zeros(n_actions)

    def _forward(self, state):
        h1 = np.maximum(0, np.dot(state, self.weights1) + self.bias1)
        h2 = np.maximum(0, np.dot(h1, self.weights2) + self.bias2)
        return np.dot(h2, self.weights3) + self.bias3

    def get_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_one_hot = np.zeros(self.n_states)
        state_one_hot[state] = 1
        q_values = self._forward(state_one_hot)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_one_hot = np.zeros(self.n_states)
            state_one_hot[state] = 1
            next_state_one_hot = np.zeros(self.n_states)
            next_state_one_hot[next_state] = 1

            current_q = self._forward(state_one_hot)

            if done:
                target = reward
            else:
                next_q = self._forward(next_state_one_hot)
                target = reward + self.gamma * np.max(next_q)

            error = target - current_q[action]
            self.training_errors.append(abs(error))
            current_q[action] += self.alpha * error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
