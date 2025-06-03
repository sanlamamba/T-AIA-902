import numpy as np
from .base import BaseAgent


class SARSAAgent(BaseAgent):
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(n_actions, n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        target = (
            reward + (1 - done) * self.gamma * self.q_table[next_state, next_action]
        )
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * error
        self.training_errors.append(abs(error))
