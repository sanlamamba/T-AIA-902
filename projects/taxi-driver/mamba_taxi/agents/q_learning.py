import numpy as np
from .base import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        super().__init__(n_actions, n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((n_states, n_actions))
        self.visit_count = np.zeros((n_states, n_actions))

    def get_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        self.visit_count[state, action] += 1
        best_next_q = np.max(self.q_table[next_state])
        target = reward + (1 - done) * self.gamma * best_next_q
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * error
        self.training_errors.append(abs(error))

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state_coverage(self):
        return np.sum(self.visit_count > 0) / (self.n_states * self.n_actions)
