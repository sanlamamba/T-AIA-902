import numpy as np
from .base import BaseAgent


class BruteForceAgent(BaseAgent):
    def get_action(self, state, explore=True):
        return np.random.randint(self.n_actions)
