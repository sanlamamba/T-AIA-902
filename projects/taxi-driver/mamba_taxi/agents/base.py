from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, n_actions, n_states=None):
        self.n_actions = n_actions
        self.n_states = n_states
        self.training_errors = []

    @abstractmethod
    def get_action(self, state, explore=True):
        pass

    def update(self, *args, **kwargs):
        pass
