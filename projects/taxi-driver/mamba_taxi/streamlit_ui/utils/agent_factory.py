import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from agents import BruteForceAgent, QLearningAgent, SARSAAgent, DQNAgent


class AgentFactory:

    @staticmethod
    def create_agent(algorithm, n_actions, n_states, **params):
        if algorithm == "BruteForce":
            return BruteForceAgent(n_actions, n_states)
        elif algorithm == "Q-Learning":
            return QLearningAgent(n_actions, n_states, **params)
        elif algorithm == "SARSA":
            return SARSAAgent(n_actions, n_states, **params)
        elif algorithm == "DQN":
            return DQNAgent(n_actions, n_states, **params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    @staticmethod
    def get_default_params(algorithm):
        if algorithm == "BruteForce":
            return {}
        elif algorithm == "Q-Learning":
            return {
                "alpha": 0.23,
                "gamma": 0.987,
                "epsilon": 0.8,
                "epsilon_decay": 0.9945,
                "epsilon_min": 0.01,
            }
        elif algorithm == "SARSA":
            return {
                "alpha": 0.19,
                "gamma": 0.991,
                "epsilon": 0.7,
                "epsilon_decay": 0.9952,
                "epsilon_min": 0.01,
            }
        elif algorithm == "DQN":
            return {
                "alpha": 0.15,
                "gamma": 0.995,
                "epsilon": 0.8,
                "epsilon_decay": 0.9945,
                "epsilon_min": 0.01,
                "memory_size": 15000,
                "batch_size": 64,
            }
        else:
            return {}

    @staticmethod
    def validate_params(algorithm, params):
        if algorithm == "BruteForce":
            return True

        required_params = ["alpha", "gamma", "epsilon", "epsilon_decay", "epsilon_min"]

        if algorithm == "DQN":
            required_params.extend(["memory_size", "batch_size"])

        for param in required_params:
            if param not in params:
                return False

        if not (0 < params["alpha"] <= 1):
            return False
        if not (0 < params["gamma"] < 1):
            return False
        if not (0 <= params["epsilon"] <= 1):
            return False
        if not (0 < params["epsilon_decay"] < 1):
            return False
        if not (0 < params["epsilon_min"] < 1):
            return False

        if algorithm == "DQN":
            if not (params["memory_size"] > 0):
                return False
            if not (params["batch_size"] > 0):
                return False

        return True
