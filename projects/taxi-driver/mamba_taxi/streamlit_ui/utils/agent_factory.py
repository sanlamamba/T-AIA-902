"""
Agent Factory for creating RL agents
"""

import sys
import os

# Add the parent directory to the path to import agents
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from agents import BruteForceAgent, QLearningAgent, SARSAAgent, DQNAgent


class AgentFactory:
    """Factory class for creating different RL agents"""

    @staticmethod
    def create_agent(algorithm, n_actions, n_states, **params):
        """
        Create agent based on algorithm and parameters

        Args:
            algorithm: Name of the algorithm
            n_actions: Number of actions in the environment
            n_states: Number of states in the environment
            **params: Additional parameters for the agent

        Returns:
            Agent instance
        """
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
        """Get default parameters for an algorithm"""
        if algorithm == "BruteForce":
            return {}
        elif algorithm in ["Q-Learning", "SARSA"]:
            return {
                "alpha": 0.15,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
            }
        elif algorithm == "DQN":
            return {
                "alpha": 0.15,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "memory_size": 10000,
                "batch_size": 32,
            }
        else:
            return {}

    @staticmethod
    def validate_params(algorithm, params):
        """Validate parameters for an algorithm"""
        if algorithm == "BruteForce":
            return True

        required_params = ["alpha", "gamma", "epsilon", "epsilon_decay", "epsilon_min"]

        if algorithm == "DQN":
            required_params.extend(["memory_size", "batch_size"])

        for param in required_params:
            if param not in params:
                return False

        # Validate ranges
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
