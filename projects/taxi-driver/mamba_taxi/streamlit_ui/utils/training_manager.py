import sys
import os
import time
import streamlit as st
from typing import Dict, List, Any

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from trainer import train_agent, evaluate_agent
from .agent_factory import AgentFactory


class TrainingManager:

    def __init__(self, env):
        self.env = env
        self.agent_factory = AgentFactory()

    def train_single_agent(
        self, algorithm: str, params: Dict, train_episodes: int, test_episodes: int
    ) -> Dict:
        # Create agent
        env_info = st.session_state
        agent = self.agent_factory.create_agent(
            algorithm, env_info.n_actions, env_info.n_states, **params
        )

        # Training
        start_time = time.time()
        if algorithm != "BruteForce":
            train_result = train_agent(self.env, agent, train_episodes, algorithm)
        else:
            train_result = {"rewards": [], "steps": [], "training_time": 0}

        training_time = time.time() - start_time

        # Evaluation
        test_result = evaluate_agent(self.env, agent, test_episodes, algorithm)
        test_result["training_time"] = training_time

        return {
            "train_result": train_result,
            "test_result": test_result,
            "agent": agent,
        }

    def compare_algorithms(
        self,
        algorithms: List[str],
        params_dict: Dict,
        train_episodes: int,
        test_episodes: int,
    ) -> Dict:
        """
        Compare multiple algorithms

        Args:
            algorithms: List of algorithm names
            params_dict: Dictionary of parameters for each algorithm
            train_episodes: Number of training episodes
            test_episodes: Number of test episodes

        Returns:
            Dictionary containing comparison results
        """
        results = []
        training_data = {}

        for algorithm in algorithms:
            params = params_dict.get(algorithm, {})

            # Get default params if not provided
            if not params and algorithm != "BruteForce":
                params = self.agent_factory.get_default_params(algorithm)

            # Train and evaluate
            result = self.train_single_agent(
                algorithm, params, train_episodes, test_episodes
            )

            results.append(result["test_result"])
            if algorithm != "BruteForce":
                training_data[algorithm] = result["train_result"]

        return {"results": results, "training_data": training_data}

    def optimize_hyperparameters(
        self,
        algorithm: str,
        param_ranges: Dict,
        n_configs: int,
        train_episodes: int,
        test_episodes: int,
        method: str = "random",
    ) -> Dict:
        """
        Optimize hyperparameters for an algorithm

        Args:
            algorithm: Algorithm to optimize
            param_ranges: Dictionary of parameter ranges
            n_configs: Number of configurations to test
            train_episodes: Training episodes per configuration
            test_episodes: Test episodes per configuration
            method: Optimization method ('random', 'grid', 'bayesian')

        Returns:
            Dictionary containing optimization results
        """
        import numpy as np

        configs = self._generate_configs(param_ranges, n_configs, method)
        results = []
        best_score = -float("inf")
        best_config = None

        for i, config in enumerate(configs):
            # Validate parameters
            if not self.agent_factory.validate_params(algorithm, config):
                continue

            try:
                result = self.train_single_agent(
                    algorithm, config, train_episodes, test_episodes
                )
                test_result = result["test_result"]
                test_result.update(config)
                test_result["config_id"] = i
                results.append(test_result)

                # Track best configuration
                current_score = test_result.get("efficiency_score", 0)
                if current_score > best_score:
                    best_score = current_score
                    best_config = config.copy()
                    best_config["score"] = current_score

            except Exception as e:
                st.warning(f"Configuration {i} échouée : {e}")
                continue

        return {
            "results": results,
            "best_config": best_config,
            "best_score": best_score,
        }

    def _generate_configs(
        self, param_ranges: Dict, n_configs: int, method: str
    ) -> List[Dict]:
        """Generate parameter configurations"""
        import numpy as np

        configs = []

        if method == "random":
            for _ in range(n_configs):
                config = {}
                for param, range_info in param_ranges.items():
                    if isinstance(range_info, tuple) and len(range_info) == 2:
                        config[param] = np.random.uniform(range_info[0], range_info[1])
                    elif isinstance(range_info, dict):
                        if "min" in range_info and "max" in range_info:
                            config[param] = np.random.uniform(
                                range_info["min"], range_info["max"]
                            )

                # Add fixed parameters
                config.update(
                    {
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    }
                )

                configs.append(config)

        elif method == "grid":
            # Simplified grid search
            alpha_values = np.linspace(
                param_ranges["alpha"][0],
                param_ranges["alpha"][1],
                int(np.sqrt(n_configs)),
            )
            gamma_values = np.linspace(
                param_ranges["gamma"][0],
                param_ranges["gamma"][1],
                int(np.sqrt(n_configs)),
            )

            for alpha in alpha_values:
                for gamma in gamma_values:
                    config = {
                        "alpha": alpha,
                        "gamma": gamma,
                        "epsilon": 1.0,
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.01,
                    }
                    configs.append(config)
                    if len(configs) >= n_configs:
                        break
                if len(configs) >= n_configs:
                    break

        return configs[:n_configs]

    def evaluate_with_progress(
        self,
        algorithm: str,
        params: Dict,
        train_episodes: int,
        test_episodes: int,
        progress_callback=None,
    ):
        """Evaluate with progress tracking"""
        result = self.train_single_agent(
            algorithm, params, train_episodes, test_episodes
        )

        if progress_callback:
            progress_callback(1.0, "Evaluation completed!")

        return result
