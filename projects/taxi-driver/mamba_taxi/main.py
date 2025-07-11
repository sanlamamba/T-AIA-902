import argparse
import gymnasium as gym
import numpy as np
import time
import json
import os
from typing import List, Dict, Any

from agents import BruteForceAgent, QLearningAgent, SARSAAgent, DQNAgent
from trainer import train_agent, evaluate_agent
from visualizer import plot_results, plot_training_analysis
from advanced_config import (
    AdvancedConfig,
    ConfigurationManager,
    ExplorationStrategy,
    RewardShaping,
)
from enhanced_statistics import AdvancedStatisticalAnalyzer


class TaxiDriver:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode=None)
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n

        self.config_manager = ConfigurationManager()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()

        self.optimized_params = {
            "q-learning": {
                "alpha": 0.15,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
            },
            "sarsa": {
                "alpha": 0.15,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
            },
            "dqn": {
                "alpha": 0.001,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "memory_size": 10000,
                "batch_size": 32,
            },
        }

    def user_mode(self, algorithm, train_episodes, test_episodes, **params):
        print(f"\n{'='*60}")
        print(f"USER MODE: {algorithm.upper()}")
        print(f"{'='*60}")
        print(f"Training Episodes: {train_episodes}")
        print(f"Test Episodes: {test_episodes}")
        print(f"Parameters: {params}")

        agent = self._create_agent(algorithm, **params)
        algo_name = algorithm.replace("-", " ").title()

        if algorithm != "bruteforce":
            print(f"\n🚀 Starting training...")
            train_result = train_agent(self.env, agent, train_episodes, algo_name)
            print(f"✅ Training completed in {train_result['training_time']:.2f}s")

        print(f"\n🔍 Starting evaluation...")
        test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

        self._print_results(test_result)
        return test_result

    def time_limited_mode(self, algorithm, time_limit, test_episodes):
        print(f"\n{'='*60}")
        print(f"TIME-LIMITED MODE: {algorithm.upper()} ({time_limit}s)")
        print(f"{'='*60}")

        params = self.optimized_params.get(algorithm, {})
        agent = self._create_agent(algorithm, **params)
        algo_name = algorithm.replace("-", " ").title()

        if algorithm == "bruteforce":
            print("⚠️  BruteForce doesn't require training")
            episodes_trained = 0
            elapsed = 0.0
        else:
            episodes_trained, elapsed = self._time_limited_training(
                agent, algorithm, time_limit
            )
            print(
                f"✅ Trained for {episodes_trained} episodes in {elapsed:.2f}s (limit was {time_limit}s)"
            )

        print(f"\n🔍 Starting evaluation...")
        test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

        self._print_results(test_result)
        return test_result

    def _time_limited_training(self, agent, algorithm, time_limit):
        start_time = time.time()
        episode = 0
        rewards = []

        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                break

            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                if time.time() - start_time >= time_limit:
                    done = True
                    break

                if algorithm == "sarsa":
                    action = agent.get_action(state)
                else:
                    action = agent.get_action(state, explore=True)

                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                if algorithm == "q-learning":
                    agent.update(state, action, reward, next_state, done)
                elif algorithm == "sarsa":
                    next_action = agent.get_action(next_state, explore=True)
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action
                elif algorithm == "dqn":
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()

                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            episode += 1

            if episode % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

        elapsed = time.time() - start_time
        return episode, elapsed

    def benchmark(self, train_episodes, test_episodes):
        print(f"\n{'='*60}")
        print("BENCHMARK MODE")
        print(f"{'='*60}")
        print(f"Training Episodes: {train_episodes}")
        print(f"Test Episodes: {test_episodes}")

        results = []
        training_data = {}

        print(f"\n🎲 Testing BruteForce baseline...")
        bf_agent = BruteForceAgent(self.n_actions, self.n_states)
        bf_result = evaluate_agent(self.env, bf_agent, test_episodes, "BruteForce")
        bf_result["training_time"] = 0
        bf_result["efficiency_score"] = 0
        results.append(bf_result)

        algorithms = [
            ("q-learning", QLearningAgent, "Q-Learning"),
            ("sarsa", SARSAAgent, "SARSA"),
            ("dqn", DQNAgent, "DQN"),
        ]

        for algo_key, agent_class, algo_name in algorithms:
            params = self.optimized_params.get(algo_key, {})
            agent = agent_class(self.n_actions, self.n_states, **params)

            train_result = train_agent(self.env, agent, train_episodes, algo_name)
            test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

            test_result["training_time"] = train_result["training_time"]
            test_result["convergence_episode"] = train_result["convergence_episode"]
            test_result["final_performance"] = train_result["final_performance"]

            results.append(test_result)
            training_data[algo_name] = train_result

        plot_results(results, training_data)
        plot_training_analysis(training_data)

        self._print_benchmark_summary(results)
        return results

    def advanced_benchmark(self, train_episodes, test_episodes, config_variants=None):
        """Advanced benchmarking with comprehensive analysis"""
        print(f"\n{'='*80}")
        print("ADVANCED BENCHMARK MODE")
        print(f"{'='*80}")
        print(f"Training Episodes: {train_episodes}")
        print(f"Test Episodes: {test_episodes}")

        results = []
        training_data = []
        configurations = []

        bf_agent = BruteForceAgent(self.n_actions, self.n_states)
        bf_result = evaluate_agent(self.env, bf_agent, test_episodes, "BruteForce")
        bf_result["training_time"] = 0
        bf_result["efficiency_score"] = 0
        results.append(bf_result)
        training_data.append({"rewards": [], "steps": [], "algorithm": "BruteForce"})
        configurations.append({"algorithm": "BruteForce", "config": {}})

        if config_variants:
            for i, (algo_key, config) in enumerate(config_variants):

                agent_class = self._get_agent_class(algo_key)
                agent = agent_class(self.n_actions, self.n_states, **config)

                algo_name = f"{algo_key.replace('-', ' ').title()}_v{i+1}"

                train_result = train_agent(self.env, agent, train_episodes, algo_name)
                test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

                # Combine results
                test_result["training_time"] = train_result["training_time"]
                test_result["convergence_episode"] = train_result["convergence_episode"]
                test_result["final_performance"] = train_result["final_performance"]

                results.append(test_result)
                training_data.append(train_result)
                configurations.append({"algorithm": algo_key, "config": config})
        else:
            algorithms = [
                ("q-learning", QLearningAgent, "Q-Learning"),
                ("sarsa", SARSAAgent, "SARSA"),
                ("dqn", DQNAgent, "DQN"),
            ]

            for algo_key, agent_class, algo_name in algorithms:
                print(f"\n🤖 Training {algo_name}...")
                params = self.optimized_params.get(algo_key, {})
                agent = agent_class(self.n_actions, self.n_states, **params)

                train_result = train_agent(self.env, agent, train_episodes, algo_name)
                test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

                test_result["training_time"] = train_result["training_time"]
                test_result["convergence_episode"] = train_result["convergence_episode"]
                test_result["final_performance"] = train_result["final_performance"]

                results.append(test_result)
                training_data.append(train_result)
                configurations.append({"algorithm": algo_key, "config": params})

        statistical_analysis = self.stat_analyzer.comprehensive_analysis(
            results, training_data
        )

        plot_results(
            results, {td["algorithm"]: td for td in training_data if "algorithm" in td}
        )
        plot_training_analysis(
            {td["algorithm"]: td for td in training_data if "algorithm" in td}
        )

        self._save_detailed_results(
            results, training_data, configurations, statistical_analysis
        )

        self._print_comprehensive_summary(results, statistical_analysis)

        return {
            "results": results,
            "training_data": training_data,
            "configurations": configurations,
            "statistical_analysis": statistical_analysis,
        }

    def hyperparameter_optimization(
        self,
        algorithm="q-learning",
        optimization_method="grid",
        train_episodes=500,
        test_episodes=50,
        n_configs=20,
    ):
        """Perform hyperparameter optimization"""
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER OPTIMIZATION: {algorithm.upper()}")
        print(f"{'='*80}")
        print(f"Optimization Method: {optimization_method}")
        print(f"Training Episodes: {train_episodes}")
        print(f"Test Episodes: {test_episodes}")
        print(f"Number of Configurations: {n_configs}")

        base_config = AdvancedConfig()

        if optimization_method == "grid":
            configs = self.config_manager.create_parameter_grid(base_config)[:n_configs]
        elif optimization_method == "bayesian":
            configs = self.config_manager.optimize_bayesian(base_config, n_configs)
        elif optimization_method == "multi_objective":
            configs = self.config_manager.create_multi_objective_configs(base_config)
        else:
            configs = [base_config]

        config_variants = []

        for i, config in enumerate(configs):
            config_dict = self._config_to_dict(config)
            config_variants.append((algorithm, config_dict))

            self.config_manager.save_config(config, f"{algorithm}_config_{i}", {})

        benchmark_results = self.advanced_benchmark(
            train_episodes, test_episodes, config_variants
        )

        best_idx = np.argmax(
            [r.get("efficiency_score", 0) for r in benchmark_results["results"][1:]]
        )
        best_config = configs[best_idx]
        best_result = benchmark_results["results"][best_idx + 1]

        print(f"\n🏆 BEST CONFIGURATION FOUND:")
        print(f"Configuration Index: {best_idx}")
        print(f"Efficiency Score: {best_result.get('efficiency_score', 0):.4f}")
        print(f"Win Rate: {best_result.get('win_rate', 0):.2%}")
        print(f"Mean Reward: {best_result.get('mean_reward', 0):.2f}")

        self.config_manager.save_config(best_config, f"{algorithm}_best", best_result)

        return {
            "best_config": best_config,
            "best_result": best_result,
            "all_results": benchmark_results,
            "optimization_summary": self._create_optimization_summary(
                configs, benchmark_results["results"][1:]
            ),
        }

    def comparative_analysis(
        self,
        algorithms=None,
        train_episodes=1000,
        test_episodes=100,
        n_runs=5,
        statistical_significance=True,
    ):
        """Perform comprehensive comparative analysis"""
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS MODE")
        print(f"{'='*80}")
        print(f"Training Episodes: {train_episodes}")
        print(f"Test Episodes: {test_episodes}")
        print(f"Number of Runs: {n_runs}")

        if algorithms is None:
            algorithms = ["q-learning", "sarsa", "dqn"]

        all_results = []
        all_training_data = []

        for run in range(n_runs):
            run_results = []
            run_training_data = []

            bf_agent = BruteForceAgent(self.n_actions, self.n_states)
            bf_result = evaluate_agent(
                self.env, bf_agent, test_episodes, f"BruteForce_run{run}"
            )
            bf_result["training_time"] = 0
            bf_result["efficiency_score"] = 0
            bf_result["run"] = run
            run_results.append(bf_result)
            run_training_data.append(
                {"rewards": [], "steps": [], "algorithm": "BruteForce"}
            )

            for algorithm in algorithms:
                print(f"  🤖 Training {algorithm}...")

                params = self.optimized_params.get(algorithm, {})
                agent_class = self._get_agent_class(algorithm)
                agent = agent_class(self.n_actions, self.n_states, **params)

                algo_name = f"{algorithm.replace('-', ' ').title()}_run{run}"

                train_result = train_agent(self.env, agent, train_episodes, algo_name)
                test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

                test_result["training_time"] = train_result["training_time"]
                test_result["run"] = run
                test_result["algorithm_base"] = algorithm

                run_results.append(test_result)
                run_training_data.append(train_result)

            all_results.extend(run_results)
            all_training_data.extend(run_training_data)

        statistical_analysis = self.stat_analyzer.comprehensive_analysis(
            all_results, all_training_data
        )

        comparative_stats = self._perform_comparative_statistics(
            all_results, algorithms, n_runs
        )

        report = self._generate_comparative_report(
            all_results, statistical_analysis, comparative_stats
        )

        self._save_comparative_results(
            all_results, statistical_analysis, comparative_stats
        )

        return {
            "all_results": all_results,
            "statistical_analysis": statistical_analysis,
            "comparative_stats": comparative_stats,
            "report": report,
        }

    def _create_agent(self, algorithm, **params):
        if algorithm == "bruteforce":
            return BruteForceAgent(self.n_actions, self.n_states)
        elif algorithm == "q-learning":
            return QLearningAgent(self.n_actions, self.n_states, **params)
        elif algorithm == "sarsa":
            return SARSAAgent(self.n_actions, self.n_states, **params)
        elif algorithm == "dqn":
            return DQNAgent(self.n_actions, self.n_states, **params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _get_agent_class(self, algorithm):
        """Get agent class by algorithm name"""
        if algorithm == "bruteforce":
            return BruteForceAgent
        elif algorithm == "q-learning":
            return QLearningAgent
        elif algorithm == "sarsa":
            return SARSAAgent
        elif algorithm == "dqn":
            return DQNAgent
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _config_to_dict(self, config: AdvancedConfig) -> dict:
        """Convert AdvancedConfig to dictionary for agent creation"""
        return {
            "alpha": config.alpha,
            "gamma": config.gamma,
            "epsilon": config.epsilon,
            "epsilon_decay": config.epsilon_decay,
            "epsilon_min": config.epsilon_min,
            "memory_size": getattr(config, "replay_buffer_size", 10000),
            "batch_size": getattr(config, "batch_size", 32),
        }

    def _save_detailed_results(
        self, results, training_data, configurations, statistical_analysis
    ):
        """Save detailed results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "performance_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        with open(os.path.join(results_dir, "training_data.json"), "w") as f:
            json.dump(training_data, f, indent=2, default=str)

        with open(os.path.join(results_dir, "configurations.json"), "w") as f:
            json.dump(configurations, f, indent=2, default=str)

        with open(os.path.join(results_dir, "statistical_analysis.json"), "w") as f:
            json.dump(statistical_analysis, f, indent=2, default=str)

        report = self.stat_analyzer.generate_comprehensive_report(statistical_analysis)
        with open(os.path.join(results_dir, "comprehensive_report.txt"), "w") as f:
            f.write(report)

    def _print_comprehensive_summary(self, results, statistical_analysis):
        """Print comprehensive summary of results"""
        print(f"\n🏆 COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*80}")

        rl_results = [r for r in results if r["algorithm"] != "BruteForce"]

        if rl_results:
            ranked_by_efficiency = sorted(
                rl_results, key=lambda x: x.get("efficiency_score", 0), reverse=True
            )
            for i, result in enumerate(ranked_by_efficiency, 1):
                print(
                    f"  {i}. {result['algorithm']}: {result.get('efficiency_score', 0):.4f}"
                )

            ranked_by_winrate = sorted(
                rl_results, key=lambda x: x.get("win_rate", 0), reverse=True
            )
            for i, result in enumerate(ranked_by_winrate, 1):
                print(f"  {i}. {result['algorithm']}: {result.get('win_rate', 0):.2%}")

        if "descriptive_stats" in statistical_analysis:
            desc_stats = statistical_analysis["descriptive_stats"]
            for metric, stats in desc_stats.items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"{metric.replace('_', ' ').title()}:")
                    print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
                    print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
                    print(f"  CV: {stats['cv']:.3f}")

    def _print_results(self, result):
        print(f"\n📈 RESULTS FOR {result['algorithm'].upper()}")
        print(f"{'─'*50}")
        print(
            f"Mean Reward: {result['mean_reward']:.2f} ± {result.get('std_reward', 0):.2f}"
        )
        print(
            f"Mean Steps: {result['mean_steps']:.1f} ± {result.get('std_steps', 0):.1f}"
        )
        print(f"Win Rate: {result['win_rate']:.2%}")
        print(f"Efficiency Score: {result.get('efficiency_score', 0):.4f}")
        if result.get("mean_success_steps", float("inf")) != float("inf"):
            print(f"Mean Steps (Success Only): {result['mean_success_steps']:.1f}")

    def _print_benchmark_summary(self, results):
        print(f"\n🏆 BENCHMARK SUMMARY")
        print(f"{'='*80}")

        best_reward = max(results, key=lambda x: x["mean_reward"])
        best_efficiency = max(results, key=lambda x: x.get("efficiency_score", 0))
        best_winrate = max(results, key=lambda x: x["win_rate"])

        print(
            f"🥇 Best Average Reward: {best_reward['algorithm']} ({best_reward['mean_reward']:.2f})"
        )
        print(
            f"🥇 Best Efficiency: {best_efficiency['algorithm']} ({best_efficiency.get('efficiency_score', 0):.4f})"
        )
        print(
            f"🥇 Best Win Rate: {best_winrate['algorithm']} ({best_winrate['win_rate']:.2%})"
        )

        bf_result = next(r for r in results if r["algorithm"] == "BruteForce")
        print(f"\n📊 IMPROVEMENT OVER BRUTE FORCE:")
        print(f"{'─'*50}")
        for result in results:
            if result["algorithm"] != "BruteForce":
                reward_improvement = (
                    (result["mean_reward"] - bf_result["mean_reward"])
                    / abs(bf_result["mean_reward"])
                ) * 100
                step_improvement = (
                    (bf_result["mean_steps"] - result["mean_steps"])
                    / bf_result["mean_steps"]
                ) * 100
                print(
                    f"{result['algorithm']:12} | Reward: {reward_improvement:+6.1f}% | Steps: {step_improvement:+6.1f}%"
                )

    def _create_optimization_summary(self, configs, results):
        """Create optimization summary"""
        return {
            "total_configs_tested": len(configs),
            "best_performance": max([r.get("efficiency_score", 0) for r in results]),
            "performance_range": {
                "min": min([r.get("efficiency_score", 0) for r in results]),
                "max": max([r.get("efficiency_score", 0) for r in results]),
                "mean": np.mean([r.get("efficiency_score", 0) for r in results]),
                "std": np.std([r.get("efficiency_score", 0) for r in results]),
            },
        }

    def _perform_comparative_statistics(self, all_results, algorithms, n_runs):
        """Perform comparative statistics across multiple runs"""
        comparative_stats = {}

        for algorithm in algorithms + ["BruteForce"]:
            algo_results = [
                r
                for r in all_results
                if r.get("algorithm_base") == algorithm or r["algorithm"] == algorithm
            ]

            if algo_results:
                metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]
                algo_stats = {}

                for metric in metrics:
                    values = [r.get(metric, 0) for r in algo_results]
                    if values:
                        algo_stats[metric] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "runs": len(values),
                        }

                comparative_stats[algorithm] = algo_stats

        return comparative_stats

    def _generate_comparative_report(
        self, all_results, statistical_analysis, comparative_stats
    ):
        """Generate comprehensive comparative report"""
        report = []
        report.append("COMPREHENSIVE COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Total Experiments: {len(all_results)}")
        report.append("")

        # Add comparative statistics
        report.append("ALGORITHM COMPARISON")
        report.append("-" * 30)

        for algorithm, stats in comparative_stats.items():
            report.append(f"\n{algorithm.upper()}:")
            for metric, metric_stats in stats.items():
                report.append(
                    f"  {metric}: {metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}"
                )

        return "\n".join(report)

    def _save_comparative_results(
        self, all_results, statistical_analysis, comparative_stats
    ):
        """Save comparative analysis results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = f"comparative_analysis_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        # Save all results
        with open(os.path.join(results_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        with open(os.path.join(results_dir, "statistical_analysis.json"), "w") as f:
            json.dump(statistical_analysis, f, indent=2, default=str)

        with open(os.path.join(results_dir, "comparative_stats.json"), "w") as f:
            json.dump(comparative_stats, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Taxi Driver RL - Advanced Analysis Version"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "user",
            "time-limited",
            "benchmark",
            "advanced-benchmark",
            "hyperopt",
            "comparative",
        ],
        default="advanced-benchmark",
        help="Mode to run",
    )
    parser.add_argument(
        "--algorithm",
        choices=["bruteforce", "q-learning", "sarsa", "dqn"],
        default="q-learning",
        help="Algorithm to use",
    )
    parser.add_argument(
        "--train-episodes", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--test-episodes", type=int, default=100, help="Number of test episodes"
    )
    parser.add_argument(
        "--time-limit", type=int, default=60, help="Time limit in seconds"
    )
    parser.add_argument("--alpha", type=float, default=0.23, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.987, help="Discount factor")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.8,
        help="Initial exploration rate (optimized default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--optimization-method",
        choices=["grid", "bayesian", "multi_objective"],
        default="grid",
        help="Hyperparameter optimization method",
    )
    parser.add_argument(
        "--n-configs", type=int, default=20, help="Number of configurations to test"
    )
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of runs for comparative analysis"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"🚕 TAXI DRIVER RL - Advanced Analysis Version")
    print(f"Random seed: {args.seed}")

    driver = TaxiDriver()

    if args.mode == "user":
        params = {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        }
        driver.user_mode(
            args.algorithm, args.train_episodes, args.test_episodes, **params
        )
    elif args.mode == "time-limited":
        driver.time_limited_mode(args.algorithm, args.time_limit, args.test_episodes)
    elif args.mode == "benchmark":
        driver.benchmark(args.train_episodes, args.test_episodes)
    elif args.mode == "advanced-benchmark":
        driver.advanced_benchmark(args.train_episodes, args.test_episodes)
    elif args.mode == "hyperopt":
        driver.hyperparameter_optimization(
            args.algorithm,
            args.optimization_method,
            args.train_episodes,
            args.test_episodes,
            args.n_configs,
        )
    elif args.mode == "comparative":
        driver.comparative_analysis(
            None, args.train_episodes, args.test_episodes, args.n_runs
        )


if __name__ == "__main__":
    main()
