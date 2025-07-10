import argparse
import gymnasium as gym
import numpy as np
import time

from agents import BruteForceAgent, QLearningAgent, SARSAAgent, DQNAgent
from trainer import train_agent, evaluate_agent
from visualizer import plot_results, plot_training_analysis


class TaxiDriver:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode=None)
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n

        # Optimized parameters based on experiments
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
        else:
            episodes_trained = self._time_limited_training(agent, algorithm, time_limit)
            print(f"✅ Trained for {episodes_trained} episodes in {time_limit}s")

        print(f"\n🔍 Starting evaluation...")
        test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

        self._print_results(test_result)
        return test_result

    def _time_limited_training(self, agent, algorithm, time_limit):
        start_time = time.time()
        episode = 0
        rewards = []

        while time.time() - start_time < time_limit:
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            if algorithm == "sarsa":
                action = agent.get_action(state)

            while not done:
                if algorithm != "sarsa":
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

        return episode

    def benchmark(self, train_episodes, test_episodes):
        print(f"\n{'='*60}")
        print("BENCHMARK MODE")
        print(f"{'='*60}")
        print(f"Training Episodes: {train_episodes}")
        print(f"Test Episodes: {test_episodes}")

        results = []
        training_data = {}

        # BruteForce baseline
        print(f"\n🎲 Testing BruteForce baseline...")
        bf_agent = BruteForceAgent(self.n_actions, self.n_states)
        bf_result = evaluate_agent(self.env, bf_agent, test_episodes, "BruteForce")
        bf_result["training_time"] = 0
        bf_result["efficiency_score"] = 0
        results.append(bf_result)

        # RL algorithms
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

            # Combine results
            test_result["training_time"] = train_result["training_time"]
            test_result["convergence_episode"] = train_result["convergence_episode"]
            test_result["final_performance"] = train_result["final_performance"]

            results.append(test_result)
            training_data[algo_name] = train_result

        # Generate comprehensive analysis
        print(f"\n📊 Generating analysis...")
        plot_results(results, training_data)
        plot_training_analysis(training_data)

        self._print_benchmark_summary(results)
        return results

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

        # Best performer by different metrics
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

        # Performance improvement over brute force
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


def main():
    parser = argparse.ArgumentParser(description="Taxi Driver RL - Enhanced Version")
    parser.add_argument(
        "--mode",
        choices=["user", "time-limited", "benchmark"],
        default="benchmark",
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
    parser.add_argument("--alpha", type=float, default=0.15, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial exploration rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    print(f"🚕 TAXI DRIVER RL - Enhanced Version")
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


if __name__ == "__main__":
    main()
