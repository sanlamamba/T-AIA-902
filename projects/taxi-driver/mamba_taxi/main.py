import argparse
import gymnasium as gym
import numpy as np
import time

from agents import BruteForceAgent, QLearningAgent, SARSAAgent, DQNAgent
from trainer import train_agent, evaluate_agent
from visualizer import plot_results


class TaxiDriver:
    def __init__(self):
        self.env = gym.make("Taxi-v3", render_mode=None)
        self.n_actions = self.env.action_space.n
        self.n_states = self.env.observation_space.n

    def user_mode(self, algorithm, train_episodes, test_episodes, **params):
        print(f"\n=== User Mode: {algorithm} ===")

        if algorithm == "bruteforce":
            agent = BruteForceAgent(self.n_actions, self.n_states)
        elif algorithm == "q-learning":
            agent = QLearningAgent(self.n_actions, self.n_states, **params)
        elif algorithm == "sarsa":
            agent = SARSAAgent(self.n_actions, self.n_states, **params)
        elif algorithm == "dqn":
            agent = DQNAgent(self.n_actions, self.n_states, **params)

        algo_name = algorithm.replace("-", " ").title()

        if algorithm != "bruteforce":
            train_result = train_agent(self.env, agent, train_episodes, algo_name)

        test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

        print(f"\nResults for {algo_name}:")
        print(f"Mean Reward: {test_result['mean_reward']:.2f}")
        print(f"Mean Steps: {test_result['mean_steps']:.2f}")
        print(f"Win Rate: {test_result['win_rate']:.2%}")

        return test_result

    def time_limited_mode(self, algorithm, time_limit, test_episodes):
        print(f"\n=== Time-Limited Mode: {algorithm} ({time_limit}s) ===")

        optimized_params = {
            "q-learning": {"alpha": 0.2, "gamma": 0.99, "epsilon": 0.1},
            "sarsa": {"alpha": 0.2, "gamma": 0.99, "epsilon": 0.1},
            "dqn": {
                "alpha": 0.001,
                "gamma": 0.99,
                "epsilon": 1.0,
                "epsilon_decay": 0.97,
                "epsilon_min": 0.05,
            },
        }

        params = optimized_params.get(algorithm, {})

        if algorithm == "bruteforce":
            agent = BruteForceAgent(self.n_actions, self.n_states)
        elif algorithm == "q-learning":
            agent = QLearningAgent(self.n_actions, self.n_states, **params)
        elif algorithm == "sarsa":
            agent = SARSAAgent(self.n_actions, self.n_states, **params)
        elif algorithm == "dqn":
            agent = DQNAgent(self.n_actions, self.n_states, **params)

        algo_name = algorithm.replace("-", " ").title()

        start_time = time.time()
        episode = 0

        while time.time() - start_time < time_limit:
            state, _ = self.env.reset()
            done = False

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

            episode += 1
            if episode % 100 == 0:
                print(f"Completed {episode} episodes...")

        print(f"Trained for {episode} episodes in {time_limit}s")

        test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)

        print(f"\nResults for {algo_name}:")
        print(f"Mean Reward: {test_result['mean_reward']:.2f}")
        print(f"Mean Steps: {test_result['mean_steps']:.2f}")
        print(f"Win Rate: {test_result['win_rate']:.2%}")

        return test_result

    def benchmark(self, train_episodes, test_episodes):
        print("\n=== Benchmark Mode ===")

        results = []
        training_data = {}

        print("\n--- BruteForce ---")
        bf_agent = BruteForceAgent(self.n_actions, self.n_states)
        bf_result = evaluate_agent(self.env, bf_agent, test_episodes, "BruteForce")
        bf_result["training_time"] = 0
        results.append(bf_result)

        algorithms = [
            ("q-learning", QLearningAgent, "Q-Learning"),
            ("sarsa", SARSAAgent, "SARSA"),
            ("dqn", DQNAgent, "DQN"),
        ]

        for algo_key, agent_class, algo_name in algorithms:
            print(f"\n--- {algo_name} ---")
            agent = agent_class(self.n_actions, self.n_states)

            train_result = train_agent(self.env, agent, train_episodes, algo_name)
            test_result = evaluate_agent(self.env, agent, test_episodes, algo_name)
            test_result["training_time"] = train_result["training_time"]

            results.append(test_result)
            training_data[algo_name] = {
                "rewards": train_result["rewards"],
                "steps": train_result["steps"],
            }

        plot_results(results, training_data)
        return results


def main():
    parser = argparse.ArgumentParser(description="Taxi Driver RL")
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
        "--train-episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--test-episodes", type=int, default=100, help="Number of test episodes"
    )
    parser.add_argument(
        "--time-limit", type=int, default=60, help="Time limit in seconds"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")

    args = parser.parse_args()

    np.random.seed(42)
    driver = TaxiDriver()

    if args.mode == "user":
        params = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}
        driver.user_mode(
            args.algorithm, args.train_episodes, args.test_episodes, **params
        )
    elif args.mode == "time-limited":
        driver.time_limited_mode(args.algorithm, args.time_limit, args.test_episodes)
    elif args.mode == "benchmark":
        driver.benchmark(args.train_episodes, args.test_episodes)


if __name__ == "__main__":
    main()
