import time
import numpy as np
import gymnasium as gym


def train_agent(env, agent, n_episodes, algorithm_name):
    start_time = time.time()
    rewards = []
    steps = []
    success_rate_window = []
    convergence_window = 100

    print(f"\nTraining {algorithm_name} for {n_episodes} episodes...")

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        if algorithm_name == "SARSA":
            action = agent.get_action(state)

        while not done:
            if algorithm_name != "SARSA":
                action = agent.get_action(state, explore=True)

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            if algorithm_name == "Q-Learning":
                agent.update(state, action, reward, next_state, done)
            elif algorithm_name == "SARSA":
                next_action = agent.get_action(next_state, explore=True)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action
            elif algorithm_name == "DQN":
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

            state = next_state
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        steps.append(step_count)
        success_rate_window.append(1 if total_reward > 0 else 0)

        if (episode + 1) % 100 == 0:
            recent_rewards = rewards[-100:]
            recent_steps = steps[-100:]
            recent_success = success_rate_window[-100:]

            print(
                f"Episode {episode + 1}: "
                f"Avg Reward = {np.mean(recent_rewards):.2f}, "
                f"Avg Steps = {np.mean(recent_steps):.2f}, "
                f"Success Rate = {np.mean(recent_success):.2%}"
            )

            if hasattr(agent, "epsilon"):
                print(f"  Epsilon: {agent.epsilon:.4f}")

    training_time = time.time() - start_time

    # Calculate convergence metrics
    final_performance = (
        np.mean(rewards[-convergence_window:])
        if len(rewards) >= convergence_window
        else np.mean(rewards)
    )

    return {
        "algorithm": algorithm_name,
        "training_time": training_time,
        "rewards": rewards,
        "steps": steps,
        "success_rates": success_rate_window,
        "final_performance": final_performance,
        "convergence_episode": _find_convergence_point(rewards, convergence_window),
    }


def _find_convergence_point(rewards, window_size=100):
    """Find the episode where the algorithm converged (stabilized performance)"""
    if len(rewards) < window_size * 2:
        return len(rewards)

    for i in range(window_size, len(rewards) - window_size):
        current_window = rewards[i : i + window_size]
        next_window = rewards[i + window_size : i + 2 * window_size]

        # Check if the difference between windows is small (converged)
        if abs(np.mean(current_window) - np.mean(next_window)) < 0.5:
            return i

    return len(rewards)


def evaluate_agent(env, agent, n_episodes, algorithm_name, render=False):
    print(f"\nEvaluating {algorithm_name} for {n_episodes} episodes...")

    test_rewards = []
    test_steps = []
    wins = 0
    step_efficiency = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.get_action(state, explore=False)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            if render and episode == 0:
                env.render()
                time.sleep(0.1)

            state = next_state
            total_reward += reward
            step_count += 1

        test_rewards.append(total_reward)
        test_steps.append(step_count)
        if total_reward > 0:
            wins += 1
            step_efficiency.append(step_count)

    return {
        "algorithm": algorithm_name,
        "mean_reward": np.mean(test_rewards),
        "std_reward": np.std(test_rewards),
        "mean_steps": np.mean(test_steps),
        "std_steps": np.std(test_steps),
        "win_rate": wins / n_episodes,
        "mean_success_steps": (
            np.mean(step_efficiency) if step_efficiency else float("inf")
        ),
        "efficiency_score": (
            wins / np.mean(test_steps) if np.mean(test_steps) > 0 else 0
        ),
    }
