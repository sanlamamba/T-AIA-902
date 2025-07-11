import time
import numpy as np
import gymnasium as gym
from scipy import stats


def train_agent(env, agent, n_episodes, algorithm_name):
    start_time = time.time()
    rewards = []
    steps = []
    success_rate_window = []
    convergence_window = 100
    episode_times = []
    q_value_changes = []
    exploration_rates = []

    print(f"\nTraining {algorithm_name} for {n_episodes} episodes...")

    for episode in range(n_episodes):
        episode_start = time.time()
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        episode_q_changes = []

        if algorithm_name == "SARSA":
            action = agent.get_action(state)

        while not done:
            if algorithm_name != "SARSA":
                action = agent.get_action(state, explore=True)

            # Track Q-value changes for tabular methods
            if hasattr(agent, "q_table"):
                old_q = agent.q_table[state, action].copy()

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

            # Track Q-value changes
            if hasattr(agent, "q_table"):
                new_q = agent.q_table[state, action]
                q_change = abs(new_q - old_q)
                episode_q_changes.append(q_change)

            state = next_state
            total_reward += reward
            step_count += 1

        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        rewards.append(total_reward)
        steps.append(step_count)
        success_rate_window.append(1 if total_reward > 0 else 0)

        # Track exploration rate
        if hasattr(agent, "epsilon"):
            exploration_rates.append(agent.epsilon)
        else:
            exploration_rates.append(0)

        # Track average Q-value change for this episode
        if episode_q_changes:
            q_value_changes.append(np.mean(episode_q_changes))
        else:
            q_value_changes.append(0)

        if (episode + 1) % 100 == 0:
            recent_rewards = rewards[-100:]
            recent_steps = steps[-100:]
            recent_success = success_rate_window[-100:]
            recent_times = episode_times[-100:]

            print(
                f"Episode {episode + 1}: "
                f"Avg Reward = {np.mean(recent_rewards):.2f}, "
                f"Avg Steps = {np.mean(recent_steps):.2f}, "
                f"Success Rate = {np.mean(recent_success):.2%}, "
                f"Avg Time = {np.mean(recent_times):.3f}s"
            )

            if hasattr(agent, "epsilon"):
                print(f"  Epsilon: {agent.epsilon:.4f}")

    training_time = time.time() - start_time

    # Calculate advanced metrics
    final_performance = (
        np.mean(rewards[-convergence_window:])
        if len(rewards) >= convergence_window
        else np.mean(rewards)
    )

    # Learning efficiency metrics
    learning_curve_auc = _calculate_learning_curve_auc(rewards)
    stability_score = _calculate_stability_score(
        rewards[-convergence_window:] if len(rewards) >= convergence_window else rewards
    )
    sample_efficiency = _calculate_sample_efficiency(rewards, success_rate_window)

    return {
        "algorithm": algorithm_name,
        "training_time": training_time,
        "rewards": rewards,
        "steps": steps,
        "success_rates": success_rate_window,
        "episode_times": episode_times,
        "exploration_rates": exploration_rates,
        "q_value_changes": q_value_changes,
        "final_performance": final_performance,
        "convergence_episode": _find_convergence_point(rewards, convergence_window),
        "learning_curve_auc": learning_curve_auc,
        "stability_score": stability_score,
        "sample_efficiency": sample_efficiency,
        "reward_variance": np.var(rewards),
        "step_variance": np.var(steps),
        "peak_performance": np.max(rewards),
        "peak_performance_episode": np.argmax(rewards),
    }


def _calculate_learning_curve_auc(rewards, window_size=50):
    """Calculate Area Under the Learning Curve - measures learning speed"""
    if len(rewards) < window_size:
        return 0

    smoothed_rewards = []
    for i in range(window_size, len(rewards)):
        smoothed_rewards.append(np.mean(rewards[i - window_size : i]))

    # Normalize to [0, 1] and calculate AUC
    if len(smoothed_rewards) > 0:
        min_reward = min(smoothed_rewards)
        max_reward = max(smoothed_rewards)
        if max_reward > min_reward:
            normalized = [
                (r - min_reward) / (max_reward - min_reward) for r in smoothed_rewards
            ]
            return np.trapz(normalized) / len(normalized)
    return 0


def _calculate_stability_score(rewards):
    """Calculate stability score - lower variance means more stable"""
    if len(rewards) < 2:
        return 0

    variance = np.var(rewards)
    mean_reward = np.mean(rewards)
    if mean_reward != 0:
        cv = np.sqrt(variance) / abs(mean_reward)  # Coefficient of variation
        return 1 / (1 + cv)  # Higher score = more stable
    return 0


def _calculate_sample_efficiency(rewards, success_rates, threshold=0.8):
    """Calculate sample efficiency - episodes needed to reach threshold performance"""
    window_size = 50
    for i in range(window_size, len(success_rates)):
        window_success = np.mean(success_rates[i - window_size : i])
        if window_success >= threshold:
            return i
    return len(rewards)  # Never reached threshold


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
    test_times = []
    wins = 0
    step_efficiency = []
    reward_distribution = []
    consecutive_wins = []
    current_win_streak = 0
    max_win_streak = 0

    for episode in range(n_episodes):
        episode_start = time.time()
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

        episode_time = time.time() - episode_start
        test_times.append(episode_time)
        test_rewards.append(total_reward)
        test_steps.append(step_count)
        reward_distribution.append(total_reward)

        if total_reward > 0:
            wins += 1
            step_efficiency.append(step_count)
            current_win_streak += 1
            max_win_streak = max(max_win_streak, current_win_streak)
        else:
            current_win_streak = 0

        consecutive_wins.append(current_win_streak)

    # Advanced statistical metrics
    rewards_array = np.array(test_rewards)
    steps_array = np.array(test_steps)

    # Percentile analysis
    reward_percentiles = {
        "25th": np.percentile(rewards_array, 25),
        "50th": np.percentile(rewards_array, 50),
        "75th": np.percentile(rewards_array, 75),
        "90th": np.percentile(rewards_array, 90),
        "95th": np.percentile(rewards_array, 95),
    }

    step_percentiles = {
        "25th": np.percentile(steps_array, 25),
        "50th": np.percentile(steps_array, 50),
        "75th": np.percentile(steps_array, 75),
        "90th": np.percentile(steps_array, 90),
        "95th": np.percentile(steps_array, 95),
    }

    # Consistency metrics
    reward_cv = (
        np.std(rewards_array) / np.mean(rewards_array)
        if np.mean(rewards_array) != 0
        else float("inf")
    )
    step_cv = (
        np.std(steps_array) / np.mean(steps_array)
        if np.mean(steps_array) != 0
        else float("inf")
    )

    # Performance reliability
    success_consistency = wins / n_episodes
    performance_index = (np.mean(rewards_array) * success_consistency) / (
        np.mean(steps_array) + 1
    )

    return {
        "algorithm": algorithm_name,
        "mean_reward": np.mean(test_rewards),
        "std_reward": np.std(test_rewards),
        "mean_steps": np.mean(test_steps),
        "std_steps": np.std(test_steps),
        "mean_time": np.mean(test_times),
        "std_time": np.std(test_times),
        "win_rate": wins / n_episodes,
        "mean_success_steps": (
            np.mean(step_efficiency) if step_efficiency else float("inf")
        ),
        "efficiency_score": (
            wins / np.mean(test_steps) if np.mean(test_steps) > 0 else 0
        ),
        "reward_percentiles": reward_percentiles,
        "step_percentiles": step_percentiles,
        "reward_cv": reward_cv,
        "step_cv": step_cv,
        "max_win_streak": max_win_streak,
        "avg_win_streak": (
            np.mean([x for x in consecutive_wins if x > 0])
            if any(x > 0 for x in consecutive_wins)
            else 0
        ),
        "performance_index": performance_index,
        "success_consistency": success_consistency,
        "reward_distribution": reward_distribution,
        "step_distribution": test_steps,
        "min_reward": np.min(rewards_array),
        "max_reward": np.max(rewards_array),
        "min_steps": np.min(steps_array),
        "max_steps": np.max(steps_array),
        "reward_range": np.max(rewards_array) - np.min(rewards_array),
        "step_range": np.max(steps_array) - np.min(steps_array),
    }
