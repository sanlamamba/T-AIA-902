import time
import numpy as np
import gymnasium as gym


def train_agent(env, agent, n_episodes, algorithm_name):
    start_time = time.time()
    rewards = []
    steps = []

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

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}: Avg Reward = {np.mean(rewards[-100:]):.2f}, Avg Steps = {np.mean(steps[-100:]):.2f}"
            )

    training_time = time.time() - start_time
    return {
        "algorithm": algorithm_name,
        "training_time": training_time,
        "rewards": rewards,
        "steps": steps,
    }


def evaluate_agent(env, agent, n_episodes, algorithm_name, render=False):
    print(f"\nEvaluating {algorithm_name} for {n_episodes} episodes...")

    test_rewards = []
    test_steps = []
    wins = 0

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

    return {
        "algorithm": algorithm_name,
        "mean_reward": np.mean(test_rewards),
        "mean_steps": np.mean(test_steps),
        "win_rate": wins / n_episodes,
    }
