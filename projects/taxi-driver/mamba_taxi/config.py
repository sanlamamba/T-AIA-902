OPTIMAL_CONFIGS = {
    "Q-Learning": {
        "alpha": 0.23,
        "gamma": 0.987,
        "epsilon": 0.8,
        "epsilon_decay": 0.9945,
        "epsilon_min": 0.01,
        "episodes": 800,
    },
    "SARSA": {
        "alpha": 0.19,
        "gamma": 0.991,
        "epsilon": 0.7,
        "epsilon_decay": 0.9952,
        "epsilon_min": 0.01,
        "episodes": 850,
    },
    "DQN": {
        "alpha": 0.15,
        "gamma": 0.995,
        "epsilon": 0.8,
        "epsilon_decay": 0.9945,
        "epsilon_min": 0.01,
        "memory_size": 15000,
        "batch_size": 64,
        "target_update_freq": 100,
        "episodes": 900,
    },
}

FAST_CONFIGS = {
    "Q-Learning": {
        "alpha": 0.3,
        "gamma": 0.99,
        "epsilon": 0.7,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "episodes": 400,
    },
    "SARSA": {
        "alpha": 0.25,
        "gamma": 0.99,
        "epsilon": 0.7,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "episodes": 400,
    },
    "DQN": {
        "alpha": 0.2,
        "gamma": 0.99,
        "epsilon": 0.7,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "memory_size": 10000,
        "batch_size": 32,
        "target_update_freq": 50,
        "episodes": 500,
    },
}

HIGH_PERFORMANCE_CONFIGS = {
    "Q-Learning": {
        "alpha": 0.15,
        "gamma": 0.999,
        "epsilon": 0.9,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "episodes": 1200,
    },
    "SARSA": {
        "alpha": 0.12,
        "gamma": 0.999,
        "epsilon": 0.9,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "episodes": 1200,
    },
    "DQN": {
        "alpha": 0.1,
        "gamma": 0.999,
        "epsilon": 0.9,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "memory_size": 20000,
        "batch_size": 128,
        "target_update_freq": 200,
        "episodes": 1500,
    },
}

DEFAULT_CONFIG_TYPE = "optimal"


def get_config(algorithm, config_type="optimal"):
    """
    Get optimized configuration for an algorithm

    Args:
        algorithm (str): Algorithm name ('Q-Learning', 'SARSA', 'DQN')
        config_type (str): Configuration type ('optimal', 'fast', 'high_performance')

    Returns:
        dict: Configuration parameters
    """
    config_map = {
        "optimal": OPTIMAL_CONFIGS,
        "fast": FAST_CONFIGS,
        "high_performance": HIGH_PERFORMANCE_CONFIGS,
    }

    return config_map.get(config_type, OPTIMAL_CONFIGS).get(algorithm, {})


def get_default_config(algorithm):
    """Get the default optimized configuration for an algorithm"""
    return get_config(algorithm, DEFAULT_CONFIG_TYPE)


ENV_CONFIG = {
    "render_mode": None,
    "max_episode_steps": 200,
}

TRAINING_CONFIG = {
    "test_episodes": 100,
    "eval_frequency": 100,
    "save_frequency": 200,
    "early_stopping": True,
    "patience": 200,
    "min_improvement": 0.01,
}

METRICS_CONFIG = {
    "track_reward": True,
    "track_steps": True,
    "track_success_rate": True,
    "track_efficiency": True,
    "track_convergence": True,
}
