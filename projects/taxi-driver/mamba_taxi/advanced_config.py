import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import os


class ExplorationStrategy(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    BOLTZMANN = "boltzmann"
    THOMPSON_SAMPLING = "thompson_sampling"
    ADAPTIVE_EPSILON = "adaptive_epsilon"
    CURIOSITY_DRIVEN = "curiosity_driven"


class LearningRateSchedule(Enum):
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    STEP_DECAY = "step_decay"
    COSINE_ANNEALING = "cosine_annealing"
    ADAPTIVE = "adaptive"


class RewardShaping(Enum):
    NONE = "none"
    DISTANCE_BASED = "distance_based"
    POTENTIAL_BASED = "potential_based"
    CURIOSITY_BONUS = "curiosity_bonus"
    TEMPORAL_DIFFERENCE = "temporal_difference"


@dataclass
class AdvancedConfig:
    """Advanced configuration with extensive tweaking options"""

    # Basic parameters
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Advanced exploration
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    ucb_c: float = 2.0
    boltzmann_temperature: float = 1.0
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    curiosity_factor: float = 0.1

    # Learning rate scheduling
    lr_schedule: LearningRateSchedule = LearningRateSchedule.CONSTANT
    lr_decay_rate: float = 0.99
    lr_step_size: int = 1000
    lr_min: float = 0.001
    lr_warmup_episodes: int = 100

    # Reward engineering
    reward_shaping: RewardShaping = RewardShaping.NONE
    step_penalty_base: float = -0.1
    step_penalty_scale: float = 1.0
    pickup_bonus: float = 10.0
    dropoff_bonus: float = 20.0
    illegal_penalty: float = -10.0
    distance_factor: float = 0.1
    time_factor: float = 0.05
    efficiency_bonus: float = 5.0

    # Advanced training parameters
    batch_learning: bool = False
    batch_size: int = 32
    experience_replay: bool = False
    replay_buffer_size: int = 10000
    prioritized_replay: bool = False
    target_network: bool = False
    target_update_freq: int = 100
    double_q_learning: bool = False

    # Network architecture (for neural networks)
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation_function: str = "relu"
    dropout_rate: float = 0.0
    batch_normalization: bool = False
    weight_initialization: str = "xavier"

    # Training strategies
    curriculum_learning: bool = False
    curriculum_stages: List[Dict] = field(default_factory=list)
    transfer_learning: bool = False
    pretrained_weights: Optional[str] = None
    fine_tuning_layers: List[int] = field(default_factory=list)

    # Convergence criteria
    early_stopping: bool = True
    patience: int = 200
    min_improvement: float = 0.01
    convergence_window: int = 100
    performance_threshold: float = 0.8

    # Evaluation parameters
    evaluation_frequency: int = 100
    evaluation_episodes: int = 20
    checkpoint_frequency: int = 500
    save_best_model: bool = True

    # Statistical tracking
    track_q_values: bool = True
    track_policy_entropy: bool = True
    track_value_estimates: bool = True
    track_exploration_coverage: bool = True
    track_learning_dynamics: bool = True

    # Experimental features
    meta_learning: bool = False
    multi_objective: bool = False
    ensemble_learning: bool = False
    uncertainty_estimation: bool = False
    continual_learning: bool = False


class ConfigurationManager:
    """Manages advanced configurations and parameter optimization"""

    def __init__(self, base_dir: str = "advanced_configs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.config_history = []
        self.performance_history = []

    def create_parameter_grid(
        self, base_config: AdvancedConfig
    ) -> List[AdvancedConfig]:
        """Create parameter grid for hyperparameter search"""
        grid_configs = []

        # Learning rate grid
        alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
        gammas = [0.9, 0.95, 0.99, 0.995]

        # Exploration strategies
        exploration_strategies = [
            ExplorationStrategy.EPSILON_GREEDY,
            ExplorationStrategy.UCB,
            ExplorationStrategy.BOLTZMANN,
            ExplorationStrategy.ADAPTIVE_EPSILON,
        ]

        # Reward shaping options
        reward_shapings = [
            RewardShaping.NONE,
            RewardShaping.DISTANCE_BASED,
            RewardShaping.POTENTIAL_BASED,
            RewardShaping.CURIOSITY_BONUS,
        ]

        config_id = 0
        for alpha in alphas:
            for gamma in gammas:
                for exploration in exploration_strategies:
                    for reward_shape in reward_shapings:
                        config = self._copy_config(base_config)
                        config.alpha = alpha
                        config.gamma = gamma
                        config.exploration_strategy = exploration
                        config.reward_shaping = reward_shape

                        # Adjust related parameters
                        if exploration == ExplorationStrategy.BOLTZMANN:
                            config.boltzmann_temperature = np.random.uniform(0.5, 2.0)
                        elif exploration == ExplorationStrategy.UCB:
                            config.ucb_c = np.random.uniform(1.0, 3.0)

                        grid_configs.append(config)
                        config_id += 1

                        # Limit grid size
                        if len(grid_configs) >= 200:
                            return grid_configs

        return grid_configs

    def create_adaptive_configs(
        self, performance_history: List[Dict]
    ) -> List[AdvancedConfig]:
        """Create adaptive configurations based on performance history"""
        if not performance_history:
            return [AdvancedConfig()]

        # Analyze top performers
        sorted_history = sorted(
            performance_history,
            key=lambda x: x.get("efficiency_score", 0),
            reverse=True,
        )

        top_configs = sorted_history[:5]
        adaptive_configs = []

        for config_data in top_configs:
            base_config = config_data.get("config", AdvancedConfig())

            # Create variations around successful configurations
            for _ in range(3):
                new_config = self._mutate_config(base_config)
                adaptive_configs.append(new_config)

        return adaptive_configs

    def optimize_bayesian(
        self, base_config: AdvancedConfig, n_iterations: int = 50
    ) -> List[AdvancedConfig]:
        """Bayesian optimization of hyperparameters"""
        # Simplified Bayesian optimization
        configs = []

        # Define parameter bounds
        param_bounds = {
            "alpha": (0.01, 0.5),
            "gamma": (0.9, 0.999),
            "epsilon_decay": (0.99, 0.9999),
            "step_penalty_base": (-1.0, -0.01),
            "pickup_bonus": (1.0, 20.0),
            "dropoff_bonus": (5.0, 50.0),
        }

        for i in range(n_iterations):
            config = self._copy_config(base_config)

            # Sample parameters using quasi-random sequences
            for param, (low, high) in param_bounds.items():
                if hasattr(config, param):
                    # Use Sobol sequence for better coverage
                    value = low + (high - low) * np.random.random()
                    setattr(config, param, value)

            configs.append(config)

        return configs

    def create_multi_objective_configs(
        self, base_config: AdvancedConfig
    ) -> List[AdvancedConfig]:
        """Create configurations optimized for multiple objectives"""
        configs = []

        # Speed-optimized configuration
        speed_config = self._copy_config(base_config)
        speed_config.alpha = 0.3
        speed_config.epsilon_decay = 0.99
        speed_config.early_stopping = True
        speed_config.patience = 50
        configs.append(speed_config)

        # Stability-optimized configuration
        stability_config = self._copy_config(base_config)
        stability_config.alpha = 0.05
        stability_config.gamma = 0.99
        stability_config.epsilon_decay = 0.999
        stability_config.convergence_window = 200
        configs.append(stability_config)

        # Exploration-optimized configuration
        explore_config = self._copy_config(base_config)
        explore_config.exploration_strategy = ExplorationStrategy.UCB
        explore_config.ucb_c = 2.5
        explore_config.curiosity_factor = 0.2
        configs.append(explore_config)

        # Sample efficiency configuration
        efficiency_config = self._copy_config(base_config)
        efficiency_config.experience_replay = True
        efficiency_config.batch_learning = True
        efficiency_config.prioritized_replay = True
        configs.append(efficiency_config)

        return configs

    def _copy_config(self, config: AdvancedConfig) -> AdvancedConfig:
        """Deep copy configuration"""
        return AdvancedConfig(**config.__dict__)

    def _mutate_config(
        self, config: AdvancedConfig, mutation_rate: float = 0.1
    ) -> AdvancedConfig:
        """Mutate configuration parameters"""
        new_config = self._copy_config(config)

        # Mutate numerical parameters
        if np.random.random() < mutation_rate:
            new_config.alpha *= np.random.uniform(0.8, 1.2)
            new_config.alpha = max(0.001, min(1.0, new_config.alpha))

        if np.random.random() < mutation_rate:
            new_config.gamma *= np.random.uniform(0.98, 1.02)
            new_config.gamma = max(0.8, min(0.999, new_config.gamma))

        if np.random.random() < mutation_rate:
            new_config.epsilon_decay *= np.random.uniform(0.99, 1.01)
            new_config.epsilon_decay = max(0.9, min(0.9999, new_config.epsilon_decay))

        # Mutate categorical parameters
        if np.random.random() < mutation_rate:
            strategies = list(ExplorationStrategy)
            new_config.exploration_strategy = np.random.choice(strategies)

        if np.random.random() < mutation_rate:
            shapings = list(RewardShaping)
            new_config.reward_shaping = np.random.choice(shapings)

        return new_config

    def save_config(
        self, config: AdvancedConfig, name: str, performance: Dict[str, Any] = None
    ):
        """Save configuration with performance data"""
        config_data = {
            "name": name,
            "config": config.__dict__,
            "performance": performance or {},
            "timestamp": np.datetime64("now").isoformat(),
        }

        filepath = os.path.join(self.base_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        self.config_history.append(config_data)

    def load_config(self, name: str) -> Optional[AdvancedConfig]:
        """Load configuration by name"""
        filepath = os.path.join(self.base_dir, f"{name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                return AdvancedConfig(**data["config"])
        return None

    def get_best_configs(
        self, metric: str = "efficiency_score", n_configs: int = 10
    ) -> List[AdvancedConfig]:
        """Get best performing configurations"""
        if not self.config_history:
            return []

        scored_configs = []
        for config_data in self.config_history:
            performance = config_data.get("performance", {})
            if metric in performance:
                scored_configs.append((performance[metric], config_data["config"]))

        scored_configs.sort(reverse=True)
        return [AdvancedConfig(**config) for _, config in scored_configs[:n_configs]]
