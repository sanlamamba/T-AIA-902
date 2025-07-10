# Comprehensive Usage Guide for Enhanced Taxi Driver RL

This guide covers all the advanced features and modes available in the enhanced Taxi Driver RL project.

## Quick Start

### 1. Installation and Setup

```bash
cd /workspaces/T-AIA-902/projects/taxi-driver/mamba_taxi
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Quick test with default settings
python main.py

# User mode with custom parameters
python main.py --mode user --algorithm q-learning --alpha 0.2 --gamma 0.95

# Time-limited training
python main.py --mode time-limited --algorithm sarsa --time-limit 120
```

## Available Modes

### 1. **User Mode** - Manual Parameter Tuning
```bash
python main.py --mode user --algorithm q-learning \
    --train-episodes 1000 --test-episodes 50 \
    --alpha 0.15 --gamma 0.99 --epsilon 1.0
```

**Features:**
- Manual control over all hyperparameters
- Real-time training progress
- Detailed performance metrics
- Parameter validation

### 2. **Time-Limited Mode** - Optimized Quick Training
```bash
python main.py --mode time-limited --algorithm dqn \
    --time-limit 300 --test-episodes 100
```

**Features:**
- Uses pre-optimized parameters
- Trains for specified time duration
- Automatic parameter selection
- Performance tracking during training

### 3. **Standard Benchmark Mode** - Basic Comparison
```bash
python main.py --mode benchmark \
    --train-episodes 2000 --test-episodes 100
```

**Features:**
- Compares Q-Learning, SARSA, and DQN
- BruteForce baseline comparison
- Standard visualizations
- Basic statistical analysis

### 4. **Advanced Benchmark Mode** - Comprehensive Analysis
```bash
python main.py --mode advanced-benchmark \
    --train-episodes 1500 --test-episodes 200
```

**Features:**
- Enhanced statistical analysis
- Comprehensive performance clustering
- Learning dynamics analysis
- Advanced visualizations (radar charts, distributions)
- Detailed result exports

### 5. **Hyperparameter Optimization Mode** - Automated Tuning
```bash
# Grid search optimization
python main.py --mode hyperopt --algorithm q-learning \
    --optimization-method grid --n-configs 50 \
    --train-episodes 500 --test-episodes 30

# Bayesian optimization
python main.py --mode hyperopt --algorithm dqn \
    --optimization-method bayesian --n-configs 30

# Multi-objective optimization
python main.py --mode hyperopt --algorithm sarsa \
    --optimization-method multi_objective
```

**Features:**
- Multiple optimization strategies
- Automatic parameter search
- Best configuration identification
- Performance tracking across configurations

### 6. **Comparative Analysis Mode** - Statistical Significance Testing
```bash
python main.py --mode comparative \
    --train-episodes 800 --test-episodes 50 --n-runs 10
```

**Features:**
- Multiple runs for statistical significance
- Hypothesis testing between algorithms
- Effect size analysis
- Confidence intervals
- Comprehensive statistical reports

## Advanced Configuration System

### Using Configuration Files

```python
from advanced_config import AdvancedConfig, ConfigurationManager

# Create custom configuration
config = AdvancedConfig(
    alpha=0.2,
    gamma=0.99,
    exploration_strategy=ExplorationStrategy.UCB,
    reward_shaping=RewardShaping.DISTANCE_BASED,
    early_stopping=True,
    patience=100
)

# Save configuration
config_manager = ConfigurationManager()
config_manager.save_config(config, "my_custom_config")
```

### Exploration Strategies Available

1. **Epsilon-Greedy** (default)
2. **UCB (Upper Confidence Bound)**
3. **Boltzmann Exploration**
4. **Thompson Sampling**
5. **Adaptive Epsilon**
6. **Curiosity-Driven**

### Reward Shaping Options

1. **None** (default environment rewards)
2. **Distance-Based** (rewards based on distance to goal)
3. **Potential-Based** (shaped rewards using potential functions)
4. **Curiosity Bonus** (rewards for exploring new states)
5. **Temporal Difference** (time-based reward modifications)

## Comprehensive Statistics and Analysis

### Accessing Advanced Statistics

The enhanced system provides extensive statistical analysis:

```python
from enhanced_statistics import AdvancedStatisticalAnalyzer

analyzer = AdvancedStatisticalAnalyzer()
analysis = analyzer.comprehensive_analysis(results, training_data)

# Available analysis types:
# - descriptive_stats: Mean, std, skewness, kurtosis, etc.
# - distribution_analysis: Normality tests, distribution fitting
# - correlation_analysis: Pearson, partial correlations
# - performance_clustering: K-means clustering of agents
# - learning_dynamics: Learning rate, convergence speed
# - stability_analysis: Performance consistency
# - outlier_analysis: Multiple outlier detection methods
# - hypothesis_testing: Statistical significance tests
# - effect_size_analysis: Cohen's d, etc.
```

## Output Files and Results

### Generated Files

After running advanced modes, you'll find:

```
results_YYYYMMDD_HHMMSS/
├── performance_results.json      # Raw performance metrics
├── training_data.json           # Training progression data
├── configurations.json          # Algorithm configurations used
├── statistical_analysis.json    # Comprehensive statistical analysis
├── comprehensive_report.txt     # Human-readable report
├── comprehensive_analysis.png   # Advanced visualizations
└── training_analysis.png       # Training dynamics plots
```

### Visualization Files

1. **comprehensive_analysis.png**: Multi-panel analysis including:
   - Performance comparison charts
   - Statistical distributions
   - Radar charts for multi-dimensional comparison
   - Learning efficiency analysis
   - Performance reliability metrics

2. **training_analysis.png**: Training dynamics including:
   - Learning curves with confidence intervals
   - Exploration vs exploitation patterns
   - Q-value learning dynamics
   - Convergence analysis

## Advanced Usage Examples

### 1. **Custom Hyperparameter Search**

```python
# Create custom parameter grid
from advanced_config import AdvancedConfig, ConfigurationManager

base_config = AdvancedConfig()
config_manager = ConfigurationManager()

# Generate parameter variants
configs = config_manager.create_parameter_grid(base_config)

# Test specific configurations
custom_variants = []
for config in configs[:10]:  # Test first 10 configurations
    config_dict = {
        'alpha': config.alpha,
        'gamma': config.gamma,
        'epsilon': config.epsilon,
        'epsilon_decay': config.epsilon_decay
    }
    custom_variants.append(('q-learning', config_dict))

# Run with custom variants
driver = TaxiDriver()
results = driver.advanced_benchmark(1000, 100, custom_variants)
```

### 2. **Multi-Algorithm Comparison with Custom Metrics**

```bash
# Compare all algorithms with extensive training
python main.py --mode comparative --n-runs 20 \
    --train-episodes 2000 --test-episodes 200

# This will generate:
# - Statistical significance tests between all algorithm pairs
# - Effect size analysis
# - Confidence intervals for all metrics
# - Performance clustering analysis
```

### 3. **Evolutionary Parameter Optimization**

```python
# Use the configuration manager for evolutionary optimization
config_manager = ConfigurationManager()

# Start with best known configurations
best_configs = config_manager.get_best_configs(metric='efficiency_score', n_configs=5)

# Create new generation through mutation
new_configs = []
for config in best_configs:
    for _ in range(3):  # 3 mutations per best config
        mutated = config_manager._mutate_config(config)
        new_configs.append(mutated)
```

### 4. **Real-time Performance Monitoring**

```python
# Monitor training progress in real-time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def monitor_training(algorithm, episodes):
    driver = TaxiDriver()
    agent = driver._create_agent(algorithm)
    
    rewards = []
    for episode in range(episodes):
        # Training step
        # ... training code ...
        
        # Real-time plotting every 50 episodes
        if episode % 50 == 0:
            clear_output(wait=True)
            plt.plot(rewards)
            plt.title(f'Training Progress - Episode {episode}')
            plt.show()
```

## Performance Optimization Tips

### 1. **Algorithm Selection Guide**

- **Q-Learning**: Best for tabular problems, fast convergence
- **SARSA**: More conservative, good for safety-critical applications
- **DQN**: Best for complex state spaces, handles non-linearity

### 2. **Hyperparameter Tuning Guidelines**

```python
# Recommended starting points:
optimized_params = {
    "q-learning": {
        "alpha": 0.15,        # Learning rate
        "gamma": 0.99,        # Discount factor
        "epsilon": 1.0,       # Initial exploration
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01
    },
    "sarsa": {
        "alpha": 0.15,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01
    },
    "dqn": {
        "alpha": 0.001,       # Lower learning rate for neural networks
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "memory_size": 10000,
        "batch_size": 32
    }
}
```

### 3. **Training Strategies**

```bash
# For quick results
python main.py --mode hyperopt --optimization-method multi_objective

# For thorough analysis
python main.py --mode comparative --n-runs 15

# For parameter sensitivity analysis
python main.py --mode hyperopt --optimization-method grid --n-configs 100
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Configurations**
   ```bash
   # Reduce batch size and memory size for DQN
   python main.py --mode user --algorithm dqn --train-episodes 500
   ```

2. **Slow Performance**
   ```bash
   # Use fewer episodes for testing
   python main.py --mode benchmark --train-episodes 500 --test-episodes 50
   ```

3. **Missing Dependencies**
   ```bash
   pip install scikit-learn seaborn
   ```

### Performance Expectations

- **BruteForce**: ~350 steps average, 0% learning
- **Basic Q-Learning**: ~50-100 steps after 1000 episodes
- **Optimized Algorithms**: ~20-30 steps after 2000 episodes
- **Advanced DQN**: ~15-25 steps with proper tuning

## Advanced Features Summary

### Statistical Analysis Features
- ✅ Descriptive statistics (mean, std, skewness, kurtosis)
- ✅ Distribution analysis and fitting
- ✅ Correlation and partial correlation analysis
- ✅ Performance clustering
- ✅ Outlier detection (multiple methods)
- ✅ Hypothesis testing
- ✅ Effect size analysis
- ✅ Confidence intervals

### Visualization Features
- ✅ Multi-metric performance comparison
- ✅ Training progress with confidence intervals
- ✅ Statistical distribution plots
- ✅ Radar charts for multi-dimensional comparison
- ✅ Learning efficiency analysis
- ✅ Convergence analysis plots

### Configuration Features
- ✅ Advanced parameter grid search
- ✅ Bayesian optimization
- ✅ Multi-objective optimization
- ✅ Configuration persistence
- ✅ Evolutionary parameter optimization

### Analysis Modes
- ✅ User mode with custom parameters
- ✅ Time-limited optimization
- ✅ Comprehensive benchmarking
- ✅ Hyperparameter optimization
- ✅ Comparative statistical analysis

This enhanced system provides research-grade analysis capabilities for reinforcement learning algorithm comparison and optimization.
