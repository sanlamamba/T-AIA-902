import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def plot_results(results, training_data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    df = pd.DataFrame(results)
    metrics = ["mean_reward", "win_rate", "efficiency_score"]
    x = np.arange(len(df))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = df[metric] if metric in df.columns else [0] * len(df)
        ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Value")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["algorithm"])
    ax.legend()

    ax = axes[0, 1]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            rewards = data["rewards"]
            smoothed = _smooth_curve(rewards, window=50)
            ax.plot(smoothed, label=algo, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (Smoothed)")
    ax.set_title("Training Rewards Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success rate over time
    ax = axes[0, 2]
    for algo, data in training_data.items():
        if algo != "BruteForce" and "success_rates" in data:
            success_rates = data["success_rates"]
            smoothed_success = _smooth_curve(success_rates, window=100)
            ax.plot(smoothed_success, label=algo, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (Smoothed)")
    ax.set_title("Success Rate Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Steps distribution
    ax = axes[1, 0]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            steps = data["steps"]
            ax.hist(steps, bins=30, alpha=0.6, label=algo, density=True)
    ax.set_xlabel("Steps per Episode")
    ax.set_ylabel("Density")
    ax.set_title("Steps Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convergence analysis
    ax = axes[1, 1]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            convergence_point = data.get("convergence_episode", len(data["rewards"]))
            ax.axvline(x=convergence_point, label=f"{algo} convergence", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Convergence Point")
    ax.set_title("Algorithm Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary statistics table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for result in results:
        table_data.append(
            [
                result["algorithm"],
                f"{result['mean_reward']:.2f}±{result.get('std_reward', 0):.2f}",
                f"{result['mean_steps']:.1f}±{result.get('std_steps', 0):.1f}",
                f"{result['win_rate']:.2%}",
                f"{result.get('efficiency_score', 0):.4f}",
                f"{result.get('training_time', 0):.1f}s",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=[
            "Algorithm",
            "Reward±Std",
            "Steps±Std",
            "Win Rate",
            "Efficiency",
            "Time",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title("Comprehensive Statistics")

    plt.tight_layout()
    plt.savefig("results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    df = pd.DataFrame(results)
    print("\nDetailed Statistics:")
    print(df.to_string(index=False))

    # Statistical significance testing
    print("\n" + "-" * 50)
    print("STATISTICAL ANALYSIS")
    print("-" * 50)

    if len(results) > 2:
        algorithms = [r["algorithm"] for r in results if r["algorithm"] != "BruteForce"]
        print(f"\nRanking by efficiency score:")
        ranked = sorted(
            [r for r in results if r["algorithm"] != "BruteForce"],
            key=lambda x: x.get("efficiency_score", 0),
            reverse=True,
        )
        for i, result in enumerate(ranked, 1):
            print(
                f"{i}. {result['algorithm']}: {result.get('efficiency_score', 0):.4f}"
            )


def _smooth_curve(data, window=10):
    """Apply moving average smoothing to data"""
    if len(data) < window:
        return data

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed


def plot_training_analysis(training_data):
    """Create additional detailed training analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            rewards = data["rewards"]
            episodes = range(len(rewards))
            ax.plot(episodes, rewards, alpha=0.3, color="gray")
            smoothed = _smooth_curve(rewards, window=100)
            ax.plot(episodes, smoothed, label=algo, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curves (Raw + Smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training efficiency
    ax = axes[0, 1]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            steps = data["steps"]
            episodes = range(len(steps))
            smoothed_steps = _smooth_curve(steps, window=100)
            ax.plot(episodes, smoothed_steps, label=algo, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps to Complete")
    ax.set_title("Training Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
