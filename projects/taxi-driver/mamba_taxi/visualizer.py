import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for beautiful plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_results(results, training_data):
    """Create comprehensive performance analysis plots"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # 1. Performance Comparison (Multi-metric)
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_performance_comparison(ax1, results)

    # 2. Training Progress
    ax2 = fig.add_subplot(gs[0, 2:])
    _plot_training_progress(ax2, training_data)

    # 3. Statistical Distribution
    ax3 = fig.add_subplot(gs[1, :2])
    _plot_statistical_distributions(ax3, results)

    # 4. Convergence Analysis
    ax4 = fig.add_subplot(gs[1, 2:])
    _plot_convergence_analysis(ax4, training_data)

    # 5. Advanced Metrics Radar Chart
    ax5 = fig.add_subplot(gs[2, :2], projection="polar")
    _plot_radar_chart(ax5, results)

    # 6. Learning Efficiency
    ax6 = fig.add_subplot(gs[2, 2:])
    _plot_learning_efficiency(ax6, training_data)

    # 7. Performance Reliability
    ax7 = fig.add_subplot(gs[3, :2])
    _plot_performance_reliability(ax7, results)

    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[3, 2:])
    _create_advanced_summary_table(ax8, results)

    plt.suptitle(
        "Comprehensive RL Algorithm Analysis", fontsize=20, fontweight="bold", y=0.98
    )
    plt.savefig("comprehensive_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Generate detailed statistical report
    _generate_statistical_report(results, training_data)


def _plot_performance_comparison(ax, results):
    """Enhanced performance comparison with multiple metrics"""
    df = pd.DataFrame(results)
    algorithms = df["algorithm"].values

    # Metrics to compare
    metrics = {
        "Win Rate": "win_rate",
        "Efficiency Score": "efficiency_score",
        "Performance Index": "performance_index",
        "Success Consistency": "success_consistency",
    }

    x = np.arange(len(algorithms))
    width = 0.2
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))

    for i, (label, metric) in enumerate(metrics.items()):
        values = [result.get(metric, 0) for result in results]
        bars = ax.bar(
            x + i * width, values, width, label=label, color=colors[i], alpha=0.8
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel("Algorithm", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Multi-Metric Performance Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_training_progress(ax, training_data):
    """Enhanced training progress with confidence intervals"""
    for algo, data in training_data.items():
        if algo != "BruteForce":
            rewards = data["rewards"]
            episodes = range(len(rewards))

            # Calculate moving average and confidence interval
            window = 100
            smoothed = _smooth_curve(rewards, window)

            # Calculate rolling standard deviation for confidence interval
            rolling_std = []
            for i in range(len(rewards)):
                start = max(0, i - window // 2)
                end = min(len(rewards), i + window // 2 + 1)
                rolling_std.append(np.std(rewards[start:end]))

            smoothed_array = np.array(smoothed)
            std_array = np.array(rolling_std)

            ax.plot(episodes, smoothed, label=f"{algo} (smoothed)", linewidth=2)
            ax.fill_between(
                episodes,
                smoothed_array - std_array,
                smoothed_array + std_array,
                alpha=0.2,
            )

    ax.set_xlabel("Episode", fontweight="bold")
    ax.set_ylabel("Reward", fontweight="bold")
    ax.set_title(
        "Training Progress with Confidence Intervals", fontweight="bold", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_statistical_distributions(ax, results):
    """Plot statistical distributions of performance metrics"""
    algorithms = [r["algorithm"] for r in results if r["algorithm"] != "BruteForce"]

    if not algorithms:
        ax.text(
            0.5,
            0.5,
            "No RL algorithms to compare",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Box plot for reward distributions
    reward_data = []
    labels = []

    for result in results:
        if result["algorithm"] != "BruteForce" and "reward_distribution" in result:
            reward_data.append(result["reward_distribution"])
            labels.append(result["algorithm"])

    if reward_data:
        bp = ax.boxplot(reward_data, labels=labels, patch_artist=True, notch=True)

        # Customize box plot colors
        colors = plt.cm.Set2(np.linspace(0, 1, len(bp["boxes"])))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xlabel("Algorithm", fontweight="bold")
    ax.set_ylabel("Reward Distribution", fontweight="bold")
    ax.set_title("Performance Distribution Analysis", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)


def _plot_convergence_analysis(ax, training_data):
    """Analyze and plot convergence characteristics"""
    convergence_data = []
    algorithms = []

    for algo, data in training_data.items():
        if algo != "BruteForce":
            convergence_point = data.get("convergence_episode", len(data["rewards"]))
            sample_efficiency = data.get("sample_efficiency", len(data["rewards"]))

            convergence_data.append([convergence_point, sample_efficiency])
            algorithms.append(algo)

    if convergence_data:
        convergence_array = np.array(convergence_data)

        # Scatter plot of convergence vs sample efficiency
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))

        for i, (algo, color) in enumerate(zip(algorithms, colors)):
            ax.scatter(
                convergence_array[i, 0],
                convergence_array[i, 1],
                s=100,
                c=[color],
                alpha=0.7,
                label=algo,
                edgecolors="black",
            )

        # Add diagonal line for reference
        max_episodes = max(convergence_array.flatten())
        ax.plot(
            [0, max_episodes],
            [0, max_episodes],
            "k--",
            alpha=0.5,
            label="Convergence = Efficiency",
        )

    ax.set_xlabel("Convergence Episode", fontweight="bold")
    ax.set_ylabel("Sample Efficiency (80% threshold)", fontweight="bold")
    ax.set_title("Convergence vs Sample Efficiency", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_radar_chart(ax, results):
    """Create radar chart for multi-dimensional performance comparison"""
    # Select algorithms (exclude BruteForce for cleaner visualization)
    algorithms = [r for r in results if r["algorithm"] != "BruteForce"]

    if not algorithms:
        ax.text(
            0.5,
            0.5,
            "No RL algorithms for radar chart",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    # Define metrics for radar chart
    metrics = [
        "win_rate",
        "efficiency_score",
        "performance_index",
        "success_consistency",
    ]
    metric_labels = [
        "Win Rate",
        "Efficiency",
        "Performance\nIndex",
        "Success\nConsistency",
    ]

    # Number of metrics
    N = len(metrics)

    # Angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))

    for i, (result, color) in enumerate(zip(algorithms, colors)):
        values = [result.get(metric, 0) for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles, values, "o-", linewidth=2, label=result["algorithm"], color=color
        )
        ax.fill(angles, values, alpha=0.25, color=color)

    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Multi-Dimensional Performance Radar", fontweight="bold", fontsize=14, pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)


def _plot_learning_efficiency(ax, training_data):
    """Plot learning efficiency metrics"""
    algorithms = []
    auc_scores = []
    stability_scores = []

    for algo, data in training_data.items():
        if algo != "BruteForce":
            algorithms.append(algo)
            auc_scores.append(data.get("learning_curve_auc", 0))
            stability_scores.append(data.get("stability_score", 0))

    if algorithms:
        x = np.arange(len(algorithms))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, auc_scores, width, label="Learning Speed (AUC)", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, stability_scores, width, label="Stability Score", alpha=0.8
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    ax.set_xlabel("Algorithm", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Learning Efficiency Analysis", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_performance_reliability(ax, results):
    """Plot performance reliability metrics"""
    algorithms = []
    win_streaks = []
    cv_rewards = []
    cv_steps = []

    for result in results:
        if result["algorithm"] != "BruteForce":
            algorithms.append(result["algorithm"])
            win_streaks.append(result.get("max_win_streak", 0))
            cv_rewards.append(
                1 / (1 + result.get("reward_cv", float("inf")))
            )  # Inverse CV for better is higher
            cv_steps.append(1 / (1 + result.get("step_cv", float("inf"))))

    if algorithms:
        x = np.arange(len(algorithms))
        width = 0.25

        bars1 = ax.bar(x - width, win_streaks, width, label="Max Win Streak", alpha=0.8)
        bars2 = ax.bar(x, cv_rewards, width, label="Reward Consistency", alpha=0.8)
        bars3 = ax.bar(x + width, cv_steps, width, label="Step Consistency", alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Algorithm", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance Reliability Analysis", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _create_advanced_summary_table(ax, results):
    """Create comprehensive summary statistics table"""
    ax.axis("off")

    # Prepare table data
    table_data = []
    headers = [
        "Algorithm",
        "Mean Reward±Std",
        "Win Rate",
        "Efficiency",
        "Consistency",
        "Max Streak",
        "Performance Index",
    ]

    for result in results:
        table_data.append(
            [
                result["algorithm"],
                f"{result['mean_reward']:.2f}±{result.get('std_reward', 0):.2f}",
                f"{result['win_rate']:.2%}",
                f"{result.get('efficiency_score', 0):.4f}",
                f"{result.get('success_consistency', 0):.3f}",
                f"{result.get('max_win_streak', 0):.0f}",
                f"{result.get('performance_index', 0):.4f}",
            ]
        )

    # Create table
    table = ax.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color code performance
    for i in range(1, len(table_data) + 1):
        if "BruteForce" in table_data[i - 1][0]:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#ffcccc")
        else:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#e6f3ff")

    ax.set_title(
        "Comprehensive Performance Summary", fontweight="bold", fontsize=14, pad=20
    )


def _generate_statistical_report(results, training_data):
    """Generate detailed statistical analysis report"""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    print("=" * 100)

    # Overall performance ranking
    rl_results = [r for r in results if r["algorithm"] != "BruteForce"]

    if rl_results:
        print("\n📊 PERFORMANCE RANKINGS")
        print("-" * 50)

        # Rank by different metrics
        rankings = {
            "Win Rate": sorted(rl_results, key=lambda x: x["win_rate"], reverse=True),
            "Efficiency Score": sorted(
                rl_results, key=lambda x: x.get("efficiency_score", 0), reverse=True
            ),
            "Performance Index": sorted(
                rl_results, key=lambda x: x.get("performance_index", 0), reverse=True
            ),
            "Consistency": sorted(
                rl_results, key=lambda x: x.get("success_consistency", 0), reverse=True
            ),
        }

        for metric, ranked_algos in rankings.items():
            print(f"\n{metric}:")
            for i, algo in enumerate(ranked_algos, 1):
                score = algo.get(metric.lower().replace(" ", "_"), 0)
                print(f"  {i}. {algo['algorithm']}: {score:.4f}")

    # Statistical significance testing
    if len(rl_results) > 1:
        print(f"\n🔬 STATISTICAL SIGNIFICANCE TESTING")
        print("-" * 50)

        # Pairwise comparisons for reward distributions
        for i, algo1 in enumerate(rl_results):
            for j, algo2 in enumerate(rl_results[i + 1 :], i + 1):
                if "reward_distribution" in algo1 and "reward_distribution" in algo2:
                    dist1 = algo1["reward_distribution"]
                    dist2 = algo2["reward_distribution"]

                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(
                        dist1, dist2, alternative="two-sided"
                    )

                    significance = (
                        "***"
                        if p_value < 0.001
                        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    )

                    print(
                        f"{algo1['algorithm']} vs {algo2['algorithm']}: p = {p_value:.4f} {significance}"
                    )

    # Performance improvement analysis
    bf_result = next((r for r in results if r["algorithm"] == "BruteForce"), None)
    if bf_result:
        print(f"\n📈 IMPROVEMENT OVER BRUTE FORCE")
        print("-" * 50)

        for result in rl_results:
            reward_improvement = (
                (result["mean_reward"] - bf_result["mean_reward"])
                / abs(bf_result["mean_reward"])
            ) * 100
            step_improvement = (
                (bf_result["mean_steps"] - result["mean_steps"])
                / bf_result["mean_steps"]
            ) * 100
            efficiency_ratio = result.get("efficiency_score", 0) / max(
                bf_result.get("efficiency_score", 0.001), 0.001
            )

            print(
                f"{result['algorithm']:15} | Reward: {reward_improvement:+7.1f}% | Steps: {step_improvement:+7.1f}% | Efficiency: {efficiency_ratio:.1f}x"
            )

    # Learning characteristics
    print(f"\n🎯 LEARNING CHARACTERISTICS")
    print("-" * 50)

    for algo, data in training_data.items():
        if algo != "BruteForce":
            convergence = data.get("convergence_episode", len(data["rewards"]))
            sample_eff = data.get("sample_efficiency", len(data["rewards"]))
            learning_auc = data.get("learning_curve_auc", 0)
            stability = data.get("stability_score", 0)

            print(
                f"{algo:15} | Convergence: {convergence:4d} ep | Sample Eff: {sample_eff:4d} ep | Learning AUC: {learning_auc:.3f} | Stability: {stability:.3f}"
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
    """Create detailed training analysis with advanced metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Advanced Training Analysis", fontsize=16, fontweight="bold")

    # 1. Learning curves with trend analysis
    ax = axes[0, 0]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            rewards = data["rewards"]
            episodes = range(len(rewards))

            # Plot raw data with transparency
            ax.plot(episodes, rewards, alpha=0.3, linewidth=0.5)

            # Plot smoothed curve
            smoothed = _smooth_curve(rewards, window=100)
            ax.plot(episodes, smoothed, label=algo, linewidth=2)

            # Add trend line
            if len(rewards) > 50:
                z = np.polyfit(
                    episodes[-len(rewards) // 2 :], rewards[-len(rewards) // 2 :], 1
                )
                p = np.poly1d(z)
                ax.plot(
                    episodes[-len(rewards) // 2 :],
                    p(episodes[-len(rewards) // 2 :]),
                    "--",
                    alpha=0.7,
                    linewidth=1,
                )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curves with Trend Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Exploration vs Exploitation
    ax = axes[0, 1]
    for algo, data in training_data.items():
        if algo != "BruteForce" and "exploration_rates" in data:
            exploration = data["exploration_rates"]
            episodes = range(len(exploration))
            ax.plot(episodes, exploration, label=f"{algo} ε-rate", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration Rate")
    ax.set_title("Exploration Rate Decay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Q-value change analysis
    ax = axes[1, 0]
    for algo, data in training_data.items():
        if algo != "BruteForce" and "q_value_changes" in data:
            q_changes = data["q_value_changes"]
            smoothed_changes = _smooth_curve(q_changes, window=50)
            episodes = range(len(smoothed_changes))
            ax.plot(episodes, smoothed_changes, label=f"{algo} Q-changes", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Q-value Change")
    ax.set_title("Q-value Learning Dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 4. Episode duration analysis
    ax = axes[1, 1]
    for algo, data in training_data.items():
        if algo != "BruteForce" and "episode_times" in data:
            times = data["episode_times"]
            smoothed_times = _smooth_curve(times, window=50)
            episodes = range(len(smoothed_times))
            ax.plot(episodes, smoothed_times, label=f"{algo} duration", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Duration (s)")
    ax.set_title("Training Episode Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Success rate progression
    ax = axes[2, 0]
    for algo, data in training_data.items():
        if algo != "BruteForce" and "success_rates" in data:
            success_rates = data["success_rates"]
            # Calculate rolling success rate
            window = 100
            rolling_success = []
            for i in range(len(success_rates)):
                start = max(0, i - window + 1)
                rolling_success.append(np.mean(success_rates[start : i + 1]))

            episodes = range(len(rolling_success))
            ax.plot(
                episodes, rolling_success, label=f"{algo} success rate", linewidth=2
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Success Rate")
    ax.set_title("Success Rate Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 6. Performance variance analysis
    ax = axes[2, 1]
    algorithms = []
    reward_vars = []
    step_vars = []

    for algo, data in training_data.items():
        if algo != "BruteForce":
            algorithms.append(algo)
            reward_vars.append(data.get("reward_variance", 0))
            step_vars.append(data.get("step_variance", 0))

    x = np.arange(len(algorithms))
    width = 0.35

    ax.bar(x - width / 2, reward_vars, width, label="Reward Variance", alpha=0.8)
    ax.bar(x + width / 2, step_vars, width, label="Step Variance", alpha=0.8)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Variance")
    ax.set_title("Performance Variance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("advanced_training_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
