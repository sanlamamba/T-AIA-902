import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_results(results, training_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    df = pd.DataFrame(results)
    metrics = ["mean_reward", "mean_steps", "win_rate"]
    x = np.arange(len(df))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Value")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["algorithm"])
    ax.legend()

    ax = axes[0, 1]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            ax.plot(data["rewards"], label=algo, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards")
    ax.legend()

    ax = axes[1, 0]
    for algo, data in training_data.items():
        if algo != "BruteForce":
            ax.plot(data["steps"], label=algo, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Training Steps")
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for result in results:
        table_data.append(
            [
                result["algorithm"],
                f"{result['mean_reward']:.2f}",
                f"{result['mean_steps']:.2f}",
                f"{result['win_rate']:.2%}",
                f"{result.get('training_time', 0):.2f}s",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Algorithm", "Avg Reward", "Avg Steps", "Win Rate", "Time"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Summary Statistics")

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()

    print("\nPerformance Summary:")
    print(pd.DataFrame(results).to_string(index=False))
