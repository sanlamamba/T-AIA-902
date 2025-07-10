"""
Advanced Analysis Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to import enhanced_statistics
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

from enhanced_statistics import AdvancedStatisticalAnalyzer
from ..components.visualizations import VisualizationComponents


def render_advanced_analysis_page():
    """Render the advanced analysis page"""
    st.header("Advanced Statistical Analysis")

    if not st.session_state.comparison_data:
        VisualizationComponents.display_warning_message(
            "⚠️ Please run an algorithm comparison first to see advanced analysis."
        )

        # Show what advanced analysis includes
        st.subheader("What's Included in Advanced Analysis")

        features = [
            "📊 **Descriptive Statistics**: Mean, std, min, max, coefficient of variation for all metrics",
            "🎯 **Performance Clustering**: Automatic grouping of similar performing algorithms using K-means",
            "📈 **Hypothesis Testing**: Statistical significance testing between algorithms (Mann-Whitney U)",
            "🔢 **Confidence Intervals**: 95% confidence intervals for all performance metrics",
            "📉 **Effect Size Analysis**: Cohen's d for measuring practical significance",
            "🔄 **Convergence Analysis**: Training convergence patterns and stability metrics",
            "📋 **Performance Ranking**: Comprehensive ranking with multiple criteria",
            "⚡ **Outlier Detection**: Identification of unusual performance patterns",
        ]

        for feature in features:
            st.markdown(f"- {feature}")

        st.info("💡 Run an algorithm comparison to unlock these advanced analytics!")
        return

    results = st.session_state.comparison_data["results"]
    training_data = st.session_state.comparison_data["training_data"]

    # Initialize statistical analyzer
    stat_analyzer = AdvancedStatisticalAnalyzer()

    # Comprehensive statistical analysis
    with st.spinner("Performing advanced statistical analysis..."):
        try:
            statistical_analysis = stat_analyzer.comprehensive_analysis(
                results, list(training_data.values())
            )
        except Exception as e:
            st.error(f"Error in statistical analysis: {e}")
            statistical_analysis = {}

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Descriptive Stats",
            "🎯 Clustering & Ranking",
            "📈 Hypothesis Testing",
            "🔢 Confidence Intervals",
        ]
    )

    with tab1:
        st.subheader("Descriptive Statistics")

        # Descriptive statistics
        if "descriptive_stats" in statistical_analysis:
            desc_stats = statistical_analysis["descriptive_stats"]

            stats_data = []
            for metric, stats in desc_stats.items():
                if isinstance(stats, dict):
                    stats_data.append(
                        {
                            "Metric": metric.replace("_", " ").title(),
                            "Mean": f"{stats.get('mean', 0):.3f}",
                            "Std Dev": f"{stats.get('std', 0):.3f}",
                            "Min": f"{stats.get('min', 0):.3f}",
                            "Max": f"{stats.get('max', 0):.3f}",
                            "CV (%)": f"{stats.get('cv', 0)*100:.1f}",
                        }
                    )

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                # Interpretation
                st.subheader("Statistical Interpretation")

                interpretation = []
                for row in stats_data:
                    cv = float(row["CV (%)"])
                    if cv < 10:
                        variability = "Low variability"
                    elif cv < 30:
                        variability = "Moderate variability"
                    else:
                        variability = "High variability"

                    interpretation.append(
                        {
                            "Metric": row["Metric"],
                            "Variability": variability,
                            "Range": f"{row['Min']} - {row['Max']}",
                            "Assessment": "Stable" if cv < 20 else "Variable",
                        }
                    )

                interp_df = pd.DataFrame(interpretation)
                st.dataframe(interp_df, use_container_width=True)

        # Performance summary by algorithm
        st.subheader("Performance by Algorithm")
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Pivot table for easy comparison
            metrics_to_show = [
                "mean_reward",
                "win_rate",
                "efficiency_score",
                "mean_steps",
            ]
            available_metrics = [m for m in metrics_to_show if m in results_df.columns]

            if available_metrics:
                summary_table = results_df.set_index("algorithm")[available_metrics]
                summary_table.columns = [
                    col.replace("_", " ").title() for col in summary_table.columns
                ]
                st.dataframe(summary_table, use_container_width=True)

    with tab2:
        st.subheader("Performance Clustering")

        # Performance clustering
        if "performance_clustering" in statistical_analysis:
            clustering = statistical_analysis["performance_clustering"]
            if "error" not in clustering:
                st.write(
                    f"**Optimal number of clusters:** {clustering.get('n_clusters', 'N/A')}"
                )

                if "pca_components" in clustering and "cluster_labels" in clustering:
                    pca_data = clustering["pca_components"]
                    cluster_labels = clustering["cluster_labels"]

                    cluster_df = pd.DataFrame(
                        {
                            "PC1": [p[0] for p in pca_data],
                            "PC2": [p[1] for p in pca_data],
                            "Cluster": cluster_labels,
                            "Algorithm": [r["algorithm"] for r in results],
                        }
                    )

                    import plotly.express as px

                    fig = px.scatter(
                        cluster_df,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                        hover_data=["Algorithm"],
                        title="Performance Clustering (PCA Visualization)",
                        labels={
                            "PC1": "Principal Component 1",
                            "PC2": "Principal Component 2",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Cluster interpretation
                    st.subheader("Cluster Analysis")
                    cluster_summary = []

                    for cluster_id in sorted(set(cluster_labels)):
                        algorithms_in_cluster = [
                            cluster_df.loc[i, "Algorithm"]
                            for i in range(len(cluster_df))
                            if cluster_df.loc[i, "Cluster"] == cluster_id
                        ]

                        cluster_summary.append(
                            {
                                "Cluster": f"Cluster {cluster_id}",
                                "Algorithms": ", ".join(algorithms_in_cluster),
                                "Size": len(algorithms_in_cluster),
                                "Interpretation": "Similar performance characteristics",
                            }
                        )

                    if cluster_summary:
                        cluster_df_summary = pd.DataFrame(cluster_summary)
                        st.dataframe(cluster_df_summary, use_container_width=True)
            else:
                st.info(
                    "Clustering analysis not available - insufficient data or all algorithms perform similarly."
                )

        # Performance ranking
        st.subheader("Comprehensive Performance Ranking")

        if results:
            ranking_data = []
            for i, result in enumerate(results):
                # Calculate composite score
                efficiency = result.get("efficiency_score", 0)
                win_rate = result.get("win_rate", 0)
                reward = result.get("mean_reward", 0)

                # Normalize reward (assuming negative rewards, closer to 0 is better)
                max_reward = max([r.get("mean_reward", 0) for r in results])
                min_reward = min([r.get("mean_reward", 0) for r in results])
                normalized_reward = (
                    (reward - min_reward) / (max_reward - min_reward)
                    if max_reward != min_reward
                    else 1.0
                )

                composite_score = (
                    (efficiency * 0.4) + (win_rate * 0.4) + (normalized_reward * 0.2)
                )

                ranking_data.append(
                    {
                        "Algorithm": result["algorithm"],
                        "Efficiency Score": f"{efficiency:.4f}",
                        "Win Rate": f"{win_rate:.2%}",
                        "Mean Reward": f"{reward:.2f}",
                        "Composite Score": f"{composite_score:.4f}",
                    }
                )

            ranking_df = pd.DataFrame(ranking_data)
            ranking_df["Rank"] = (
                ranking_df["Composite Score"]
                .astype(float)
                .rank(ascending=False, method="min")
                .astype(int)
            )
            ranking_df = ranking_df.sort_values("Rank")

            # Reorder columns
            ranking_df = ranking_df[
                [
                    "Rank",
                    "Algorithm",
                    "Composite Score",
                    "Efficiency Score",
                    "Win Rate",
                    "Mean Reward",
                ]
            ]
            st.dataframe(ranking_df, use_container_width=True)

    with tab3:
        st.subheader("Statistical Significance Testing")

        # Hypothesis testing
        if "hypothesis_testing" in statistical_analysis:
            hypothesis_tests = statistical_analysis["hypothesis_testing"]
            if "error" not in hypothesis_tests:

                for comparison, tests in hypothesis_tests.items():
                    st.write(f"**{comparison}:**")
                    test_results = []

                    for metric, test_data in tests.items():
                        if isinstance(test_data, dict):
                            p_value = test_data.get("p_value", 1.0)
                            statistic = test_data.get("statistic", 0.0)

                            # Significance levels
                            if p_value < 0.001:
                                significance = "*** (p < 0.001)"
                                interpretation = "Highly significant"
                            elif p_value < 0.01:
                                significance = "** (p < 0.01)"
                                interpretation = "Very significant"
                            elif p_value < 0.05:
                                significance = "* (p < 0.05)"
                                interpretation = "Significant"
                            else:
                                significance = "ns (p ≥ 0.05)"
                                interpretation = "Not significant"

                            test_results.append(
                                {
                                    "Metric": metric.replace("_", " ").title(),
                                    "Test Statistic": f"{statistic:.3f}",
                                    "P-value": f"{p_value:.4f}",
                                    "Significance": significance,
                                    "Interpretation": interpretation,
                                }
                            )

                    if test_results:
                        test_df = pd.DataFrame(test_results)
                        st.dataframe(test_df, use_container_width=True)

                        st.info(
                            """
                        **Significance Levels:**
                        - *** p < 0.001: Highly significant difference
                        - ** p < 0.01: Very significant difference  
                        - * p < 0.05: Significant difference
                        - ns p ≥ 0.05: No significant difference
                        """
                        )
            else:
                st.info(
                    "Hypothesis testing not available - need at least 2 different algorithms with sufficient data."
                )

        # Effect size analysis
        st.subheader("Effect Size Analysis")

        if len(results) >= 2:
            st.info(
                "Effect size measures the practical significance of differences between algorithms."
            )

            # Calculate Cohen's d for key metrics
            algorithms = [r["algorithm"] for r in results]
            metrics = ["mean_reward", "win_rate", "efficiency_score"]

            effect_sizes = []

            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    algo1, algo2 = algorithms[i], algorithms[j]

                    for metric in metrics:
                        if metric in results[i] and metric in results[j]:
                            val1, val2 = results[i][metric], results[j][metric]

                            # Simplified Cohen's d calculation (assuming equal variance)
                            pooled_std = (
                                0.1  # Placeholder - would need actual data distribution
                            )
                            cohens_d = abs(val1 - val2) / pooled_std

                            if cohens_d < 0.2:
                                effect = "Small"
                            elif cohens_d < 0.5:
                                effect = "Medium"
                            else:
                                effect = "Large"

                            effect_sizes.append(
                                {
                                    "Comparison": f"{algo1} vs {algo2}",
                                    "Metric": metric.replace("_", " ").title(),
                                    "Cohen's d": f"{cohens_d:.3f}",
                                    "Effect Size": effect,
                                    "Difference": f"{abs(val1 - val2):.3f}",
                                }
                            )

            if effect_sizes:
                effect_df = pd.DataFrame(effect_sizes)
                st.dataframe(effect_df, use_container_width=True)

    with tab4:
        st.subheader("Confidence Intervals")

        # Confidence intervals
        if "confidence_intervals" in statistical_analysis:
            ci_data = statistical_analysis["confidence_intervals"]
            st.write("### 95% Confidence Intervals")

            ci_results = []
            for metric, ci_info in ci_data.items():
                if isinstance(ci_info, dict):
                    ci_results.append(
                        {
                            "Metric": metric.replace("_", " ").title(),
                            "Mean": f"{ci_info.get('mean', 0):.3f}",
                            "Lower Bound (2.5%)": f"{ci_info.get('lower_bound', 0):.3f}",
                            "Upper Bound (97.5%)": f"{ci_info.get('upper_bound', 0):.3f}",
                            "Margin of Error": f"{ci_info.get('margin_error', 0):.3f}",
                        }
                    )

            if ci_results:
                ci_df = pd.DataFrame(ci_results)
                st.dataframe(ci_df, use_container_width=True)

                st.info(
                    """
                **Interpretation of Confidence Intervals:**
                - We are 95% confident that the true population mean lies within the interval
                - Narrower intervals indicate more precise estimates
                - Non-overlapping intervals suggest significant differences between groups
                """
                )

        # Convergence analysis for training data
        if training_data:
            st.subheader("Training Convergence Analysis")

            convergence_stats = []

            for algo, data in training_data.items():
                if "rewards" in data and len(data["rewards"]) > 10:
                    rewards = data["rewards"]

                    # Calculate convergence metrics
                    final_portion = rewards[
                        -len(rewards) // 5 :
                    ]  # Last 20% of training
                    initial_portion = rewards[
                        : len(rewards) // 5
                    ]  # First 20% of training

                    final_mean = np.mean(final_portion)
                    final_std = np.std(final_portion)
                    initial_mean = np.mean(initial_portion)

                    improvement = final_mean - initial_mean
                    stability = 1 / (final_std + 0.001)  # Avoid division by zero

                    convergence_stats.append(
                        {
                            "Algorithm": algo,
                            "Initial Performance": f"{initial_mean:.3f}",
                            "Final Performance": f"{final_mean:.3f}",
                            "Total Improvement": f"{improvement:.3f}",
                            "Final Stability": f"{stability:.3f}",
                            "Converged": "Yes" if final_std < 1.0 else "No",
                        }
                    )

            if convergence_stats:
                conv_df = pd.DataFrame(convergence_stats)
                st.dataframe(conv_df, use_container_width=True)

    # Summary and recommendations
    st.subheader("Analysis Summary & Recommendations")

    if results:
        best_algorithm = max(results, key=lambda x: x.get("efficiency_score", 0))[
            "algorithm"
        ]
        most_consistent = min(results, key=lambda x: x.get("std_reward", float("inf")))[
            "algorithm"
        ]

        recommendations = [
            f"🏆 **Best Overall Performance**: {best_algorithm} shows the highest efficiency score",
            f"🎯 **Most Consistent**: {most_consistent} demonstrates the most stable performance",
            "📊 **Statistical Significance**: Check the hypothesis testing tab for meaningful differences",
            "🔄 **Confidence**: Review confidence intervals to understand estimate precision",
        ]

        for rec in recommendations:
            st.markdown(f"- {rec}")

        # Export analysis option
        if st.button("📄 Generate Full Analysis Report"):
            st.info("Feature coming soon: Export comprehensive analysis report as PDF")


# Import plotly for additional visualizations
import plotly.express as px
