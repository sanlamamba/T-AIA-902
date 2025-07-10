"""
Algorithm Comparison Page
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from ..config import ALGORITHMS
from ..components.visualizations import VisualizationComponents
from ..utils.training_manager import TrainingManager
from ..utils.data_manager import DataManager


def render_algorithm_comparison_page():
    """Render the algorithm comparison page"""
    st.header("Algorithm Comparison")

    # Algorithm selection for comparison
    st.subheader("Select Algorithms to Compare")
    algorithms_to_compare = st.multiselect(
        "Choose algorithms",
        ALGORITHMS,
        default=["BruteForce", "Q-Learning"],
        help="Select at least 2 algorithms to compare their performance",
    )

    if len(algorithms_to_compare) < 2:
        VisualizationComponents.display_warning_message(
            "⚠️ Please select at least 2 algorithms to compare."
        )
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Comparison Parameters")
        train_episodes = st.number_input(
            "Training Episodes",
            min_value=100,
            max_value=3000,
            value=1000,
            key="comp_train",
            help="Number of episodes for training each algorithm",
        )
        test_episodes = st.number_input(
            "Test Episodes",
            min_value=50,
            max_value=500,
            value=100,
            key="comp_test",
            help="Number of episodes for evaluating each algorithm",
        )

        # Global parameters for all algorithms
        st.subheader("Algorithm Parameters")
        alpha = st.slider(
            "Learning Rate (α)",
            0.01,
            1.0,
            0.15,
            0.01,
            key="comp_alpha",
            help="Learning rate for all algorithms",
        )
        gamma = st.slider(
            "Discount Factor (γ)",
            0.8,
            0.999,
            0.99,
            0.001,
            key="comp_gamma",
            help="Discount factor for all algorithms",
        )

    with col2:
        st.subheader("Comparison Options")
        include_stats = st.checkbox("Include Statistical Analysis", value=True)
        show_training_curves = st.checkbox("Show Training Curves", value=True)
        save_results_auto = st.checkbox("Auto-save Results", value=True)
        show_radar_chart = st.checkbox("Show Radar Chart", value=True)

    # Run comparison button
    if st.button("🔄 Run Comparison", type="primary"):
        data_manager = DataManager()
        training_manager = TrainingManager(st.session_state.env)

        results = []
        training_data = {}

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Results container for real-time updates
        results_container = st.empty()

        try:
            for i, algorithm in enumerate(algorithms_to_compare):
                status_text.text(
                    f"🏃 Training {algorithm}... ({i+1}/{len(algorithms_to_compare)})"
                )

                # Prepare parameters
                params = {
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": 1.0,
                    "epsilon_decay": 0.995,
                    "epsilon_min": 0.01,
                }
                if algorithm == "DQN":
                    params.update({"memory_size": 10000, "batch_size": 32})
                elif algorithm == "BruteForce":
                    params = {}

                # Train and evaluate
                result = training_manager.train_single_agent(
                    algorithm, params, train_episodes, test_episodes
                )

                test_result = result["test_result"]
                train_result = result["train_result"]

                results.append(test_result)
                if algorithm != "BruteForce":
                    training_data[algorithm] = train_result

                progress_bar.progress((i + 1) / len(algorithms_to_compare))

                # Update results display in real-time
                with results_container.container():
                    st.subheader("Current Results")
                    cols = st.columns(len(results))
                    for j, res in enumerate(results):
                        VisualizationComponents.display_metrics_in_column(res, cols[j])

            status_text.text("✅ All algorithms completed!")

            # Store in session state
            st.session_state.comparison_data = {
                "results": results,
                "training_data": training_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Display comprehensive results
            st.subheader("Comparison Results")

            # Performance metrics grid
            st.subheader("Performance Overview")
            cols = st.columns(len(results))
            for i, result in enumerate(results):
                VisualizationComponents.display_metrics_in_column(result, cols[i])

            # Comparison charts
            st.subheader("Visual Comparison")

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(
                ["📊 Bar Charts", "🎯 Radar Chart", "📈 Training Curves"]
            )

            with tab1:
                # Bar chart comparison
                comparison_fig = VisualizationComponents.create_comparison_chart(
                    results
                )
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)

            with tab2:
                # Radar chart
                if show_radar_chart:
                    radar_fig = VisualizationComponents.create_radar_chart(results)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.info("Radar chart disabled. Enable it in the options above.")

            with tab3:
                # Training curves
                if show_training_curves and training_data:
                    training_fig = (
                        VisualizationComponents.create_training_progress_chart(
                            training_data
                        )
                    )
                    if training_fig:
                        st.plotly_chart(training_fig, use_container_width=True)
                else:
                    st.info("No training data available or training curves disabled.")

            # Statistical analysis
            if include_stats:
                st.subheader("Statistical Analysis")

                # Create summary table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Performance ranking
                st.subheader("Performance Ranking")
                ranking_data = []
                for result in results:
                    ranking_data.append(
                        {
                            "Algorithm": result["algorithm"],
                            "Efficiency Score": result.get("efficiency_score", 0),
                            "Mean Reward": result["mean_reward"],
                            "Win Rate": result["win_rate"],
                            "Mean Steps": result["mean_steps"],
                        }
                    )

                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values("Efficiency Score", ascending=False)
                ranking_df["Rank"] = range(1, len(ranking_df) + 1)

                # Reorder columns
                ranking_df = ranking_df[
                    [
                        "Rank",
                        "Algorithm",
                        "Efficiency Score",
                        "Mean Reward",
                        "Win Rate",
                        "Mean Steps",
                    ]
                ]
                st.dataframe(ranking_df, use_container_width=True)

                # Performance improvement over baseline
                if any(r["algorithm"] == "BruteForce" for r in results):
                    st.subheader("Improvement Over BruteForce Baseline")
                    bf_result = next(
                        r for r in results if r["algorithm"] == "BruteForce"
                    )

                    improvement_data = []
                    for result in results:
                        if result["algorithm"] != "BruteForce":
                            reward_improvement = (
                                (
                                    (result["mean_reward"] - bf_result["mean_reward"])
                                    / abs(bf_result["mean_reward"])
                                )
                                * 100
                                if bf_result["mean_reward"] != 0
                                else 0
                            )

                            step_improvement = (
                                (
                                    (bf_result["mean_steps"] - result["mean_steps"])
                                    / bf_result["mean_steps"]
                                )
                                * 100
                                if bf_result["mean_steps"] != 0
                                else 0
                            )

                            efficiency_gain = result.get("efficiency_score", 0) / max(
                                bf_result.get("efficiency_score", 0.001), 0.001
                            )

                            improvement_data.append(
                                {
                                    "Algorithm": result["algorithm"],
                                    "Reward Improvement (%)": round(
                                        reward_improvement, 1
                                    ),
                                    "Step Improvement (%)": round(step_improvement, 1),
                                    "Efficiency Gain (x)": round(efficiency_gain, 2),
                                }
                            )

                    if improvement_data:
                        improvement_df = pd.DataFrame(improvement_data)
                        st.dataframe(improvement_df, use_container_width=True)

            # Save results
            if save_results_auto:
                json_file, csv_file = data_manager.save_results(
                    results, training_data, "comparison"
                )
                VisualizationComponents.display_success_message(
                    f"✅ Results automatically saved to {json_file}"
                )

                # Download buttons
                col_download1, col_download2 = st.columns(2)
                with col_download1:
                    if csv_file:
                        with open(csv_file, "r") as f:
                            st.download_button(
                                "📥 Download CSV",
                                f.read(),
                                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

                with col_download2:
                    with open(json_file, "r") as f:
                        st.download_button(
                            "📥 Download JSON",
                            f.read(),
                            file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )

        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Error occurred!")
            VisualizationComponents.display_error_message(
                f"An error occurred during comparison: {str(e)}"
            )
            st.error(f"Comparison failed: {e}")

    # Show previous comparison if available
    elif st.session_state.comparison_data:
        st.subheader("Previous Comparison Results")

        previous_results = st.session_state.comparison_data["results"]
        previous_training = st.session_state.comparison_data["training_data"]

        # Quick overview
        st.write(f"**Timestamp:** {st.session_state.comparison_data['timestamp']}")
        st.write(
            f"**Algorithms Compared:** {', '.join([r['algorithm'] for r in previous_results])}"
        )

        # Show condensed results
        cols = st.columns(len(previous_results))
        for i, result in enumerate(previous_results):
            VisualizationComponents.display_metrics_in_column(result, cols[i])

        # Option to clear previous results
        if st.button("🗑️ Clear Previous Results"):
            st.session_state.comparison_data = {}
            st.experimental_rerun()
