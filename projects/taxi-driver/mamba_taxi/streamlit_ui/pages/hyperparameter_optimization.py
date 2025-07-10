"""
Hyperparameter Optimization Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from ..config import ALGORITHMS
from ..components.visualizations import VisualizationComponents
from ..utils.training_manager import TrainingManager
from ..utils.data_manager import DataManager


def render_hyperparameter_optimization_page():
    """Render the hyperparameter optimization page"""
    st.header("Hyperparameter Optimization")

    st.markdown(
        """
    Automatically find the best hyperparameters for your chosen algorithm using different optimization strategies.
    """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Optimization Configuration")

        algorithm = st.selectbox(
            "Algorithm to Optimize",
            [algo for algo in ALGORITHMS if algo != "BruteForce"],
            key="opt_algo",
        )

        optimization_method = st.selectbox(
            "Optimization Method",
            ["Random Search", "Grid Search"],
            help="Random Search: Test random parameter combinations\nGrid Search: Test all combinations in a grid",
        )

        st.subheader("Search Parameters")
        n_configs = st.number_input(
            "Number of Configurations",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of parameter combinations to test",
        )

        train_episodes = st.number_input(
            "Training Episodes per Config",
            min_value=100,
            max_value=1000,
            value=300,
            key="opt_train",
        )

        test_episodes = st.number_input(
            "Test Episodes per Config",
            min_value=20,
            max_value=100,
            value=50,
            key="opt_test",
        )

        st.subheader("Parameter Ranges")
        alpha_range = st.slider(
            "Learning Rate Range",
            0.01,
            0.5,
            (0.05, 0.3),
            key="opt_alpha_range",
            help="Range of learning rates to explore",
        )

        gamma_range = st.slider(
            "Discount Factor Range",
            0.9,
            0.999,
            (0.95, 0.99),
            key="opt_gamma_range",
            help="Range of discount factors to explore",
        )

        # Advanced options
        with st.expander("Advanced Options"):
            early_stopping = st.checkbox(
                "Early Stopping", value=False, help="Stop if no improvement found"
            )
            if early_stopping:
                patience = st.number_input(
                    "Patience", min_value=3, max_value=10, value=5
                )

            save_all_configs = st.checkbox("Save All Configurations", value=True)

    with col2:
        if st.button("🎯 Start Optimization", type="primary"):
            st.subheader("Optimization Progress")

            data_manager = DataManager()
            training_manager = TrainingManager(st.session_state.env)

            try:
                # Prepare parameter ranges
                param_ranges = {"alpha": alpha_range, "gamma": gamma_range}

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()

                # Best configuration tracking
                best_container = st.empty()

                # Run optimization
                optimization_result = training_manager.optimize_hyperparameters(
                    algorithm,
                    param_ranges,
                    n_configs,
                    train_episodes,
                    test_episodes,
                    method=optimization_method.lower().replace(" ", ""),
                )

                results = optimization_result["results"]
                best_config = optimization_result["best_config"]
                best_score = optimization_result["best_score"]

                status_text.text("✅ Optimization completed!")
                progress_bar.progress(1.0)

                # Display final results
                st.subheader("Optimization Results")

                if best_config:
                    VisualizationComponents.display_success_message(
                        f"""
                        <h4>🏆 Best Configuration Found:</h4>
                        <p><strong>Learning Rate (α):</strong> {best_config['alpha']:.4f}</p>
                        <p><strong>Discount Factor (γ):</strong> {best_config['gamma']:.4f}</p>
                        <p><strong>Efficiency Score:</strong> {best_score:.4f}</p>
                        """
                    )

                    # Configuration comparison with default
                    default_config = {
                        "alpha": 0.15,
                        "gamma": 0.99,
                        "efficiency_score": 0.0,  # Placeholder
                    }

                    comparison_data = {
                        "Configuration": ["Default", "Optimized"],
                        "Learning Rate": [
                            default_config["alpha"],
                            best_config["alpha"],
                        ],
                        "Discount Factor": [
                            default_config["gamma"],
                            best_config["gamma"],
                        ],
                        "Efficiency Score": [
                            default_config["efficiency_score"],
                            best_score,
                        ],
                    }

                    st.subheader("Configuration Comparison")
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                # Results analysis
                if results:
                    results_df = pd.DataFrame(results)

                    # Optimization progress chart
                    st.subheader("Optimization Progress")
                    if len(results) > 3:
                        # Parameter space visualization
                        param_fig = VisualizationComponents.create_parameter_space_plot(
                            results_df
                        )
                        if param_fig:
                            st.plotly_chart(param_fig, use_container_width=True)

                        # Progress over time
                        results_df["config_order"] = range(len(results_df))
                        progress_fig = px.line(
                            results_df,
                            x="config_order",
                            y="efficiency_score",
                            title="Optimization Progress Over Configurations",
                            labels={
                                "config_order": "Configuration Number",
                                "efficiency_score": "Efficiency Score",
                            },
                        )
                        st.plotly_chart(progress_fig, use_container_width=True)

                    # Parameter correlation analysis
                    st.subheader("Parameter Analysis")
                    corr_fig = (
                        VisualizationComponents.create_parameter_correlation_heatmap(
                            results_df
                        )
                    )
                    if corr_fig:
                        st.pyplot(corr_fig)

                    # Top configurations table
                    st.subheader("Top 10 Configurations")
                    top_configs = results_df.nlargest(
                        min(10, len(results_df)), "efficiency_score"
                    )
                    display_cols = [
                        "alpha",
                        "gamma",
                        "efficiency_score",
                        "win_rate",
                        "mean_reward",
                    ]
                    available_cols = [
                        col for col in display_cols if col in top_configs.columns
                    ]
                    st.dataframe(top_configs[available_cols], use_container_width=True)

                    # Statistical summary
                    st.subheader("Statistical Summary")
                    summary_stats = {
                        "Metric": [
                            "Best Score",
                            "Worst Score",
                            "Mean Score",
                            "Std Score",
                        ],
                        "Value": [
                            f"{results_df['efficiency_score'].max():.4f}",
                            f"{results_df['efficiency_score'].min():.4f}",
                            f"{results_df['efficiency_score'].mean():.4f}",
                            f"{results_df['efficiency_score'].std():.4f}",
                        ],
                    }
                    summary_df = pd.DataFrame(summary_stats)
                    st.dataframe(summary_df, use_container_width=True)

                # Save optimization results
                if save_all_configs:
                    json_file, csv_file = data_manager.save_results(
                        results, {}, f"optimization_{algorithm.lower()}"
                    )

                    VisualizationComponents.display_success_message(
                        f"✅ Optimization results saved to {json_file}"
                    )

                    # Download buttons
                    col_download1, col_download2 = st.columns(2)
                    with col_download1:
                        if csv_file:
                            with open(csv_file, "r") as f:
                                st.download_button(
                                    "📥 Download CSV",
                                    f.read(),
                                    file_name=f"optimization_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )

                    with col_download2:
                        with open(json_file, "r") as f:
                            st.download_button(
                                "📥 Download JSON",
                                f.read(),
                                file_name=f"optimization_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                            )

                # Option to test best configuration
                st.subheader("Test Best Configuration")
                if st.button("🧪 Test Best Configuration", key="test_best"):
                    st.info("Testing best configuration with more episodes...")

                    # Test with more episodes
                    extended_result = training_manager.train_single_agent(
                        algorithm, best_config, train_episodes * 2, test_episodes * 2
                    )

                    st.subheader("Extended Test Results")
                    VisualizationComponents.display_metrics(
                        extended_result["test_result"]
                    )

            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Optimization failed!")
                VisualizationComponents.display_error_message(
                    f"Optimization error: {str(e)}"
                )
                st.error(f"Optimization failed: {e}")

        else:
            st.info(
                "👆 Configure your optimization settings and click 'Start Optimization' to begin!"
            )

            # Show optimization tips
            st.subheader("Optimization Tips")
            tips = [
                "🎯 **Start Small**: Begin with 10-20 configurations to get a feel for the parameter space",
                "📊 **Random vs Grid**: Random search often finds good solutions faster than grid search",
                "⚡ **Episode Count**: Use fewer episodes per config for faster exploration, then test best configs longer",
                "📈 **Parameter Ranges**: Start with wider ranges, then narrow down based on results",
                "🔄 **Iterative Process**: Run multiple optimization rounds, refining ranges each time",
            ]

            for tip in tips:
                st.markdown(f"- {tip}")


# Import plotly for the progress visualization
import plotly.express as px
