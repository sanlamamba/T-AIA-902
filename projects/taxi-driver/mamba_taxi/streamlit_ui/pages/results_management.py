"""
Results Management Page
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from ..components.visualizations import VisualizationComponents
from ..utils.data_manager import DataManager


def render_results_management_page():
    """Render the results management page"""
    st.header("Results Management")

    data_manager = DataManager()

    # Results history section
    if st.session_state.results_history:
        st.subheader("Results History")

        # Create summary table
        history_df = pd.DataFrame(st.session_state.results_history)

        # Add experiment number and timestamp if not present
        if "experiment_id" not in history_df.columns:
            history_df["experiment_id"] = range(1, len(history_df) + 1)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Experiments", len(st.session_state.results_history))

        with col2:
            unique_algos = history_df["algorithm"].nunique()
            st.metric("Unique Algorithms", unique_algos)

        with col3:
            if "efficiency_score" in history_df.columns:
                best_score = history_df["efficiency_score"].max()
                st.metric("Best Efficiency Score", f"{best_score:.4f}")
            else:
                st.metric("Best Efficiency Score", "N/A")

        with col4:
            avg_win_rate = (
                history_df["win_rate"].mean() if "win_rate" in history_df.columns else 0
            )
            st.metric("Average Win Rate", f"{avg_win_rate:.1%}")

        # Filtering options
        st.subheader("Filter Results")

        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            selected_algorithms = st.multiselect(
                "Filter by Algorithm",
                options=history_df["algorithm"].unique(),
                default=history_df["algorithm"].unique(),
            )

        with col_filter2:
            if "efficiency_score" in history_df.columns:
                min_efficiency = st.slider(
                    "Minimum Efficiency Score",
                    min_value=float(history_df["efficiency_score"].min()),
                    max_value=float(history_df["efficiency_score"].max()),
                    value=float(history_df["efficiency_score"].min()),
                )
            else:
                min_efficiency = 0

        with col_filter3:
            show_recent = st.selectbox(
                "Show Results", ["All", "Last 10", "Last 20", "Last 50"], index=0
            )

        # Apply filters
        filtered_df = history_df[history_df["algorithm"].isin(selected_algorithms)]

        if "efficiency_score" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["efficiency_score"] >= min_efficiency]

        if show_recent != "All":
            n_recent = int(show_recent.split()[-1])
            filtered_df = filtered_df.tail(n_recent)

        # Display filtered results
        st.subheader("Filtered Results")

        # Select columns to display
        all_columns = filtered_df.columns.tolist()
        default_columns = [
            "experiment_id",
            "algorithm",
            "mean_reward",
            "win_rate",
            "efficiency_score",
            "mean_steps",
        ]
        available_default_columns = [
            col for col in default_columns if col in all_columns
        ]

        selected_columns = st.multiselect(
            "Select Columns to Display",
            options=all_columns,
            default=available_default_columns,
        )

        if selected_columns:
            display_df = filtered_df[selected_columns]
            st.dataframe(display_df, use_container_width=True)

            # Quick statistics for filtered data
            if len(filtered_df) > 0:
                st.subheader("Statistics for Filtered Data")

                stats_col1, stats_col2 = st.columns(2)

                with stats_col1:
                    st.write("**Algorithm Distribution:**")
                    algo_counts = filtered_df["algorithm"].value_counts()
                    st.bar_chart(algo_counts)

                with stats_col2:
                    if "efficiency_score" in filtered_df.columns:
                        st.write("**Efficiency Score Distribution:**")
                        st.line_chart(
                            filtered_df["efficiency_score"].reset_index(drop=True)
                        )

        # Download options
        st.subheader("Download Results")

        download_col1, download_col2, download_col3 = st.columns(3)

        with download_col1:
            if st.button("📥 Download Filtered as CSV"):
                if selected_columns and len(filtered_df) > 0:
                    csv = filtered_df[selected_columns].to_csv(index=False)
                    st.download_button(
                        label="Download CSV file",
                        data=csv,
                        file_name=f"taxi_rl_filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No data to download")

        with download_col2:
            if st.button("📥 Download All as JSON"):
                json_data = json.dumps(
                    st.session_state.results_history, indent=2, default=str
                )
                st.download_button(
                    label="Download JSON file",
                    data=json_data,
                    file_name=f"taxi_rl_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

        with download_col3:
            if st.button("📊 Generate Summary Report"):
                summary_report = data_manager.create_summary_report(
                    st.session_state.results_history
                )

                if summary_report:
                    st.subheader("Summary Report")

                    report_text = f"""
# Taxi Driver RL Experiments Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Experiments**: {summary_report['total_experiments']}
- **Algorithms Tested**: {', '.join(summary_report['algorithms_tested'])}
- **Best Algorithm**: {summary_report.get('best_algorithm', 'N/A')}

## Average Performance
- **Mean Reward**: {summary_report['average_performance']['mean_reward']:.3f}
- **Mean Steps**: {summary_report['average_performance']['mean_steps']:.1f}
- **Win Rate**: {summary_report['average_performance']['win_rate']:.2%}

## Performance Ranges
- **Reward Range**: {summary_report['performance_ranges']['reward_range'][0]:.3f} to {summary_report['performance_ranges']['reward_range'][1]:.3f}
- **Steps Range**: {summary_report['performance_ranges']['steps_range'][0]:.1f} to {summary_report['performance_ranges']['steps_range'][1]:.1f}
- **Win Rate Range**: {summary_report['performance_ranges']['win_rate_range'][0]:.2%} to {summary_report['performance_ranges']['win_rate_range'][1]:.2%}
                    """

                    st.markdown(report_text)

                    st.download_button(
                        label="📄 Download Report",
                        data=report_text,
                        file_name=f"taxi_rl_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                    )

        # Data management actions
        st.subheader("Data Management")

        manage_col1, manage_col2, manage_col3 = st.columns(3)

        with manage_col1:
            if st.button("🗑️ Clear All History", type="secondary"):
                if st.button("⚠️ Confirm Clear All", type="secondary"):
                    st.session_state.results_history = []
                    st.session_state.training_history = []
                    st.session_state.comparison_data = {}
                    VisualizationComponents.display_success_message(
                        "✅ All history cleared!"
                    )
                    st.experimental_rerun()

        with manage_col2:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    if key not in ["env", "n_actions", "n_states"]:  # Keep environment
                        del st.session_state[key]
                VisualizationComponents.display_success_message("✅ Session reset!")
                st.experimental_rerun()

        with manage_col3:
            if st.button("💾 Auto-Save Current Session"):
                # Save current session data
                session_data = {
                    "results_history": st.session_state.results_history,
                    "training_history": st.session_state.training_history,
                    "comparison_data": st.session_state.comparison_data,
                    "timestamp": datetime.now().isoformat(),
                }

                filename = (
                    f"session_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                json_str = json.dumps(session_data, indent=2, default=str)

                st.download_button(
                    label="💾 Download Session Backup",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                )

    else:
        st.info("No results available yet. Run some experiments to see results here!")

        st.subheader("Getting Started")
        getting_started_tips = [
            "🔬 **Run Experiments**: Start with Single Algorithm Testing to generate your first results",
            "📊 **Compare Algorithms**: Use Algorithm Comparison to evaluate multiple approaches",
            "🎯 **Optimize Parameters**: Try Hyperparameter Optimization for best performance",
            "📈 **Analyze Results**: Use Advanced Analysis for detailed insights",
            "💾 **Save Your Work**: Results are automatically tracked in your session",
        ]

        for tip in getting_started_tips:
            st.markdown(f"- {tip}")

    # Load saved results section
    st.subheader("Load Previous Results")

    uploaded_file = st.file_uploader(
        "Choose a results file",
        type=["json", "csv"],
        help="Upload a previously saved results file to continue your analysis",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".json"):
                data = json.load(uploaded_file)

                # Handle different JSON structures
                if "results_history" in data:
                    # Session backup format
                    st.session_state.results_history.extend(data["results_history"])
                    if "training_history" in data:
                        st.session_state.training_history.extend(
                            data["training_history"]
                        )
                    if "comparison_data" in data:
                        st.session_state.comparison_data = data["comparison_data"]
                elif "results" in data:
                    # Standard results format
                    st.session_state.results_history.extend(data["results"])
                elif isinstance(data, list):
                    # Direct list of results
                    st.session_state.results_history.extend(data)
                else:
                    st.error("Unrecognized JSON format")

            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.session_state.results_history.extend(df.to_dict("records"))

            VisualizationComponents.display_success_message(
                "✅ Results loaded successfully!"
            )
            st.experimental_rerun()

        except Exception as e:
            VisualizationComponents.display_error_message(f"Error loading file: {e}")

    # Show saved files in the results directory
    saved_files = data_manager.get_saved_files()
    if saved_files:
        st.subheader("Previously Saved Files")

        st.write("📁 **Files in results directory:**")
        for file_path in saved_files[:10]:  # Show last 10 files
            file_name = file_path.split("/")[-1]
            file_size = (
                f"{os.path.getsize(file_path) / 1024:.1f} KB"
                if os.path.exists(file_path)
                else "Unknown"
            )
            file_date = (
                datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
                    "%Y-%m-%d %H:%M"
                )
                if os.path.exists(file_path)
                else "Unknown"
            )

            st.write(f"- **{file_name}** ({file_size}) - {file_date}")

        if len(saved_files) > 10:
            st.write(f"... and {len(saved_files) - 10} more files")


# Import os for file operations
import os
