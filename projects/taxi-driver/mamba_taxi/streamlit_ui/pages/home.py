"""
Home Page Component
"""

import streamlit as st
import pandas as pd


def render_home_page():
    """Render the home page"""
    st.markdown(
        """
    ## Welcome to the Taxi Driver RL Interface!
    
    This interactive interface allows you to:
    - Test individual algorithms with custom parameters
    - Compare multiple algorithms side by side
    - Optimize hyperparameters automatically
    - Perform advanced statistical analysis
    - Save and manage your results
    
    ### Quick Start:
    1. Go to **Single Algorithm Testing** to test individual algorithms
    2. Use **Algorithm Comparison** to compare multiple algorithms
    3. Try **Hyperparameter Optimization** for automated tuning
    4. Check **Advanced Analysis** for detailed statistics
    
    ### Available Algorithms:
    - **BruteForce**: Random baseline (no training required)
    - **Q-Learning**: Off-policy temporal difference learning
    - **SARSA**: On-policy temporal difference learning  
    - **DQN**: Deep Q-Network with experience replay
    
    ### Features:
    - 🚀 Real-time training progress visualization
    - 📊 Interactive comparison charts and radar plots
    - 🎯 Automated hyperparameter optimization
    - 📈 Advanced statistical analysis with confidence intervals
    - 💾 Export results in CSV/JSON formats
    - 🔄 Session state management for seamless workflow
    """
    )

    # Quick stats if we have results
    if st.session_state.results_history:
        st.subheader("Recent Results Summary")
        df = pd.DataFrame(st.session_state.results_history[-5:])  # Last 5 results

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Experiments", len(st.session_state.results_history))

        with col2:
            unique_algos = df["algorithm"].nunique()
            st.metric("Algorithms Tested", unique_algos)

        with col3:
            best_algo = (
                df.loc[df["efficiency_score"].idxmax(), "algorithm"]
                if "efficiency_score" in df.columns
                else "N/A"
            )
            st.metric("Best Algorithm", best_algo)

        with col4:
            avg_win_rate = df["win_rate"].mean() if "win_rate" in df.columns else 0
            st.metric("Avg Win Rate", f"{avg_win_rate:.1%}")

        # Recent results table
        st.subheader("Recent Results")
        display_cols = ["algorithm", "mean_reward", "win_rate", "efficiency_score"]
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)

        # Quick visualization
        if len(df) > 1:
            st.subheader("Performance Overview")
            chart_data = df.set_index("algorithm")["efficiency_score"]
            st.bar_chart(chart_data)

    else:
        st.info("👋 Welcome! Start by testing an algorithm to see your results here.")

        # Show getting started tips
        st.subheader("Getting Started Tips")

        tips = [
            "🔧 **Start Simple**: Begin with the Single Algorithm Testing page to understand how each algorithm performs",
            "📊 **Compare**: Use Algorithm Comparison to see which algorithm works best for your needs",
            "🎯 **Optimize**: Try Hyperparameter Optimization to automatically find the best settings",
            "📈 **Analyze**: Use Advanced Analysis for detailed statistical insights",
            "💾 **Save**: Don't forget to save your results for future reference",
        ]

        for tip in tips:
            st.markdown(f"- {tip}")
