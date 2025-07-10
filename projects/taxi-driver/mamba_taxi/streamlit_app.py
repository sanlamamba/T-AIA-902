import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import time
import os
import json
from datetime import datetime
from io import BytesIO
import base64

# Import project modules
from agents import BruteForceAgent, QLearningAgent, SARSAAgent, DQNAgent
from trainer import train_agent, evaluate_agent
from advanced_config import (
    AdvancedConfig,
    ConfigurationManager,
    ExplorationStrategy,
    RewardShaping,
)
from enhanced_statistics import AdvancedStatisticalAnalyzer
import gymnasium as gym

# Set page config
st.set_page_config(
    page_title="🚕 Taxi Driver RL Interface",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitTaxiDriver:
    def __init__(self):
        if "env" not in st.session_state:
            st.session_state.env = gym.make("Taxi-v3", render_mode=None)
            st.session_state.n_actions = st.session_state.env.action_space.n
            st.session_state.n_states = st.session_state.env.observation_space.n

        self.config_manager = ConfigurationManager()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()

        # Initialize session state
        if "results_history" not in st.session_state:
            st.session_state.results_history = []
        if "training_history" not in st.session_state:
            st.session_state.training_history = []
        if "comparison_data" not in st.session_state:
            st.session_state.comparison_data = {}

    def create_agent(self, algorithm, **params):
        """Create agent based on algorithm and parameters"""
        if algorithm == "BruteForce":
            return BruteForceAgent(
                st.session_state.n_actions, st.session_state.n_states
            )
        elif algorithm == "Q-Learning":
            return QLearningAgent(
                st.session_state.n_actions, st.session_state.n_states, **params
            )
        elif algorithm == "SARSA":
            return SARSAAgent(
                st.session_state.n_actions, st.session_state.n_states, **params
            )
        elif algorithm == "DQN":
            return DQNAgent(
                st.session_state.n_actions, st.session_state.n_states, **params
            )

    def display_metrics(self, result):
        """Display performance metrics in a formatted way"""
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>{result['algorithm']}</h4>
            <p><strong>Mean Reward:</strong> {result['mean_reward']:.2f} ± {result.get('std_reward', 0):.2f}</p>
            <p><strong>Mean Steps:</strong> {result['mean_steps']:.1f} ± {result.get('std_steps', 0):.1f}</p>
            <p><strong>Win Rate:</strong> {result['win_rate']:.2%}</p>
            <p><strong>Efficiency Score:</strong> {result.get('efficiency_score', 0):.4f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def display_metrics_in_column(self, result, col):
        """Display performance metrics in a specific column"""
        with col:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>{result['algorithm']}</h4>
                <p><strong>Mean Reward:</strong> {result['mean_reward']:.2f} ± {result.get('std_reward', 0):.2f}</p>
                <p><strong>Mean Steps:</strong> {result['mean_steps']:.1f} ± {result.get('std_steps', 0):.1f}</p>
                <p><strong>Win Rate:</strong> {result['win_rate']:.2%}</p>
                <p><strong>Efficiency Score:</strong> {result.get('efficiency_score', 0):.4f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def create_comparison_chart(self, results):
        """Create interactive comparison charts"""
        if not results:
            return None

        df = pd.DataFrame(results)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Mean Reward Comparison",
                "Win Rate Comparison",
                "Efficiency Score Comparison",
                "Mean Steps Comparison",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        algorithms = df["algorithm"].values
        colors = px.colors.qualitative.Set1[: len(algorithms)]

        # Mean Reward
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=df["mean_reward"],
                name="Mean Reward",
                marker_color=colors,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Win Rate
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=df["win_rate"],
                name="Win Rate",
                marker_color=colors,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Efficiency Score
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=df.get("efficiency_score", [0] * len(df)),
                name="Efficiency",
                marker_color=colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Mean Steps
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=df["mean_steps"],
                name="Mean Steps",
                marker_color=colors,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(height=600, title_text="Algorithm Performance Comparison")
        return fig

    def create_training_progress_chart(self, training_data):
        """Create training progress visualization"""
        if not training_data:
            return None

        fig = go.Figure()

        for algo, data in training_data.items():
            if "rewards" in data:
                episodes = list(range(len(data["rewards"])))
                # Smooth the data
                window = min(50, len(data["rewards"]) // 10)
                if window > 1:
                    smoothed = np.convolve(
                        data["rewards"], np.ones(window) / window, mode="valid"
                    )
                    episodes_smooth = episodes[: len(smoothed)]
                else:
                    smoothed = data["rewards"]
                    episodes_smooth = episodes

                fig.add_trace(
                    go.Scatter(
                        x=episodes_smooth,
                        y=smoothed,
                        mode="lines",
                        name=f"{algo} (smoothed)",
                        line=dict(width=2),
                    )
                )

                # Add raw data with transparency
                fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=data["rewards"],
                        mode="lines",
                        name=f"{algo} (raw)",
                        line=dict(width=1),
                        opacity=0.3,
                    )
                )

        fig.update_layout(
            title="Training Progress Over Episodes",
            xaxis_title="Episode",
            yaxis_title="Reward",
            height=500,
        )
        return fig

    def create_radar_chart(self, results):
        """Create radar chart for multi-dimensional comparison"""
        if not results:
            return None

        # Normalize metrics to 0-1 scale
        metrics = ["mean_reward", "win_rate", "efficiency_score"]
        df = pd.DataFrame(results)

        # Normalize mean_reward and efficiency_score
        if "mean_reward" in df.columns:
            min_reward = df["mean_reward"].min()
            max_reward = df["mean_reward"].max()
            if max_reward > min_reward:
                df["mean_reward_norm"] = (df["mean_reward"] - min_reward) / (
                    max_reward - min_reward
                )
            else:
                df["mean_reward_norm"] = 1.0

        if "efficiency_score" in df.columns:
            max_eff = df["efficiency_score"].max()
            if max_eff > 0:
                df["efficiency_score_norm"] = df["efficiency_score"] / max_eff
            else:
                df["efficiency_score_norm"] = 0.0

        fig = go.Figure()

        for idx, row in df.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=[
                        row.get("mean_reward_norm", 0),
                        row.get("win_rate", 0),
                        row.get("efficiency_score_norm", 0),
                        row.get("mean_reward_norm", 0),
                    ],  # Close the shape
                    theta=["Reward", "Win Rate", "Efficiency", "Reward"],
                    fill="toself",
                    name=row["algorithm"],
                    opacity=0.7,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Multi-Dimensional Performance Comparison",
            height=500,
        )
        return fig

    def save_results(self, results, training_data, filename_prefix="taxi_rl_results"):
        """Save results and create download links"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory if it doesn't exist
        results_dir = "streamlit_results"
        os.makedirs(results_dir, exist_ok=True)

        # Save JSON results
        json_filename = f"{results_dir}/{filename_prefix}_{timestamp}.json"
        save_data = {
            "timestamp": timestamp,
            "results": results,
            "training_data": training_data,
            "metadata": {
                "total_algorithms": len(results),
                "best_performer": (
                    max(results, key=lambda x: x.get("efficiency_score", 0))[
                        "algorithm"
                    ]
                    if results
                    else None
                ),
            },
        }

        with open(json_filename, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        # Save CSV summary
        if results:
            df = pd.DataFrame(results)
            csv_filename = f"{results_dir}/{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)

            return json_filename, csv_filename

        return json_filename, None


def main():
    # Initialize the app
    app = StreamlitTaxiDriver()

    # Main header
    st.markdown(
        '<h1 class="main-header">🚕 Taxi Driver RL Interactive Interface</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "🏠 Home",
            "🔧 Single Algorithm Testing",
            "📊 Algorithm Comparison",
            "🎯 Hyperparameter Optimization",
            "📈 Advanced Analysis",
            "💾 Results Management",
        ],
    )

    if page == "🏠 Home":
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
        """
        )

        # Quick stats if we have results
        if st.session_state.results_history:
            st.subheader("Recent Results Summary")
            df = pd.DataFrame(st.session_state.results_history[-5:])  # Last 5 results
            st.dataframe(
                df[["algorithm", "mean_reward", "win_rate", "efficiency_score"]],
                use_container_width=True,
            )

    elif page == "🔧 Single Algorithm Testing":
        st.header("Single Algorithm Testing")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Configuration")

            # Algorithm selection
            algorithm = st.selectbox(
                "Choose Algorithm", ["BruteForce", "Q-Learning", "SARSA", "DQN"]
            )

            # Training parameters
            st.subheader("Training Parameters")
            train_episodes = st.number_input(
                "Training Episodes", min_value=1, max_value=5000, value=1000
            )
            test_episodes = st.number_input(
                "Test Episodes", min_value=1, max_value=1000, value=100
            )

            # Algorithm-specific parameters
            if algorithm != "BruteForce":
                st.subheader("Algorithm Parameters")
                alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.15, 0.01)
                gamma = st.slider("Discount Factor (γ)", 0.8, 0.999, 0.99, 0.001)
                epsilon = st.slider("Initial Exploration (ε)", 0.1, 1.0, 1.0, 0.1)
                epsilon_decay = st.slider(
                    "Exploration Decay", 0.99, 0.9999, 0.995, 0.0001
                )
                epsilon_min = st.slider("Min Exploration", 0.001, 0.1, 0.01, 0.001)

                if algorithm == "DQN":
                    memory_size = st.number_input(
                        "Memory Size", min_value=1000, max_value=50000, value=10000
                    )
                    batch_size = st.number_input(
                        "Batch Size", min_value=16, max_value=128, value=32
                    )

                params = {
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "epsilon_decay": epsilon_decay,
                    "epsilon_min": epsilon_min,
                }
                if algorithm == "DQN":
                    params.update(
                        {"memory_size": memory_size, "batch_size": batch_size}
                    )
            else:
                params = {}

            # Run button
            if st.button("🚀 Run Training & Evaluation", type="primary"):
                with col2:
                    st.subheader("Training Progress")

                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Training phase
                    if algorithm != "BruteForce":
                        status_text.text("🏋️ Training in progress...")

                        agent = app.create_agent(algorithm, **params)

                        # Custom training with progress updates
                        start_time = time.time()
                        progress_container = st.empty()

                        with progress_container.container():
                            train_result = train_agent(
                                st.session_state.env, agent, train_episodes, algorithm
                            )

                        progress_bar.progress(0.7)
                        training_time = time.time() - start_time
                        status_text.text(
                            f"✅ Training completed in {training_time:.2f}s"
                        )
                    else:
                        agent = app.create_agent(algorithm)
                        train_result = {"rewards": [], "steps": [], "training_time": 0}
                        progress_bar.progress(0.7)

                    # Evaluation phase
                    status_text.text("🔍 Evaluating performance...")
                    test_result = evaluate_agent(
                        st.session_state.env, agent, test_episodes, algorithm
                    )
                    progress_bar.progress(1.0)
                    status_text.text("🎉 Evaluation completed!")

                    # Store results
                    test_result["training_time"] = train_result.get("training_time", 0)
                    st.session_state.results_history.append(test_result)
                    if algorithm != "BruteForce":
                        st.session_state.training_history.append(
                            {algorithm: train_result}
                        )

                    # Display results
                    st.subheader("Results")
                    app.display_metrics(test_result)

                    # Training progress chart
                    if algorithm != "BruteForce" and train_result.get("rewards"):
                        st.subheader("Training Progress")
                        training_fig = app.create_training_progress_chart(
                            {algorithm: train_result}
                        )
                        if training_fig:
                            st.plotly_chart(training_fig, use_container_width=True)

        with col2:
            if not st.session_state.results_history:
                st.info("👆 Configure and run an algorithm to see results here!")

    elif page == "📊 Algorithm Comparison":
        st.header("Algorithm Comparison")

        # Algorithm selection for comparison
        st.subheader("Select Algorithms to Compare")
        algorithms_to_compare = st.multiselect(
            "Choose algorithms",
            ["BruteForce", "Q-Learning", "SARSA", "DQN"],
            default=["BruteForce", "Q-Learning"],
        )

        if len(algorithms_to_compare) < 2:
            st.warning("Please select at least 2 algorithms to compare.")
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
            )
            test_episodes = st.number_input(
                "Test Episodes", min_value=50, max_value=500, value=100, key="comp_test"
            )

            # Global parameters
            st.subheader("Algorithm Parameters")
            alpha = st.slider(
                "Learning Rate (α)", 0.01, 1.0, 0.15, 0.01, key="comp_alpha"
            )
            gamma = st.slider(
                "Discount Factor (γ)", 0.8, 0.999, 0.99, 0.001, key="comp_gamma"
            )

        with col2:
            st.subheader("Comparison Options")
            include_stats = st.checkbox("Include Statistical Analysis", value=True)
            show_training_curves = st.checkbox("Show Training Curves", value=True)
            save_results_auto = st.checkbox("Auto-save Results", value=True)

        if st.button("🔄 Run Comparison", type="primary"):
            results = []
            training_data = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, algorithm in enumerate(algorithms_to_compare):
                status_text.text(
                    f"🏃 Training {algorithm}... ({i+1}/{len(algorithms_to_compare)})"
                )

                params = {
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": 1.0,
                    "epsilon_decay": 0.995,
                    "epsilon_min": 0.01,
                }
                if algorithm == "DQN":
                    params.update({"memory_size": 10000, "batch_size": 32})

                agent = app.create_agent(
                    algorithm, **params if algorithm != "BruteForce" else {}
                )

                # Training
                if algorithm != "BruteForce":
                    train_result = train_agent(
                        st.session_state.env, agent, train_episodes, algorithm
                    )
                    training_data[algorithm] = train_result
                else:
                    train_result = {"training_time": 0}

                # Evaluation
                test_result = evaluate_agent(
                    st.session_state.env, agent, test_episodes, algorithm
                )
                test_result["training_time"] = train_result["training_time"]
                results.append(test_result)

                progress_bar.progress((i + 1) / len(algorithms_to_compare))

            status_text.text("✅ All algorithms completed!")

            # Store in session state
            st.session_state.comparison_data = {
                "results": results,
                "training_data": training_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Display results
            st.subheader("Comparison Results")

            # Performance metrics
            cols = st.columns(len(results))
            for i, result in enumerate(results):
                app.display_metrics_in_column(result, cols[i])

            # Comparison charts
            st.subheader("Visual Comparison")

            # Bar chart comparison
            comparison_fig = app.create_comparison_chart(results)
            if comparison_fig:
                st.plotly_chart(comparison_fig, use_container_width=True)

            # Radar chart
            radar_fig = app.create_radar_chart(results)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)

            # Training curves
            if show_training_curves and training_data:
                st.subheader("Training Progress Comparison")
                training_fig = app.create_training_progress_chart(training_data)
                if training_fig:
                    st.plotly_chart(training_fig, use_container_width=True)

            # Statistical analysis
            if include_stats:
                st.subheader("Statistical Analysis")

                # Create summary table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Performance improvement over BruteForce
                if any(r["algorithm"] == "BruteForce" for r in results):
                    bf_result = next(
                        r for r in results if r["algorithm"] == "BruteForce"
                    )

                    st.subheader("Improvement Over BruteForce")
                    improvement_data = []

                    for result in results:
                        if result["algorithm"] != "BruteForce":
                            reward_improvement = (
                                (result["mean_reward"] - bf_result["mean_reward"])
                                / abs(bf_result["mean_reward"])
                            ) * 100
                            step_improvement = (
                                (bf_result["mean_steps"] - result["mean_steps"])
                                / bf_result["mean_steps"]
                            ) * 100

                            improvement_data.append(
                                {
                                    "Algorithm": result["algorithm"],
                                    "Reward Improvement (%)": round(
                                        reward_improvement, 1
                                    ),
                                    "Step Improvement (%)": round(step_improvement, 1),
                                    "Efficiency Gain": round(
                                        result.get("efficiency_score", 0)
                                        / max(
                                            bf_result.get("efficiency_score", 0.001),
                                            0.001,
                                        ),
                                        2,
                                    ),
                                }
                            )

                    if improvement_data:
                        improvement_df = pd.DataFrame(improvement_data)
                        st.dataframe(improvement_df, use_container_width=True)

            # Save results
            if save_results_auto:
                json_file, csv_file = app.save_results(
                    results, training_data, "comparison"
                )
                st.success(f"✅ Results saved to {json_file}")

    elif page == "🎯 Hyperparameter Optimization":
        st.header("Hyperparameter Optimization")

        st.markdown(
            """
        Automatically find the best hyperparameters for your chosen algorithm.
        """
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Optimization Configuration")

            algorithm = st.selectbox(
                "Algorithm to Optimize", ["Q-Learning", "SARSA", "DQN"], key="opt_algo"
            )
            optimization_method = st.selectbox(
                "Optimization Method", ["Grid Search", "Random Search", "Bayesian"]
            )

            st.subheader("Search Parameters")
            n_configs = st.number_input(
                "Number of Configurations", min_value=5, max_value=50, value=20
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
                "Learning Rate Range", 0.01, 0.5, (0.05, 0.3), key="opt_alpha_range"
            )
            gamma_range = st.slider(
                "Discount Factor Range", 0.9, 0.999, (0.95, 0.99), key="opt_gamma_range"
            )

        with col2:
            if st.button("🎯 Start Optimization", type="primary"):
                st.subheader("Optimization Progress")

                # Generate parameter configurations
                if optimization_method == "Grid Search":
                    # Create grid of parameters
                    alphas = np.linspace(
                        alpha_range[0], alpha_range[1], int(np.sqrt(n_configs))
                    )
                    gammas = np.linspace(
                        gamma_range[0], gamma_range[1], int(np.sqrt(n_configs))
                    )
                    configs = []
                    for alpha in alphas:
                        for gamma in gammas:
                            configs.append(
                                {
                                    "alpha": alpha,
                                    "gamma": gamma,
                                    "epsilon": 1.0,
                                    "epsilon_decay": 0.995,
                                    "epsilon_min": 0.01,
                                }
                            )
                            if len(configs) >= n_configs:
                                break
                        if len(configs) >= n_configs:
                            break

                elif optimization_method == "Random Search":
                    configs = []
                    for _ in range(n_configs):
                        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
                        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                        configs.append(
                            {
                                "alpha": alpha,
                                "gamma": gamma,
                                "epsilon": 1.0,
                                "epsilon_decay": 0.995,
                                "epsilon_min": 0.01,
                            }
                        )

                else:  # Bayesian (simplified)
                    configs = []
                    for _ in range(n_configs):
                        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
                        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                        configs.append(
                            {
                                "alpha": alpha,
                                "gamma": gamma,
                                "epsilon": 1.0,
                                "epsilon_decay": 0.995,
                                "epsilon_min": 0.01,
                            }
                        )

                # Run optimization
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()

                best_score = -float("inf")
                best_config = None

                for i, config in enumerate(configs):
                    status_text.text(
                        f"Testing configuration {i+1}/{len(configs)}: α={config['alpha']:.3f}, γ={config['gamma']:.3f}"
                    )

                    # Create and train agent
                    agent = app.create_agent(algorithm, **config)

                    if algorithm != "BruteForce":
                        train_result = train_agent(
                            st.session_state.env,
                            agent,
                            train_episodes,
                            f"{algorithm}_opt",
                        )
                    else:
                        train_result = {"training_time": 0}

                    test_result = evaluate_agent(
                        st.session_state.env, agent, test_episodes, f"{algorithm}_opt"
                    )
                    test_result.update(config)
                    test_result["config_id"] = i
                    results.append(test_result)

                    # Track best configuration
                    current_score = test_result.get("efficiency_score", 0)
                    if current_score > best_score:
                        best_score = current_score
                        best_config = config.copy()
                        best_config["score"] = current_score

                    progress_bar.progress((i + 1) / len(configs))

                    # Update results display
                    if (i + 1) % 5 == 0 or i == len(configs) - 1:
                        with results_container.container():
                            st.write(
                                f"**Current Best:** α={best_config['alpha']:.3f}, γ={best_config['gamma']:.3f}, Score={best_score:.4f}"
                            )

                            # Show progress chart
                            if len(results) > 3:
                                progress_df = pd.DataFrame(results)
                                fig = px.scatter(
                                    progress_df,
                                    x="alpha",
                                    y="gamma",
                                    color="efficiency_score",
                                    size="win_rate",
                                    title="Parameter Space Exploration",
                                    color_continuous_scale="viridis",
                                )
                                st.plotly_chart(fig, use_container_width=True)

                status_text.text("✅ Optimization completed!")

                # Display final results
                st.subheader("Optimization Results")

                st.markdown(
                    f"""
                <div class="success-box">
                <h4>🏆 Best Configuration Found:</h4>
                <p><strong>Learning Rate (α):</strong> {best_config['alpha']:.4f}</p>
                <p><strong>Discount Factor (γ):</strong> {best_config['gamma']:.4f}</p>
                <p><strong>Efficiency Score:</strong> {best_score:.4f}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Results analysis
                results_df = pd.DataFrame(results)

                # Parameter correlation heatmap
                st.subheader("Parameter Analysis")
                correlation_params = [
                    "alpha",
                    "gamma",
                    "efficiency_score",
                    "win_rate",
                    "mean_reward",
                ]
                corr_matrix = results_df[correlation_params].corr()

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)

                # Save optimization results
                json_file, csv_file = app.save_results(
                    results, {}, f"optimization_{algorithm.lower()}"
                )
                st.success(f"✅ Optimization results saved to {json_file}")

    elif page == "📈 Advanced Analysis":
        st.header("Advanced Statistical Analysis")

        if not st.session_state.comparison_data:
            st.warning(
                "Please run an algorithm comparison first to see advanced analysis."
            )
            return

        results = st.session_state.comparison_data["results"]
        training_data = st.session_state.comparison_data["training_data"]

        # Comprehensive statistical analysis
        with st.spinner("Performing advanced statistical analysis..."):
            statistical_analysis = app.stat_analyzer.comprehensive_analysis(
                results, list(training_data.values())
            )

        st.subheader("Statistical Summary")

        # Descriptive statistics
        if "descriptive_stats" in statistical_analysis:
            st.write("### Descriptive Statistics")
            desc_stats = statistical_analysis["descriptive_stats"]

            stats_data = []
            for metric, stats in desc_stats.items():
                if isinstance(stats, dict):
                    stats_data.append(
                        {
                            "Metric": metric.replace("_", " ").title(),
                            "Mean": f"{stats.get('mean', 0):.3f}",
                            "Std": f"{stats.get('std', 0):.3f}",
                            "Min": f"{stats.get('min', 0):.3f}",
                            "Max": f"{stats.get('max', 0):.3f}",
                            "CV": f"{stats.get('cv', 0):.3f}",
                        }
                    )

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

        # Performance clustering
        if "performance_clustering" in statistical_analysis:
            clustering = statistical_analysis["performance_clustering"]
            if "error" not in clustering:
                st.write("### Performance Clustering")
                st.write(
                    f"Optimal number of clusters: {clustering.get('n_clusters', 'N/A')}"
                )

                if "pca_components" in clustering:
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

                    fig = px.scatter(
                        cluster_df,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                        hover_data=["Algorithm"],
                        title="Performance Clustering (PCA)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Hypothesis testing
        if "hypothesis_testing" in statistical_analysis:
            hypothesis_tests = statistical_analysis["hypothesis_testing"]
            if "error" not in hypothesis_tests:
                st.write("### Statistical Significance Testing")

                for comparison, tests in hypothesis_tests.items():
                    st.write(f"**{comparison}:**")
                    test_results = []

                    for metric, test_data in tests.items():
                        if isinstance(test_data, dict):
                            # Assuming Mann-Whitney U test results
                            p_value = test_data.get("p_value", 1.0)
                            significance = (
                                "***"
                                if p_value < 0.001
                                else (
                                    "**"
                                    if p_value < 0.01
                                    else "*" if p_value < 0.05 else "ns"
                                )
                            )

                            test_results.append(
                                {
                                    "Metric": metric.replace("_", " ").title(),
                                    "P-value": f"{p_value:.4f}",
                                    "Significance": significance,
                                }
                            )

                    if test_results:
                        test_df = pd.DataFrame(test_results)
                        st.dataframe(test_df, use_container_width=True)

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
                            "Lower Bound": f"{ci_info.get('lower_bound', 0):.3f}",
                            "Upper Bound": f"{ci_info.get('upper_bound', 0):.3f}",
                            "Margin of Error": f"{ci_info.get('margin_error', 0):.3f}",
                        }
                    )

            if ci_results:
                ci_df = pd.DataFrame(ci_results)
                st.dataframe(ci_df, use_container_width=True)

    elif page == "💾 Results Management":
        st.header("Results Management")

        # Results history
        if st.session_state.results_history:
            st.subheader("Results History")

            # Create summary table
            history_df = pd.DataFrame(st.session_state.results_history)
            st.dataframe(history_df, use_container_width=True)

            # Download options
            st.subheader("Download Results")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📥 Download as CSV"):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV file",
                        data=csv,
                        file_name=f"taxi_rl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("📥 Download as JSON"):
                    json_data = json.dumps(
                        st.session_state.results_history, indent=2, default=str
                    )
                    st.download_button(
                        label="Download JSON file",
                        data=json_data,
                        file_name=f"taxi_rl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

            # Clear history
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.results_history = []
                st.session_state.training_history = []
                st.session_state.comparison_data = {}
                st.success("History cleared!")
                st.experimental_rerun()

        else:
            st.info(
                "No results available yet. Run some experiments to see results here!"
            )

        # Load saved results
        st.subheader("Load Saved Results")
        uploaded_file = st.file_uploader("Choose a results file", type=["json", "csv"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".json"):
                    data = json.load(uploaded_file)
                    if "results" in data:
                        st.session_state.results_history.extend(data["results"])
                    else:
                        st.session_state.results_history.extend(data)
                elif uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.results_history.extend(df.to_dict("records"))

                st.success("Results loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")


if __name__ == "__main__":
    main()
