import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


class VisualizationComponents:

    @staticmethod
    def display_metrics(result):
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>{result['algorithm']}</h4>
            <p><strong>Récompense Moyenne :</strong> {result['mean_reward']:.2f} ± {result.get('std_reward', 0):.2f}</p>
            <p><strong>Étapes Moyennes :</strong> {result['mean_steps']:.1f} ± {result.get('std_steps', 0):.1f}</p>
            <p><strong>Taux de Réussite :</strong> {result['win_rate']:.2%}</p>
            <p><strong>Score d'Efficacité :</strong> {result.get('efficiency_score', 0):.4f}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def display_metrics_in_column(result, col):
        with col:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>{result['algorithm']}</h4>
                <p><strong>Récompense Moyenne :</strong> {result['mean_reward']:.2f} ± {result.get('std_reward', 0):.2f}</p>
                <p><strong>Étapes Moyennes :</strong> {result['mean_steps']:.1f} ± {result.get('std_steps', 0):.1f}</p>
                <p><strong>Taux de Réussite :</strong> {result['win_rate']:.2%}</p>
                <p><strong>Score d'Efficacité :</strong> {result.get('efficiency_score', 0):.4f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    @staticmethod
    def create_comparison_chart(results):
        if not results:
            return None

        df = pd.DataFrame(results)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Comparaison Récompense Moyenne",
                "Comparaison Taux de Réussite",
                "Comparaison Score d'Efficacité",
                "Comparaison Étapes Moyennes",
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
                name="Récompense Moyenne",
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
                name="Taux de Réussite",
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
                name="Score d'Efficacité",
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
                name="Étapes Moyennes",
                marker_color=colors,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=600, title_text="Comparaison des Performances d'Algorithmes"
        )
        return fig

    @staticmethod
    def create_training_progress_chart(training_data):
        if not training_data:
            return None

        fig = go.Figure()

        for algo, data in training_data.items():
            if "rewards" in data:
                episodes = list(range(len(data["rewards"])))
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
                        name=f"{algo} (lissé)",
                        line=dict(width=2),
                    )
                )

                # Add raw data with transparency
                fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=data["rewards"],
                        mode="lines",
                        name=f"{algo} (brut)",
                        line=dict(width=1),
                        opacity=0.3,
                    )
                )

        fig.update_layout(
            title="Progression de l'Entraînement par Épisodes",
            xaxis_title="Épisode",
            yaxis_title="Récompense",
            height=500,
        )
        return fig

    @staticmethod
    def create_radar_chart(results):
        if not results:
            return None

        df = pd.DataFrame(results)

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
                    theta=[
                        "Récompense",
                        "Taux de Réussite",
                        "Efficacité",
                        "Récompense",
                    ],
                    fill="toself",
                    name=row["algorithm"],
                    opacity=0.7,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparaison de Performance Multi-Dimensionnelle",
            height=500,
        )
        return fig

    @staticmethod
    def create_parameter_correlation_heatmap(results_df):
        correlation_params = [
            "alpha",
            "gamma",
            "efficiency_score",
            "win_rate",
            "mean_reward",
        ]

        available_params = [p for p in correlation_params if p in results_df.columns]
        if len(available_params) < 2:
            return None

        corr_matrix = results_df[available_params].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        return fig

    @staticmethod
    def create_parameter_space_plot(results_df):
        if "alpha" not in results_df.columns or "gamma" not in results_df.columns:
            return None

        fig = px.scatter(
            results_df,
            x="alpha",
            y="gamma",
            color="efficiency_score",
            size="win_rate",
            title="Exploration de l'Espace des Paramètres",
            color_continuous_scale="viridis",
        )
        return fig

    @staticmethod
    def display_success_message(message):
        """Display success message"""
        st.markdown(
            f"""
        <div class="success-box">
            {message}
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def display_warning_message(message):
        """Display warning message"""
        st.markdown(
            f"""
        <div class="warning-box">
            {message}
        </div>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def display_error_message(message):
        """Display error message"""
        st.markdown(
            f"""
        <div class="error-box">
            {message}
        </div>
        """,
            unsafe_allow_html=True,
        )
