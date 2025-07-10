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
    st.header("Optimisation des Hyperparamètres")

    st.markdown(
        """
    Trouvez automatiquement les meilleurs hyperparamètres pour l'algorithme choisi en utilisant différentes stratégies d'optimisation.
    """
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration d'Optimisation")

        algorithm = st.selectbox(
            "Algorithme à Optimiser",
            [algo for algo in ALGORITHMS if algo != "BruteForce"],
            key="opt_algo",
        )

        optimization_method = st.selectbox(
            "Méthode d'Optimisation",
            ["Recherche Aléatoire", "Recherche en Grille"],
            help="Recherche Aléatoire : Teste des combinaisons de paramètres aléatoires\nRecherche en Grille : Teste toutes les combinaisons dans une grille",
        )

        st.subheader("Paramètres de Recherche")
        n_configs = st.number_input(
            "Nombre de Configurations",
            min_value=5,
            max_value=50,
            value=20,
            help="Nombre de combinaisons de paramètres à tester",
        )

        train_episodes = st.number_input(
            "Épisodes d'Entraînement par Config",
            min_value=100,
            max_value=1000,
            value=300,
            key="opt_train",
        )

        test_episodes = st.number_input(
            "Épisodes de Test par Config",
            min_value=20,
            max_value=100,
            value=50,
            key="opt_test",
        )

        st.subheader("Plages de Paramètres")
        alpha_range = st.slider(
            "Plage du Taux d'Apprentissage",
            0.01,
            0.5,
            (0.05, 0.3),
            key="opt_alpha_range",
            help="Plage des taux d'apprentissage à explorer",
        )

        gamma_range = st.slider(
            "Plage du Facteur d'Actualisation",
            0.9,
            0.999,
            (0.95, 0.99),
            key="opt_gamma_range",
            help="Plage des facteurs d'actualisation à explorer",
        )

        # Advanced options
        with st.expander("Options Avancées"):
            early_stopping = st.checkbox(
                "Arrêt Anticipé",
                value=False,
                help="Arrêter si aucune amélioration trouvée",
            )
            if early_stopping:
                patience = st.number_input(
                    "Patience", min_value=3, max_value=10, value=5
                )

            save_all_configs = st.checkbox(
                "Sauvegarder Toutes les Configurations", value=True
            )

    with col2:
        if st.button("🎯 Commencer l'Optimisation", type="primary"):
            st.subheader("Progression de l'Optimisation")

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
                method_translation = {
                    "Recherche Aléatoire": "random",
                    "Recherche en Grille": "grid",
                }
                optimization_result = training_manager.optimize_hyperparameters(
                    algorithm,
                    param_ranges,
                    n_configs,
                    train_episodes,
                    test_episodes,
                    method=method_translation.get(optimization_method, "random"),
                )

                results = optimization_result["results"]
                best_config = optimization_result["best_config"]
                best_score = optimization_result["best_score"]

                status_text.text("✅ Optimisation terminée !")
                progress_bar.progress(1.0)

                # Display final results
                st.subheader("Résultats d'Optimisation")

                if best_config:
                    VisualizationComponents.display_success_message(
                        f"""
                        <h4>🏆 Meilleure Configuration Trouvée :</h4>
                        <p><strong>Taux d'Apprentissage (α) :</strong> {best_config['alpha']:.4f}</p>
                        <p><strong>Facteur d'Actualisation (γ) :</strong> {best_config['gamma']:.4f}</p>
                        <p><strong>Score d'Efficacité :</strong> {best_score:.4f}</p>
                        """
                    )

                    # Configuration comparison with default
                    default_config = {
                        "alpha": 0.15,
                        "gamma": 0.99,
                        "efficiency_score": 0.0,  # Placeholder
                    }

                    comparison_data = {
                        "Configuration": ["Par Défaut", "Optimisée"],
                        "Taux d'Apprentissage": [
                            default_config["alpha"],
                            best_config["alpha"],
                        ],
                        "Facteur d'Actualisation": [
                            default_config["gamma"],
                            best_config["gamma"],
                        ],
                        "Score d'Efficacité": [
                            default_config["efficiency_score"],
                            best_score,
                        ],
                    }

                    st.subheader("Comparaison de Configuration")
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                # Results analysis
                if results:
                    results_df = pd.DataFrame(results)

                    # Optimization progress chart
                    st.subheader("Progression de l'Optimisation")
                    if len(results) > 3:
                        # Parameter space visualization
                        param_fig = VisualizationComponents.create_parameter_space_plot(
                            results_df
                        )
                        if param_fig:
                            st.plotly_chart(param_fig, use_container_width=True)

                        # Progress over time
                        results_df["config_order"] = range(len(results_df))

                        import plotly.express as px

                        progress_fig = px.line(
                            results_df,
                            x="config_order",
                            y="efficiency_score",
                            title="Progression de l'Optimisation par Configuration",
                            labels={
                                "config_order": "Numéro de Configuration",
                                "efficiency_score": "Score d'Efficacité",
                            },
                        )
                        st.plotly_chart(progress_fig, use_container_width=True)

                    # Parameter correlation analysis
                    st.subheader("Analyse des Paramètres")
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
                    st.subheader("Résumé Statistique")
                    summary_stats = {
                        "Métrique": [
                            "Meilleur Score",
                            "Pire Score",
                            "Score Moyen",
                            "Écart-Type Score",
                        ],
                        "Valeur": [
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
                        f"✅ Résultats d'optimisation sauvegardés dans {json_file}"
                    )

                    # Download buttons
                    col_download1, col_download2 = st.columns(2)
                    with col_download1:
                        if csv_file:
                            with open(csv_file, "r") as f:
                                st.download_button(
                                    "📥 Télécharger CSV",
                                    f.read(),
                                    file_name=f"optimization_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )

                    with col_download2:
                        with open(json_file, "r") as f:
                            st.download_button(
                                "📥 Télécharger JSON",
                                f.read(),
                                file_name=f"optimization_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                            )

                # Option to test best configuration
                st.subheader("Tester la Meilleure Configuration")
                if st.button("🧪 Tester la Meilleure Configuration", key="test_best"):
                    st.info(
                        "Test de la meilleure configuration avec plus d'épisodes..."
                    )

                    # Test with more episodes
                    extended_result = training_manager.train_single_agent(
                        algorithm, best_config, train_episodes * 2, test_episodes * 2
                    )

                    st.subheader("Résultats de Test Étendus")
                    VisualizationComponents.display_metrics(
                        extended_result["test_result"]
                    )

            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Optimisation échouée !")
                VisualizationComponents.display_error_message(
                    f"Erreur d'optimisation : {str(e)}"
                )
                st.error(f"Optimisation échouée : {e}")

        else:
            st.info(
                "👆 Configurez vos paramètres d'optimisation et cliquez sur 'Commencer l'Optimisation' pour débuter !"
            )

            # Show optimization tips
            st.subheader("Conseils d'Optimisation")
            tips = [
                "🎯 **Commencer Petit** : Commencez avec 10-20 configurations pour vous familiariser avec l'espace des paramètres",
                "📊 **Aléatoire vs Grille** : La recherche aléatoire trouve souvent de bonnes solutions plus rapidement que la recherche en grille",
                "⚡ **Nombre d'Épisodes** : Utilisez moins d'épisodes par config pour une exploration plus rapide, puis testez les meilleures configs plus longtemps",
                "📈 **Plages de Paramètres** : Commencez avec des plages plus larges, puis affinez selon les résultats",
                "🔄 **Processus Itératif** : Lancez plusieurs cycles d'optimisation, en affinant les plages à chaque fois",
            ]

            for tip in tips:
                st.markdown(f"- {tip}")
                st.markdown(f"- {tip}")


# Import plotly for the progress visualization
import plotly.express as px
