import streamlit as st
import pandas as pd
from datetime import datetime

from ..config import ALGORITHMS
from ..components.visualizations import VisualizationComponents
from ..utils.training_manager import TrainingManager
from ..utils.data_manager import DataManager


def render_algorithm_comparison_page():
    st.header("Comparaison d'Algorithmes")

    # Algorithm selection for comparison
    st.subheader("Sélectionner les Algorithmes à Comparer")
    algorithms_to_compare = st.multiselect(
        "Choisir les algorithmes",
        ALGORITHMS,
        default=["BruteForce", "Q-Learning"],
        help="Sélectionnez au moins 2 algorithmes pour comparer leurs performances",
    )

    if len(algorithms_to_compare) < 2:
        VisualizationComponents.display_warning_message(
            "⚠️ Veuillez sélectionner au moins 2 algorithmes à comparer."
        )
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Paramètres de Comparaison")
        train_episodes = st.number_input(
            "Épisodes d'Entraînement",
            min_value=100,
            max_value=3000,
            value=1000,
            key="comp_train",
            help="Nombre d'épisodes pour entraîner chaque algorithme",
        )
        test_episodes = st.number_input(
            "Épisodes de Test",
            min_value=50,
            max_value=500,
            value=100,
            key="comp_test",
            help="Nombre d'épisodes pour évaluer chaque algorithme",
        )

        # Global parameters for all algorithms
        st.subheader("Paramètres d'Algorithme")
        alpha = st.slider(
            "Taux d'Apprentissage (α)",
            0.01,
            1.0,
            0.15,
            0.01,
            key="comp_alpha",
            help="Taux d'apprentissage pour tous les algorithmes",
        )
        gamma = st.slider(
            "Facteur d'Actualisation (γ)",
            0.8,
            0.999,
            0.99,
            0.001,
            key="comp_gamma",
            help="Facteur d'actualisation pour tous les algorithmes",
        )

    with col2:
        st.subheader("Options de Comparaison")
        include_stats = st.checkbox("Inclure l'Analyse Statistique", value=True)
        show_training_curves = st.checkbox(
            "Afficher les Courbes d'Entraînement", value=True
        )
        save_results_auto = st.checkbox(
            "Sauvegarde Automatique des Résultats", value=True
        )
        show_radar_chart = st.checkbox("Afficher le Graphique Radar", value=True)

    # Run comparison button
    if st.button("🔄 Lancer la Comparaison", type="primary"):
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
                    f"🏃 Entraînement {algorithm}... ({i+1}/{len(algorithms_to_compare)})"
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
                    st.subheader("Résultats Actuels")
                    cols = st.columns(len(results))
                    for j, res in enumerate(results):
                        VisualizationComponents.display_metrics_in_column(res, cols[j])

            status_text.text("✅ Tous les algorithmes terminés !")

            # Store in session state
            st.session_state.comparison_data = {
                "results": results,
                "training_data": training_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Display comprehensive results
            st.subheader("Résultats de Comparaison")

            # Performance metrics grid
            st.subheader("Aperçu des Performances")
            cols = st.columns(len(results))
            for i, result in enumerate(results):
                VisualizationComponents.display_metrics_in_column(result, cols[i])

            # Comparison charts
            st.subheader("Comparaison Visuelle")

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(
                [
                    "📊 Graphiques en Barres",
                    "🎯 Graphique Radar",
                    "📈 Courbes d'Entraînement",
                ]
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
                    st.info(
                        "Graphique radar désactivé. Activez-le dans les options ci-dessus."
                    )

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
                    st.info(
                        "Aucune donnée d'entraînement disponible ou courbes d'entraînement désactivées."
                    )

            # Statistical analysis
            if include_stats:
                st.subheader("Analyse Statistique")

                # Create summary table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Performance ranking
                st.subheader("Classement des Performances")
                ranking_data = []
                for result in results:
                    ranking_data.append(
                        {
                            "Algorithme": result["algorithm"],
                            "Score d'Efficacité": result.get("efficiency_score", 0),
                            "Récompense Moyenne": result["mean_reward"],
                            "Taux de Réussite": result["win_rate"],
                            "Étapes Moyennes": result["mean_steps"],
                        }
                    )

                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values(
                    "Score d'Efficacité", ascending=False
                )
                ranking_df["Rang"] = range(1, len(ranking_df) + 1)

                # Reorder columns
                ranking_df = ranking_df[
                    [
                        "Rang",
                        "Algorithme",
                        "Score d'Efficacité",
                        "Récompense Moyenne",
                        "Taux de Réussite",
                        "Étapes Moyennes",
                    ]
                ]
                st.dataframe(ranking_df, use_container_width=True)

                # Performance improvement over baseline
                if any(r["algorithm"] == "BruteForce" for r in results):
                    st.subheader("Amélioration par Rapport à la Référence BruteForce")
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
                                    "Algorithme": result["algorithm"],
                                    "Amélioration Récompense (%)": round(
                                        reward_improvement, 1
                                    ),
                                    "Amélioration Étapes (%)": round(
                                        step_improvement, 1
                                    ),
                                    "Gain d'Efficacité (x)": round(efficiency_gain, 2),
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
                    f"✅ Résultats sauvegardés automatiquement dans {json_file}"
                )

                # Download buttons
                col_download1, col_download2 = st.columns(2)
                with col_download1:
                    if csv_file:
                        with open(csv_file, "r") as f:
                            st.download_button(
                                "📥 Télécharger CSV",
                                f.read(),
                                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

                with col_download2:
                    with open(json_file, "r") as f:
                        st.download_button(
                            "📥 Télécharger JSON",
                            f.read(),
                            file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )

        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ Erreur survenue !")
            VisualizationComponents.display_error_message(
                f"Une erreur s'est produite lors de la comparaison : {str(e)}"
            )
            st.error(f"Échec de la comparaison : {e}")

    # Show previous comparison if available
    elif st.session_state.comparison_data:
        st.subheader("Résultats de Comparaison Précédents")

        previous_results = st.session_state.comparison_data["results"]
        previous_training = st.session_state.comparison_data["training_data"]

        # Quick overview
        st.write(f"**Horodatage :** {st.session_state.comparison_data['timestamp']}")
        st.write(
            f"**Algorithmes Comparés :** {', '.join([r['algorithm'] for r in previous_results])}"
        )

        # Show condensed results
        cols = st.columns(len(previous_results))
        for i, result in enumerate(previous_results):
            VisualizationComponents.display_metrics_in_column(result, cols[i])

        # Option to clear previous results
        if st.button("🗑️ Effacer les Résultats Précédents"):
            st.session_state.comparison_data = {}
            st.experimental_rerun()
