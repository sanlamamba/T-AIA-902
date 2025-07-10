import streamlit as st
import time

from ..config import DEFAULT_PARAMS, ALGORITHMS
from ..components.visualizations import VisualizationComponents
from ..utils.training_manager import TrainingManager
from ..utils.agent_factory import AgentFactory


def render_single_algorithm_page():
    st.header("Test d'Algorithme Unique")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")

        algorithm = st.selectbox("Choisir un Algorithme", ALGORITHMS)

        st.subheader("Paramètres d'Entraînement")
        train_episodes = st.number_input(
            "Épisodes d'Entraînement",
            min_value=DEFAULT_PARAMS["train_episodes"]["min"],
            max_value=DEFAULT_PARAMS["train_episodes"]["max"],
            value=DEFAULT_PARAMS["train_episodes"]["default"],
        )
        test_episodes = st.number_input(
            "Épisodes de Test",
            min_value=DEFAULT_PARAMS["test_episodes"]["min"],
            max_value=DEFAULT_PARAMS["test_episodes"]["max"],
            value=DEFAULT_PARAMS["test_episodes"]["default"],
        )

        # Algorithm-specific parameters
        params = {}
        if algorithm != "BruteForce":
            st.subheader("Paramètres d'Algorithme")

            params["alpha"] = st.slider(
                "Taux d'Apprentissage (α)",
                min_value=DEFAULT_PARAMS["alpha"]["min"],
                max_value=DEFAULT_PARAMS["alpha"]["max"],
                value=DEFAULT_PARAMS["alpha"]["default"],
                step=DEFAULT_PARAMS["alpha"]["step"],
            )

            params["gamma"] = st.slider(
                "Facteur d'Actualisation (γ)",
                min_value=DEFAULT_PARAMS["gamma"]["min"],
                max_value=DEFAULT_PARAMS["gamma"]["max"],
                value=DEFAULT_PARAMS["gamma"]["default"],
                step=DEFAULT_PARAMS["gamma"]["step"],
            )

            params["epsilon"] = st.slider(
                "Exploration Initiale (ε)",
                min_value=DEFAULT_PARAMS["epsilon"]["min"],
                max_value=DEFAULT_PARAMS["epsilon"]["max"],
                value=DEFAULT_PARAMS["epsilon"]["default"],
                step=DEFAULT_PARAMS["epsilon"]["step"],
            )

            params["epsilon_decay"] = st.slider(
                "Décroissance d'Exploration",
                min_value=DEFAULT_PARAMS["epsilon_decay"]["min"],
                max_value=DEFAULT_PARAMS["epsilon_decay"]["max"],
                value=DEFAULT_PARAMS["epsilon_decay"]["default"],
                step=DEFAULT_PARAMS["epsilon_decay"]["step"],
            )

            params["epsilon_min"] = st.slider(
                "Exploration Minimale",
                min_value=DEFAULT_PARAMS["epsilon_min"]["min"],
                max_value=DEFAULT_PARAMS["epsilon_min"]["max"],
                value=DEFAULT_PARAMS["epsilon_min"]["default"],
                step=DEFAULT_PARAMS["epsilon_min"]["step"],
            )

            if algorithm == "DQN":
                params["memory_size"] = st.number_input(
                    "Taille de Mémoire",
                    min_value=DEFAULT_PARAMS["memory_size"]["min"],
                    max_value=DEFAULT_PARAMS["memory_size"]["max"],
                    value=DEFAULT_PARAMS["memory_size"]["default"],
                )
                params["batch_size"] = st.number_input(
                    "Taille de Lot",
                    min_value=DEFAULT_PARAMS["batch_size"]["min"],
                    max_value=DEFAULT_PARAMS["batch_size"]["max"],
                    value=DEFAULT_PARAMS["batch_size"]["default"],
                )

        # Validation
        if algorithm != "BruteForce":
            if not AgentFactory.validate_params(algorithm, params):
                st.error(
                    "Paramètres invalides ! Veuillez vérifier vos valeurs d'entrée."
                )
                return

        # Run button
        run_experiment = st.button(
            "🚀 Lancer l'Entraînement et l'Évaluation", type="primary"
        )

    with col2:
        if run_experiment:
            st.subheader("Progression de l'Entraînement")

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Initialize training manager
                training_manager = TrainingManager(st.session_state.env)

                # Training phase
                if algorithm != "BruteForce":
                    status_text.text("🏋️ Entraînement en cours...")
                else:
                    status_text.text("🎲 Évaluation de la référence aléatoire...")

                progress_bar.progress(0.1)

                # Run training and evaluation
                start_time = time.time()
                result = training_manager.train_single_agent(
                    algorithm, params, train_episodes, test_episodes
                )

                progress_bar.progress(0.7)
                training_time = time.time() - start_time

                status_text.text("🔍 Évaluation des performances...")
                progress_bar.progress(0.9)

                # Store results
                test_result = result["test_result"]
                train_result = result["train_result"]

                st.session_state.results_history.append(test_result)
                if algorithm != "BruteForce":
                    st.session_state.training_history.append({algorithm: train_result})

                progress_bar.progress(1.0)
                status_text.text("🎉 Évaluation terminée !")

                # Display results
                st.subheader("Résultats")
                VisualizationComponents.display_metrics(test_result)

                # Performance summary
                st.subheader("Résumé des Performances")
                col_summary1, col_summary2 = st.columns(2)

                with col_summary1:
                    st.metric("Temps d'Entraînement", f"{training_time:.2f}s")
                    st.metric("Récompense Moyenne", f"{test_result['mean_reward']:.2f}")

                with col_summary2:
                    st.metric("Taux de Réussite", f"{test_result['win_rate']:.1%}")
                    st.metric(
                        "Score d'Efficacité",
                        f"{test_result.get('efficiency_score', 0):.4f}",
                    )

                # Training progress chart
                if algorithm != "BruteForce" and train_result.get("rewards"):
                    st.subheader("Progression de l'Entraînement")
                    training_fig = (
                        VisualizationComponents.create_training_progress_chart(
                            {algorithm: train_result}
                        )
                    )
                    if training_fig:
                        st.plotly_chart(training_fig, use_container_width=True)

                # Success message
                VisualizationComponents.display_success_message(
                    f"✅ {algorithm} entraîné et évalué avec succès !"
                )

            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Erreur survenue !")
                VisualizationComponents.display_error_message(f"Erreur : {str(e)}")
                st.error(f"Une erreur s'est produite : {e}")

        elif not st.session_state.results_history:
            st.info(
                "👆 Configurez et lancez un algorithme pour voir les résultats ici !"
            )
        else:
            st.subheader("Résultats Récents")
            # Show last result if available
            if st.session_state.results_history:
                last_result = st.session_state.results_history[-1]
                VisualizationComponents.display_metrics(last_result)

                # Quick action buttons
                if st.button("🔄 Relancer avec les Mêmes Paramètres"):
                    st.experimental_rerun()

                if st.button("📊 Comparer avec d'Autres Algorithmes"):
                    st.session_state.current_page = "📊 Comparaison d'Algorithmes"
                    st.experimental_rerun()
