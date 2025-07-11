import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os

from ..components.visualizations import VisualizationComponents
from ..utils.data_manager import DataManager


def render_results_management_page():
    st.header("Gestion des Résultats")

    data_manager = DataManager()

    # Results history section
    if st.session_state.results_history:
        st.subheader("Historique des Résultats")

        # Create summary table
        history_df = pd.DataFrame(st.session_state.results_history)

        # Add experiment number and timestamp if not present
        if "experiment_id" not in history_df.columns:
            history_df["experiment_id"] = range(1, len(history_df) + 1)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Expériences Totales", len(st.session_state.results_history))

        with col2:
            unique_algos = history_df["algorithm"].nunique()
            st.metric("Algorithmes Uniques", unique_algos)

        with col3:
            if "efficiency_score" in history_df.columns:
                best_score = history_df["efficiency_score"].max()
                st.metric("Meilleur Score d'Efficacité", f"{best_score:.4f}")
            else:
                st.metric("Meilleur Score d'Efficacité", "N/A")

        with col4:
            avg_win_rate = (
                history_df["win_rate"].mean() if "win_rate" in history_df.columns else 0
            )
            st.metric("Taux de Réussite Moyen", f"{avg_win_rate:.1%}")

        # Filtering options
        st.subheader("Filtrer les Résultats")

        col_filter1, col_filter2, col_filter3 = st.columns(3)

        with col_filter1:
            selected_algorithms = st.multiselect(
                "Filtrer par Algorithme",
                options=history_df["algorithm"].unique(),
                default=history_df["algorithm"].unique(),
            )

        with col_filter2:
            if "efficiency_score" in history_df.columns:
                min_efficiency = st.slider(
                    "Score d'Efficacité Minimum",
                    min_value=float(history_df["efficiency_score"].min()),
                    max_value=float(history_df["efficiency_score"].max()),
                    value=float(history_df["efficiency_score"].min()),
                )
            else:
                min_efficiency = 0

        with col_filter3:
            show_recent = st.selectbox(
                "Afficher les Résultats",
                ["Tous", "10 Derniers", "20 Derniers", "50 Derniers"],
                index=0,
            )

        # Apply filters
        filtered_df = history_df[history_df["algorithm"].isin(selected_algorithms)]

        if "efficiency_score" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["efficiency_score"] >= min_efficiency]

        if show_recent != "Tous":
            n_recent = int(show_recent.split()[-1])
            filtered_df = filtered_df.tail(n_recent)

        # Display filtered results
        st.subheader("Résultats Filtrés")

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
            "Sélectionner les Colonnes à Afficher",
            options=all_columns,
            default=available_default_columns,
        )

        if selected_columns:
            display_df = filtered_df[selected_columns]
            st.dataframe(display_df, use_container_width=True)

            # Quick statistics for filtered data
            if len(filtered_df) > 0:
                st.subheader("Statistiques pour les Données Filtrées")

                stats_col1, stats_col2 = st.columns(2)

                with stats_col1:
                    st.write("**Distribution des Algorithmes :**")
                    algo_counts = filtered_df["algorithm"].value_counts()
                    st.bar_chart(algo_counts)

                with stats_col2:
                    if "efficiency_score" in filtered_df.columns:
                        st.write("**Distribution des Scores d'Efficacité :**")
                        st.line_chart(
                            filtered_df["efficiency_score"].reset_index(drop=True)
                        )

        # Download options
        st.subheader("Télécharger les Résultats")

        download_col1, download_col2, download_col3 = st.columns(3)

        with download_col1:
            if st.button("📥 Télécharger les Filtrés en CSV"):
                if selected_columns and len(filtered_df) > 0:
                    csv = filtered_df[selected_columns].to_csv(index=False)
                    st.download_button(
                        label="Télécharger le fichier CSV",
                        data=csv,
                        file_name=f"taxi_rl_resultats_filtres_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Aucune donnée à télécharger")

        with download_col2:
            if st.button("📥 Télécharger Tout en JSON"):
                json_data = json.dumps(
                    st.session_state.results_history, indent=2, default=str
                )
                st.download_button(
                    label="Télécharger le fichier JSON",
                    data=json_data,
                    file_name=f"taxi_rl_tous_resultats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

        with download_col3:
            if st.button("📊 Générer un Rapport de Synthèse"):
                summary_report = data_manager.create_summary_report(
                    st.session_state.results_history
                )

                if summary_report:
                    st.subheader("Rapport de Synthèse")

                    report_text = f"""
# Rapport de Synthèse des Expériences RL Taxi Driver
Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Vue d'ensemble
- **Expériences Totales** : {summary_report['total_experiments']}
- **Algorithmes Testés** : {', '.join(summary_report['algorithms_tested'])}
- **Meilleur Algorithme** : {summary_report.get('best_algorithm', 'N/A')}

## Performance Moyenne
- **Récompense Moyenne** : {summary_report['average_performance']['mean_reward']:.3f}
- **Étapes Moyennes** : {summary_report['average_performance']['mean_steps']:.1f}
- **Taux de Réussite** : {summary_report['average_performance']['win_rate']:.2%}

## Plages de Performance
- **Plage de Récompense** : {summary_report['performance_ranges']['reward_range'][0]:.3f} à {summary_report['performance_ranges']['reward_range'][1]:.3f}
- **Plage d'Étapes** : {summary_report['performance_ranges']['steps_range'][0]:.1f} à {summary_report['performance_ranges']['steps_range'][1]:.1f}
- **Plage Taux de Réussite** : {summary_report['performance_ranges']['win_rate_range'][0]:.2%} à {summary_report['performance_ranges']['win_rate_range'][1]:.2%}
                    """

                    st.markdown(report_text)

                    st.download_button(
                        label="📄 Télécharger le Rapport",
                        data=report_text,
                        file_name=f"taxi_rl_rapport_synthese_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                    )

        # Data management actions
        st.subheader("Gestion des Données")

        manage_col1, manage_col2, manage_col3 = st.columns(3)

        with manage_col1:
            if st.button("🗑️ Effacer Tout l'Historique", type="secondary"):
                if st.button("⚠️ Confirmer l'Effacement", type="secondary"):
                    st.session_state.results_history = []
                    st.session_state.training_history = []
                    st.session_state.comparison_data = {}
                    VisualizationComponents.display_success_message(
                        "✅ Tout l'historique effacé !"
                    )
                    st.experimental_rerun()

        with manage_col2:
            if st.button("🔄 Réinitialiser la Session"):
                for key in list(st.session_state.keys()):
                    if key not in ["env", "n_actions", "n_states"]:  # Keep environment
                        del st.session_state[key]
                VisualizationComponents.display_success_message(
                    "✅ Session réinitialisée !"
                )
                st.experimental_rerun()

        with manage_col3:
            if st.button("💾 Sauvegarde Auto de la Session Actuelle"):
                # Save current session data
                session_data = {
                    "results_history": st.session_state.results_history,
                    "training_history": st.session_state.training_history,
                    "comparison_data": st.session_state.comparison_data,
                    "timestamp": datetime.now().isoformat(),
                }

                filename = f"sauvegarde_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                json_str = json.dumps(session_data, indent=2, default=str)

                st.download_button(
                    label="💾 Télécharger la Sauvegarde de Session",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                )

    else:
        st.info(
            "Aucun résultat disponible pour le moment. Lancez quelques expériences pour voir les résultats ici !"
        )

        st.subheader("Pour Commencer")
        getting_started_tips = [
            "🔬 **Lancer des Expériences** : Commencez par le Test d'Algorithme Unique pour générer vos premiers résultats",
            "📊 **Comparer les Algorithmes** : Utilisez la Comparaison d'Algorithmes pour évaluer plusieurs approches",
            "🎯 **Optimiser les Paramètres** : Essayez l'Optimisation d'Hyperparamètres pour de meilleures performances",
            "📈 **Analyser les Résultats** : Utilisez l'Analyse Avancée pour des aperçus détaillés",
            "💾 **Sauvegarder votre Travail** : Les résultats sont automatiquement suivis dans votre session",
        ]

        for tip in getting_started_tips:
            st.markdown(f"- {tip}")

    # Load saved results section
    st.subheader("Charger des Résultats Précédents")

    uploaded_file = st.file_uploader(
        "Choisissez un fichier de résultats",
        type=["json", "csv"],
        help="Téléchargez un fichier de résultats précédemment sauvegardé pour continuer votre analyse",
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
                    st.error("Format JSON non reconnu")

            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.session_state.results_history.extend(df.to_dict("records"))

            VisualizationComponents.display_success_message(
                "✅ Résultats chargés avec succès !"
            )
            st.experimental_rerun()

        except Exception as e:
            VisualizationComponents.display_error_message(
                f"Erreur lors du chargement du fichier : {e}"
            )

    # Show saved files in the results directory
    saved_files = data_manager.get_saved_files()
    if saved_files:
        st.subheader("Fichiers Précédemment Sauvegardés")

        st.write("📁 **Fichiers dans le répertoire de résultats :**")
        for file_path in saved_files[:10]:  # Show last 10 files
            file_name = file_path.split("/")[-1]
            file_size = (
                f"{os.path.getsize(file_path) / 1024:.1f} Ko"
                if os.path.exists(file_path)
                else "Inconnu"
            )
            file_date = (
                datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
                    "%Y-%m-%d %H:%M"
                )
                if os.path.exists(file_path)
                else "Inconnu"
            )

            st.write(f"- **{file_name}** ({file_size}) - {file_date}")

        if len(saved_files) > 10:
            st.write(f"... et {len(saved_files) - 10} fichiers de plus")


# Import os for file operations
import os
