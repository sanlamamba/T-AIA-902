import streamlit as st
import pandas as pd


def render_home_page():
    st.markdown(
        """
    ## Bienvenue dans l'Interface RL Chauffeur de Taxi !
    
    Cette interface interactive vous permet de :
    - Tester des algorithmes individuels avec des paramètres personnalisés
    - Comparer plusieurs algorithmes côte à côte
    - Optimiser automatiquement les hyperparamètres
    - Effectuer une analyse statistique avancée
    - Sauvegarder et gérer vos résultats
    
    ### Démarrage Rapide :
    1. Allez dans **Test d'Algorithme Unique** pour tester des algorithmes individuels
    2. Utilisez **Comparaison d'Algorithmes** pour comparer plusieurs algorithmes
    3. Essayez **Optimisation d'Hyperparamètres** pour un réglage automatisé
    4. Consultez **Analyse Avancée** pour des statistiques détaillées
    
    ### Algorithmes Disponibles :
    - **BruteForce** : Ligne de base aléatoire (aucun entraînement requis)
    - **Q-Learning** : Apprentissage par différence temporelle hors-politique
    - **SARSA** : Apprentissage par différence temporelle sur-politique
    - **DQN** : Réseau Q Profond avec replay d'expérience
    
    ### Fonctionnalités :
    - 🚀 Visualisation du progrès d'entraînement en temps réel
    - 📊 Graphiques de comparaison interactifs et diagrammes radar
    - 🎯 Optimisation automatisée des hyperparamètres
    - 📈 Analyse statistique avancée avec intervalles de confiance
    - 💾 Export des résultats en formats CSV/JSON
    - 🔄 Gestion de l'état de session pour un flux de travail fluide
    """
    )

    # Quick stats if we have results
    if st.session_state.results_history:
        st.subheader("Résumé des Résultats Récents")
        df = pd.DataFrame(st.session_state.results_history[-5:])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Expériences Totales", len(st.session_state.results_history))

        with col2:
            unique_algos = df["algorithm"].nunique()
            st.metric("Algorithmes Testés", unique_algos)

        with col3:
            best_algo = (
                df.loc[df["efficiency_score"].idxmax(), "algorithm"]
                if "efficiency_score" in df.columns
                else "N/A"
            )
            st.metric("Meilleur Algorithme", best_algo)

        with col4:
            avg_win_rate = df["win_rate"].mean() if "win_rate" in df.columns else 0
            st.metric("Taux de Réussite Moyen", f"{avg_win_rate:.1%}")

        st.subheader("Résultats Récents")
        display_cols = ["algorithm", "mean_reward", "win_rate", "efficiency_score"]
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True)

        if len(df) > 1:
            st.subheader("Aperçu des Performances")
            chart_data = df.set_index("algorithm")["efficiency_score"]
            st.bar_chart(chart_data)

    else:
        st.info(
            "👋 Bienvenue ! Commencez par tester un algorithme pour voir vos résultats ici."
        )

        st.subheader("Conseils pour Débuter")

        tips = [
            "🔧 **Commencez Simple** : Commencez par la page Test d'Algorithme Unique pour comprendre les performances de chaque algorithme",
            "📊 **Comparez** : Utilisez la Comparaison d'Algorithmes pour voir quel algorithme fonctionne le mieux selon vos besoins",
            "🎯 **Optimisez** : Essayez l'Optimisation d'Hyperparamètres pour trouver automatiquement les meilleurs réglages",
            "📈 **Analysez** : Utilisez l'Analyse Avancée pour des insights statistiques détaillés",
            "💾 **Sauvegardez** : N'oubliez pas de sauvegarder vos résultats pour référence future",
        ]

        for tip in tips:
            st.markdown(f"- {tip}")
