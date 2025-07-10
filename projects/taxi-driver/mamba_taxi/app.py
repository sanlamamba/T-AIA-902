import streamlit as st

from streamlit_ui.config import setup_page_config, load_custom_css, APP_TITLE, PAGES
from streamlit_ui.session_manager import SessionStateManager

from streamlit_ui.pages.home import render_home_page
from streamlit_ui.pages.single_algorithm import render_single_algorithm_page
from streamlit_ui.pages.algorithm_comparison import render_algorithm_comparison_page
from streamlit_ui.pages.hyperparameter_optimization import (
    render_hyperparameter_optimization_page,
)
from streamlit_ui.pages.advanced_analysis import render_advanced_analysis_page
from streamlit_ui.pages.results_management import render_results_management_page


def main():
    # Setup page configuration
    setup_page_config()
    load_custom_css()

    # Initialize session state
    session_manager = SessionStateManager()

    # Main header
    st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisir une page", PAGES)

    st.session_state.current_page = page

    with st.sidebar:
        st.markdown("---")
        st.subheader("Statistiques Rapides")

        if st.session_state.results_history:
            st.metric("Expériences Totales", len(st.session_state.results_history))

            import pandas as pd

            df = pd.DataFrame(st.session_state.results_history)
            unique_algos = df["algorithm"].nunique()
            st.metric("Algorithmes Testés", unique_algos)

            if "efficiency_score" in df.columns:
                best_algo = df.loc[df["efficiency_score"].idxmax(), "algorithm"]
                st.metric("Meilleur Algorithme", best_algo)
        else:
            st.info("Aucune expérience pour le moment")

        st.markdown("---")
        st.markdown(
            """
        <small>
        💡 **Conseils :**
        - Commencez par le Test d'Algorithme Unique
        - Comparez plusieurs algorithmes
        - Optimisez les hyperparamètres
        - Analysez les résultats statistiquement
        </small>
        """,
            unsafe_allow_html=True,
        )

    try:
        if page == "🏠 Accueil":
            render_home_page()
        elif page == "🔧 Test d'Algorithme Unique":
            render_single_algorithm_page()
        elif page == "📊 Comparaison d'Algorithmes":
            render_algorithm_comparison_page()
        elif page == "🎯 Optimisation d'Hyperparamètres":
            render_hyperparameter_optimization_page()
        elif page == "📈 Analyse Avancée":
            render_advanced_analysis_page()
        elif page == "💾 Gestion des Résultats":
            render_results_management_page()
        else:
            st.error(f"Page inconnue : {page}")

    except Exception as e:
        st.error(f"Une erreur s'est produite lors du rendu de la page : {str(e)}")
        st.exception(e)

        st.subheader("Options de Récupération")
        if st.button("🔄 Réinitialiser la Session"):
            for key in list(st.session_state.keys()):
                if key not in ["env", "n_actions", "n_states"]:
                    del st.session_state[key]
            st.experimental_rerun()

        if st.button("🏠 Aller à l'Accueil"):
            st.session_state.current_page = "🏠 Accueil"
            st.experimental_rerun()


if __name__ == "__main__":
    main()
