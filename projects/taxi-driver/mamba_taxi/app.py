"""
Main Streamlit Application
Modular Taxi Driver RL Interface
"""

import streamlit as st

# Import configuration and setup
from streamlit_ui.config import setup_page_config, load_custom_css, APP_TITLE, PAGES
from streamlit_ui.session_manager import SessionStateManager

# Import page components
from streamlit_ui.pages.home import render_home_page
from streamlit_ui.pages.single_algorithm import render_single_algorithm_page
from streamlit_ui.pages.algorithm_comparison import render_algorithm_comparison_page
from streamlit_ui.pages.hyperparameter_optimization import (
    render_hyperparameter_optimization_page,
)
from streamlit_ui.pages.advanced_analysis import render_advanced_analysis_page
from streamlit_ui.pages.results_management import render_results_management_page


def main():
    """Main application entry point"""
    # Setup page configuration
    setup_page_config()
    load_custom_css()

    # Initialize session state
    session_manager = SessionStateManager()

    # Main header
    st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", PAGES)

    # Update current page in session state
    st.session_state.current_page = page

    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.subheader("Quick Stats")

        if st.session_state.results_history:
            st.metric("Total Experiments", len(st.session_state.results_history))

            import pandas as pd

            df = pd.DataFrame(st.session_state.results_history)
            unique_algos = df["algorithm"].nunique()
            st.metric("Algorithms Tested", unique_algos)

            if "efficiency_score" in df.columns:
                best_algo = df.loc[df["efficiency_score"].idxmax(), "algorithm"]
                st.metric("Best Algorithm", best_algo)
        else:
            st.info("No experiments yet")

        st.markdown("---")
        st.markdown(
            """
        <small>
        💡 **Tips:**
        - Start with Single Algorithm Testing
        - Compare multiple algorithms
        - Optimize hyperparameters
        - Analyze results statistically
        </small>
        """,
            unsafe_allow_html=True,
        )

    # Route to appropriate page
    try:
        if page == "🏠 Home":
            render_home_page()
        elif page == "🔧 Single Algorithm Testing":
            render_single_algorithm_page()
        elif page == "📊 Algorithm Comparison":
            render_algorithm_comparison_page()
        elif page == "🎯 Hyperparameter Optimization":
            render_hyperparameter_optimization_page()
        elif page == "📈 Advanced Analysis":
            render_advanced_analysis_page()
        elif page == "💾 Results Management":
            render_results_management_page()
        else:
            st.error(f"Unknown page: {page}")

    except Exception as e:
        st.error(f"An error occurred while rendering the page: {str(e)}")
        st.exception(e)

        # Provide recovery options
        st.subheader("Recovery Options")
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                if key not in ["env", "n_actions", "n_states"]:
                    del st.session_state[key]
            st.experimental_rerun()

        if st.button("🏠 Go to Home"):
            st.session_state.current_page = "🏠 Home"
            st.experimental_rerun()


if __name__ == "__main__":
    main()
