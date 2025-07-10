"""
Session State Management for Streamlit App
"""

import streamlit as st
import gymnasium as gym


class SessionStateManager:
    """Manages Streamlit session state variables"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all required session state variables"""
        # Environment setup
        if "env" not in st.session_state:
            st.session_state.env = gym.make("Taxi-v3", render_mode=None)
            st.session_state.n_actions = st.session_state.env.action_space.n
            st.session_state.n_states = st.session_state.env.observation_space.n

        # Results tracking
        if "results_history" not in st.session_state:
            st.session_state.results_history = []

        if "training_history" not in st.session_state:
            st.session_state.training_history = []

        if "comparison_data" not in st.session_state:
            st.session_state.comparison_data = {}

        # UI state
        if "current_page" not in st.session_state:
            st.session_state.current_page = "🏠 Home"

        if "last_experiment" not in st.session_state:
            st.session_state.last_experiment = None

    def clear_history(self):
        """Clear all experimental history"""
        st.session_state.results_history = []
        st.session_state.training_history = []
        st.session_state.comparison_data = {}
        st.session_state.last_experiment = None

    def add_result(self, result):
        """Add a result to the history"""
        st.session_state.results_history.append(result)

    def add_training_data(self, algorithm, training_data):
        """Add training data to the history"""
        st.session_state.training_history.append({algorithm: training_data})

    def set_comparison_data(self, results, training_data):
        """Set comparison data"""
        from datetime import datetime

        st.session_state.comparison_data = {
            "results": results,
            "training_data": training_data,
            "timestamp": datetime.now().isoformat(),
        }

    def get_environment_info(self):
        """Get environment information"""
        return {
            "n_actions": st.session_state.n_actions,
            "n_states": st.session_state.n_states,
            "env": st.session_state.env,
        }
