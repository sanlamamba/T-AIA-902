import streamlit as st
import gymnasium as gym


class SessionStateManager:

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        if "env" not in st.session_state:
            st.session_state.env = gym.make("Taxi-v3", render_mode=None)
            st.session_state.n_actions = st.session_state.env.action_space.n
            st.session_state.n_states = st.session_state.env.observation_space.n

        if "results_history" not in st.session_state:
            st.session_state.results_history = []

        if "training_history" not in st.session_state:
            st.session_state.training_history = []

        if "comparison_data" not in st.session_state:
            st.session_state.comparison_data = {}

        if "current_page" not in st.session_state:
            st.session_state.current_page = "🏠 Home"

        if "last_experiment" not in st.session_state:
            st.session_state.last_experiment = None

    def clear_history(self):
        st.session_state.results_history = []
        st.session_state.training_history = []
        st.session_state.comparison_data = {}
        st.session_state.last_experiment = None

    def add_result(self, result):
        st.session_state.results_history.append(result)

    def add_training_data(self, algorithm, training_data):
        st.session_state.training_history.append({algorithm: training_data})

    def set_comparison_data(self, results, training_data):
        from datetime import datetime

        st.session_state.comparison_data = {
            "results": results,
            "training_data": training_data,
            "timestamp": datetime.now().isoformat(),
        }

    def get_environment_info(self):
        return {
            "n_actions": st.session_state.n_actions,
            "n_states": st.session_state.n_states,
            "env": st.session_state.env,
        }
