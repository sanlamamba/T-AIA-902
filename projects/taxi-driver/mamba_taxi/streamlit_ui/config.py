import streamlit as st


def setup_page_config():
    st.set_page_config(
        page_title="🚕 Interface RL Chauffeur de Taxi",
        page_icon="🚕",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_custom_css():
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


APP_TITLE = "🚕 Interface Interactive RL Chauffeur de Taxi"
PAGES = [
    "🏠 Accueil",
    "🔧 Test d'Algorithme Unique",
    "📊 Comparaison d'Algorithmes",
    "🎯 Optimisation d'Hyperparamètres",
    "📈 Analyse Avancée",
    "💾 Gestion des Résultats",
]

ALGORITHMS = ["BruteForce", "Q-Learning", "SARSA", "DQN"]

DEFAULT_PARAMS = {
    "alpha": {"min": 0.01, "max": 1.0, "default": 0.15, "step": 0.01},
    "gamma": {"min": 0.8, "max": 0.999, "default": 0.99, "step": 0.001},
    "epsilon": {"min": 0.1, "max": 1.0, "default": 1.0, "step": 0.1},
    "epsilon_decay": {"min": 0.99, "max": 0.9999, "default": 0.995, "step": 0.0001},
    "epsilon_min": {"min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001},
    "memory_size": {"min": 1000, "max": 50000, "default": 10000},
    "batch_size": {"min": 16, "max": 128, "default": 32},
    "train_episodes": {"min": 1, "max": 100000, "default": 1000},
    "test_episodes": {"min": 1, "max": 1000, "default": 100},
}
