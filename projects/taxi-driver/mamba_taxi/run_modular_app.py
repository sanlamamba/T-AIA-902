"""
Streamlit Application Runner
Run the modular Taxi Driver RL Interface
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys

    app_file = os.path.join(current_dir, "app.py")

    sys.argv = ["streamlit", "run", app_file] + sys.argv[1:]

    stcli.main()
