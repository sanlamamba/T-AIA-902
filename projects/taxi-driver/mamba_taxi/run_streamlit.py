#!/usr/bin/env python3
"""
Streamlit Application Runner
Run the modular Taxi Driver RL Interface
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys

    # Set the path to the main app file
    app_file = os.path.join(current_dir, "app.py")

    # Replace sys.argv to run the correct file
    sys.argv = ["streamlit", "run", app_file] + sys.argv[1:]

    # Run streamlit
    stcli.main()
