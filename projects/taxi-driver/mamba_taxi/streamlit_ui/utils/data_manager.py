"""
Data Management Utilities for saving/loading results
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class DataManager:
    """Handles saving and loading of experimental results"""

    def __init__(self, results_dir: str = "streamlit_results"):
        self.results_dir = results_dir
        self._ensure_results_dir()

    def _ensure_results_dir(self):
        """Create results directory if it doesn't exist"""
        os.makedirs(self.results_dir, exist_ok=True)

    def save_results(
        self,
        results: List[Dict],
        training_data: Dict,
        filename_prefix: str = "taxi_rl_results",
    ) -> Tuple[str, Optional[str]]:
        """
        Save results and create download links

        Args:
            results: List of result dictionaries
            training_data: Training data dictionary
            filename_prefix: Prefix for the filename

        Returns:
            Tuple of (json_filename, csv_filename)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_filename = f"{self.results_dir}/{filename_prefix}_{timestamp}.json"
        save_data = {
            "timestamp": timestamp,
            "results": results,
            "training_data": training_data,
            "metadata": {
                "total_algorithms": len(results),
                "best_performer": (
                    max(results, key=lambda x: x.get("efficiency_score", 0))[
                        "algorithm"
                    ]
                    if results
                    else None
                ),
            },
        }

        with open(json_filename, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        # Save CSV summary
        csv_filename = None
        if results:
            df = pd.DataFrame(results)
            csv_filename = f"{self.results_dir}/{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)

        return json_filename, csv_filename

    def load_results_from_json(self, file_path: str) -> Dict:
        """Load results from JSON file"""
        with open(file_path, "r") as f:
            return json.load(f)

    def load_results_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load results from CSV file"""
        return pd.read_csv(file_path)

    def get_saved_files(self) -> List[str]:
        """Get list of saved result files"""
        if not os.path.exists(self.results_dir):
            return []

        files = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith((".json", ".csv")):
                files.append(os.path.join(self.results_dir, filename))

        return sorted(files, key=os.path.getmtime, reverse=True)

    def export_to_csv(self, results: List[Dict]) -> str:
        """Export results to CSV string"""
        df = pd.DataFrame(results)
        return df.to_csv(index=False)

    def export_to_json(self, results: List[Dict]) -> str:
        """Export results to JSON string"""
        return json.dumps(results, indent=2, default=str)

    def create_summary_report(self, results: List[Dict]) -> Dict:
        """Create a summary report from results"""
        if not results:
            return {}

        df = pd.DataFrame(results)

        summary = {
            "total_experiments": len(results),
            "algorithms_tested": df["algorithm"].unique().tolist(),
            "best_algorithm": (
                df.loc[df["efficiency_score"].idxmax(), "algorithm"]
                if "efficiency_score" in df.columns
                else None
            ),
            "average_performance": {
                "mean_reward": df["mean_reward"].mean(),
                "mean_steps": df["mean_steps"].mean(),
                "win_rate": df["win_rate"].mean(),
            },
            "performance_ranges": {
                "reward_range": [df["mean_reward"].min(), df["mean_reward"].max()],
                "steps_range": [df["mean_steps"].min(), df["mean_steps"].max()],
                "win_rate_range": [df["win_rate"].min(), df["win_rate"].max()],
            },
        }

        return summary
