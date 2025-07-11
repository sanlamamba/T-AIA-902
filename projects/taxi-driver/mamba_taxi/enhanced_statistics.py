import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for RL agents"""

    def __init__(self):
        self.analysis_results = {}
        self.comparative_results = {}

    def comprehensive_analysis(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        analysis = {
            "descriptive_stats": self._descriptive_statistics(results),
            "distribution_analysis": self._distribution_analysis(results),
            "correlation_analysis": self._correlation_analysis(results, training_data),
            "performance_clustering": self._performance_clustering(results),
            "learning_dynamics": self._learning_dynamics_analysis(training_data),
            "convergence_analysis": self._convergence_analysis(training_data),
            "stability_analysis": self._stability_analysis(results, training_data),
            "efficiency_analysis": self._efficiency_analysis(results, training_data),
            "outlier_analysis": self._outlier_analysis(results),
            "confidence_intervals": self._confidence_intervals(results),
            "hypothesis_testing": self._hypothesis_testing(results),
            "effect_size_analysis": self._effect_size_analysis(results),
            "meta_analysis": self._meta_analysis(results, training_data),
        }

        return analysis

    def _descriptive_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Comprehensive descriptive statistics"""
        metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]
        stats_dict = {}

        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                stats_dict[metric] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "var": np.var(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                    "iqr": np.percentile(values, 75) - np.percentile(values, 25),
                    "skewness": stats.skew(values),
                    "kurtosis": stats.kurtosis(values),
                    "cv": (
                        np.std(values) / np.mean(values)
                        if np.mean(values) != 0
                        else float("inf")
                    ),
                    "mad": np.median(
                        np.abs(values - np.median(values))
                    ),  # Median Absolute Deviation
                    "range": np.max(values) - np.min(values),
                    "geometric_mean": (
                        stats.gmean(np.array(values) + 1) - 1
                        if all(v >= 0 for v in values)
                        else None
                    ),
                    "harmonic_mean": (
                        stats.hmean(np.array(values) + 1) - 1
                        if all(v > 0 for v in values)
                        else None
                    ),
                }

        return stats_dict

    def _distribution_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze data distributions"""
        metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]
        dist_analysis = {}

        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if len(values) >= 3:
                # Normality tests
                shapiro_stat, shapiro_p = stats.shapiro(values)
                ks_stat, ks_p = stats.kstest(
                    values, "norm", args=(np.mean(values), np.std(values))
                )

                # Distribution fitting
                distributions = ["norm", "gamma", "beta", "lognorm", "exponpow"]
                best_dist = self._fit_best_distribution(values, distributions)

                dist_analysis[metric] = {
                    "shapiro_test": {
                        "statistic": shapiro_stat,
                        "p_value": shapiro_p,
                        "is_normal": shapiro_p > 0.05,
                    },
                    "ks_test": {"statistic": ks_stat, "p_value": ks_p},
                    "best_distribution": best_dist,
                    "percentiles": {
                        f"p{i}": np.percentile(values, i)
                        for i in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    },
                    "histogram_bins": np.histogram(values, bins="auto")[1].tolist(),
                    "histogram_counts": np.histogram(values, bins="auto")[0].tolist(),
                }

        return dist_analysis

    def _correlation_analysis(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze correlations between metrics"""
        # Create DataFrame from results
        df_results = pd.DataFrame(results)

        # Correlation matrix for performance metrics
        numeric_cols = df_results.select_dtypes(include=[np.number]).columns
        corr_matrix = df_results[numeric_cols].corr()

        # Partial correlations
        partial_corr = self._partial_correlation_matrix(df_results[numeric_cols])

        # Cross-correlation with training metrics
        cross_corr = {}
        if training_data:
            for i, train_data in enumerate(training_data):
                if "rewards" in train_data and i < len(results):
                    rewards = train_data["rewards"]
                    # Correlation between final performance and training progression
                    if len(rewards) > 10:
                        final_reward = results[i].get("mean_reward", 0)
                        training_trend = np.corrcoef(range(len(rewards)), rewards)[0, 1]
                        cross_corr[f"agent_{i}"] = {
                            "final_vs_trend": training_trend,
                            "final_reward": final_reward,
                        }

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "partial_correlations": partial_corr,
            "cross_correlations": cross_corr,
            "significant_correlations": self._find_significant_correlations(
                corr_matrix
            ),
        }

    def _performance_clustering(self, results: List[Dict]) -> Dict[str, Any]:
        """Cluster agents based on performance characteristics"""
        if len(results) < 3:
            return {"error": "Insufficient data for clustering"}

        # Prepare feature matrix
        features = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]
        feature_matrix = []

        for result in results:
            row = [result.get(feature, 0) for feature in features]
            feature_matrix.append(row)

        feature_matrix = np.array(feature_matrix)

        # Standardize features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Determine optimal number of clusters
        max_k = min(8, len(results) - 1)
        if max_k >= 2:
            inertias = []
            silhouette_scores = []

            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(feature_matrix_scaled)
                inertias.append(kmeans.inertia_)

                from sklearn.metrics import silhouette_score

                sil_score = silhouette_score(feature_matrix_scaled, labels)
                silhouette_scores.append(sil_score)

            # Choose optimal k
            optimal_k = np.argmax(silhouette_scores) + 2

            # Final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

            # PCA for visualization
            pca = PCA(n_components=2)
            feature_2d = pca.fit_transform(feature_matrix_scaled)

            return {
                "n_clusters": optimal_k,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
                "pca_components": feature_2d.tolist(),
                "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
                "feature_importance": self._calculate_feature_importance(
                    feature_matrix_scaled, cluster_labels
                ),
            }

        return {"error": "Insufficient data for meaningful clustering"}

    def _learning_dynamics_analysis(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Analyze learning dynamics across agents"""
        if not training_data:
            return {}

        dynamics = {}

        for i, data in enumerate(training_data):
            if "rewards" in data:
                rewards = np.array(data["rewards"])
                steps = np.array(data.get("steps", []))

                agent_dynamics = {
                    "learning_rate": self._calculate_learning_rate(rewards),
                    "convergence_speed": self._calculate_convergence_speed(rewards),
                    "stability_index": self._calculate_stability_index(rewards),
                    "exploration_efficiency": self._calculate_exploration_efficiency(
                        rewards, steps
                    ),
                    "learning_phases": self._identify_learning_phases(rewards),
                    "plateau_detection": self._detect_plateaus(rewards),
                    "breakthrough_episodes": self._detect_breakthroughs(rewards),
                    "forgetting_analysis": self._analyze_forgetting(rewards),
                }

                dynamics[f"agent_{i}"] = agent_dynamics

        return dynamics

    def _convergence_analysis(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Detailed convergence analysis"""
        convergence_stats = {}

        for i, data in enumerate(training_data):
            if "rewards" in data:
                rewards = np.array(data["rewards"])

                # Multiple convergence criteria
                convergence_stats[f"agent_{i}"] = {
                    "mean_convergence": self._detect_mean_convergence(rewards),
                    "variance_convergence": self._detect_variance_convergence(rewards),
                    "trend_convergence": self._detect_trend_convergence(rewards),
                    "statistical_convergence": self._statistical_convergence_test(
                        rewards
                    ),
                    "convergence_confidence": self._convergence_confidence(rewards),
                }

        return convergence_stats

    def _stability_analysis(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze performance stability"""
        stability_metrics = {}

        # Performance stability across episodes
        for i, data in enumerate(training_data):
            if "rewards" in data:
                rewards = np.array(data["rewards"])

                stability_metrics[f"agent_{i}"] = {
                    "coefficient_of_variation": (
                        np.std(rewards) / np.mean(rewards)
                        if np.mean(rewards) != 0
                        else float("inf")
                    ),
                    "stability_ratio": self._calculate_stability_ratio(rewards),
                    "volatility_index": self._calculate_volatility_index(rewards),
                    "consistency_score": self._calculate_consistency_score(rewards),
                    "robustness_measure": self._calculate_robustness_measure(rewards),
                }

        return stability_metrics

    def _efficiency_analysis(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze learning and performance efficiency"""
        efficiency_metrics = {}

        for i, (result, train_data) in enumerate(zip(results, training_data)):
            if "rewards" in train_data:
                rewards = np.array(train_data["rewards"])
                steps = np.array(train_data.get("steps", []))

                efficiency_metrics[f"agent_{i}"] = {
                    "sample_efficiency": self._calculate_sample_efficiency(rewards),
                    "computational_efficiency": self._calculate_computational_efficiency(
                        train_data
                    ),
                    "data_efficiency": self._calculate_data_efficiency(
                        rewards, len(rewards)
                    ),
                    "asymptotic_performance": self._estimate_asymptotic_performance(
                        rewards
                    ),
                    "learning_efficiency_curve": self._learning_efficiency_curve(
                        rewards
                    ),
                    "improvement_rate": self._calculate_improvement_rate(rewards),
                }

        return efficiency_metrics

    def _outlier_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Detect and analyze outliers"""
        outlier_analysis = {}
        metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]

        for metric in metrics:
            values = np.array([r.get(metric, 0) for r in results if metric in r])
            if len(values) > 3:
                # Multiple outlier detection methods
                outlier_analysis[metric] = {
                    "iqr_outliers": self._iqr_outliers(values),
                    "zscore_outliers": self._zscore_outliers(values),
                    "modified_zscore_outliers": self._modified_zscore_outliers(values),
                    "isolation_forest_outliers": self._isolation_forest_outliers(
                        values.reshape(-1, 1)
                    ),
                }

        return outlier_analysis

    def _confidence_intervals(
        self, results: List[Dict], confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for all metrics"""
        confidence_intervals = {}
        metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]

        alpha = 1 - confidence_level

        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if len(values) > 1:
                mean = np.mean(values)
                sem = stats.sem(values)  # Standard error of the mean

                # t-distribution for small samples
                if len(values) < 30:
                    t_critical = stats.t.ppf(1 - alpha / 2, len(values) - 1)
                    margin_error = t_critical * sem
                else:
                    z_critical = stats.norm.ppf(1 - alpha / 2)
                    margin_error = z_critical * sem

                confidence_intervals[metric] = {
                    "mean": mean,
                    "lower_bound": mean - margin_error,
                    "upper_bound": mean + margin_error,
                    "margin_error": margin_error,
                    "confidence_level": confidence_level,
                }

        return confidence_intervals

    def _hypothesis_testing(self, results: List[Dict]) -> Dict[str, Any]:
        """Perform various hypothesis tests"""
        if len(results) < 2:
            return {"error": "Insufficient data for hypothesis testing"}

        # Separate results by algorithm
        algo_groups = {}
        for result in results:
            algo = result.get("algorithm", "unknown")
            if algo not in algo_groups:
                algo_groups[algo] = []
            algo_groups[algo].append(result)

        if len(algo_groups) < 2:
            return {"error": "Need at least 2 different algorithms for comparison"}

        hypothesis_tests = {}
        metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]

        algos = list(algo_groups.keys())

        for i, algo1 in enumerate(algos):
            for algo2 in algos[i + 1 :]:
                comparison_key = f"{algo1}_vs_{algo2}"
                hypothesis_tests[comparison_key] = {}

                for metric in metrics:
                    values1 = [
                        r.get(metric, 0) for r in algo_groups[algo1] if metric in r
                    ]
                    values2 = [
                        r.get(metric, 0) for r in algo_groups[algo2] if metric in r
                    ]

                    if len(values1) > 0 and len(values2) > 0:
                        # Multiple statistical tests
                        tests = self._perform_comparison_tests(values1, values2)
                        hypothesis_tests[comparison_key][metric] = tests

        return hypothesis_tests

    def _effect_size_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate effect sizes for comparisons"""
        # Similar structure to hypothesis testing but focusing on effect sizes
        algo_groups = {}
        for result in results:
            algo = result.get("algorithm", "unknown")
            if algo not in algo_groups:
                algo_groups[algo] = []
            algo_groups[algo].append(result)

        effect_sizes = {}
        metrics = ["mean_reward", "mean_steps", "win_rate", "efficiency_score"]
        algos = list(algo_groups.keys())

        for i, algo1 in enumerate(algos):
            for algo2 in algos[i + 1 :]:
                comparison_key = f"{algo1}_vs_{algo2}"
                effect_sizes[comparison_key] = {}

                for metric in metrics:
                    values1 = [
                        r.get(metric, 0) for r in algo_groups[algo1] if metric in r
                    ]
                    values2 = [
                        r.get(metric, 0) for r in algo_groups[algo2] if metric in r
                    ]

                    if len(values1) > 1 and len(values2) > 1:
                        effect_sizes[comparison_key][metric] = (
                            self._calculate_effect_sizes(values1, values2)
                        )

        return effect_sizes

    def _meta_analysis(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Perform meta-analysis across all experiments"""
        meta_stats = {
            "overall_performance": self._overall_performance_summary(results),
            "learning_patterns": self._identify_learning_patterns(training_data),
            "success_factors": self._identify_success_factors(results, training_data),
            "failure_modes": self._identify_failure_modes(results, training_data),
            "generalization_analysis": self._generalization_analysis(results),
            "robustness_summary": self._robustness_summary(results, training_data),
        }

        return meta_stats

    # Helper methods for statistical calculations
    def _fit_best_distribution(
        self, data: List[float], distributions: List[str]
    ) -> Dict[str, Any]:
        """Fit multiple distributions and return the best one"""
        best_dist = None
        best_aic = float("inf")

        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                aic = 2 * len(params) - 2 * np.sum(dist.logpdf(data, *params))

                if aic < best_aic:
                    best_aic = aic
                    best_dist = {
                        "name": dist_name,
                        "params": params,
                        "aic": aic,
                        "bic": len(params) * np.log(len(data))
                        - 2 * np.sum(dist.logpdf(data, *params)),
                    }
            except:
                continue

        return best_dist or {
            "name": "unknown",
            "params": [],
            "aic": float("inf"),
            "bic": float("inf"),
        }

    def _partial_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate partial correlation matrix"""
        try:
            from sklearn.covariance import GraphicalLassoCV

            # Use GraphicalLasso to estimate partial correlations
            model = GraphicalLassoCV()
            model.fit(df.fillna(0))

            # Convert precision matrix to partial correlations
            precision = model.precision_
            partial_corr = np.zeros_like(precision)

            for i in range(len(precision)):
                for j in range(len(precision)):
                    if i != j:
                        partial_corr[i, j] = -precision[i, j] / np.sqrt(
                            precision[i, i] * precision[j, j]
                        )

            return pd.DataFrame(
                partial_corr, index=df.columns, columns=df.columns
            ).to_dict()
        except:
            return {}

    def _find_significant_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.5
    ) -> List[Dict]:
        """Find statistically significant correlations"""
        significant = []

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        significant.append(
                            {
                                "variable1": col1,
                                "variable2": col2,
                                "correlation": corr_val,
                                "strength": (
                                    "strong" if abs(corr_val) > 0.7 else "moderate"
                                ),
                            }
                        )

        return significant

    def _calculate_learning_rate(self, rewards: np.ndarray) -> float:
        """Calculate learning rate based on reward progression"""
        if len(rewards) < 10:
            return 0.0

        # Fit linear trend to rewards
        x = np.arange(len(rewards))
        slope, _, r_value, _, _ = stats.linregress(x, rewards)

        # Normalize by episode length
        return slope / len(rewards) if len(rewards) > 0 else 0.0

    def _calculate_convergence_speed(self, rewards: np.ndarray) -> int:
        """Calculate how quickly the agent converges"""
        if len(rewards) < 20:
            return len(rewards)

        # Find when moving average stabilizes
        window = min(20, len(rewards) // 4)
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

        # Look for stabilization (low variance in moving average)
        for i in range(len(moving_avg) - window):
            window_var = np.var(moving_avg[i : i + window])
            if window_var < 0.1:  # Threshold for stabilization
                return i + window

        return len(rewards)

    def _calculate_stability_index(self, rewards: np.ndarray) -> float:
        """Calculate stability index based on reward variance"""
        if len(rewards) < 2:
            return 0.0

        # Use coefficient of variation as stability measure
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        if mean_reward == 0:
            return 0.0

        # Return inverse of coefficient of variation (higher = more stable)
        cv = std_reward / abs(mean_reward)
        return 1.0 / (1.0 + cv)

    def _calculate_exploration_efficiency(
        self, rewards: np.ndarray, steps: np.ndarray
    ) -> float:
        """Calculate exploration efficiency"""
        if len(rewards) == 0:
            return 0.0

        # Simple efficiency measure: improvement rate per episode
        if len(rewards) < 2:
            return 0.0

        # Calculate how much reward improves over time
        x = np.arange(len(rewards))
        slope, _, r_value, _, _ = stats.linregress(x, rewards)

        # Normalize by length and return absolute improvement rate
        return abs(slope) if not np.isnan(slope) else 0.0

    def _identify_learning_phases(self, rewards: np.ndarray) -> Dict[str, Any]:
        """Identify different learning phases"""
        if len(rewards) < 10:
            return {"phases": [], "total_phases": 0}

        # Simple phase detection based on trend changes
        window_size = max(10, len(rewards) // 10)
        phases = []

        for i in range(0, len(rewards) - window_size, window_size):
            window_rewards = rewards[i : i + window_size]
            if len(window_rewards) >= 2:
                x = np.arange(len(window_rewards))
                slope, _, r_value, p_value, _ = stats.linregress(x, window_rewards)

                phase_type = "stable"
                if p_value < 0.05:  # Significant trend
                    if slope > 0:
                        phase_type = "improving"
                    else:
                        phase_type = "declining"

                phases.append(
                    {
                        "start_episode": i,
                        "end_episode": i + window_size,
                        "type": phase_type,
                        "slope": slope,
                        "r_value": r_value,
                    }
                )

        return {"phases": phases, "total_phases": len(phases)}

    def _detect_plateaus(self, rewards: np.ndarray) -> List[Dict[str, Any]]:
        """Detect plateau periods in learning"""
        if len(rewards) < 20:
            return []

        plateaus = []
        window_size = max(10, len(rewards) // 20)

        for i in range(window_size, len(rewards) - window_size):
            window_rewards = rewards[i - window_size : i + window_size]

            # Check if variance is low (plateau)
            if np.var(window_rewards) < 0.1:
                plateaus.append(
                    {
                        "start": i - window_size,
                        "end": i + window_size,
                        "mean_reward": np.mean(window_rewards),
                        "variance": np.var(window_rewards),
                    }
                )

        return plateaus

    def _detect_breakthroughs(self, rewards: np.ndarray) -> List[int]:
        """Detect breakthrough episodes (sudden improvements)"""
        if len(rewards) < 10:
            return []

        breakthroughs = []

        # Look for sudden jumps in reward
        for i in range(5, len(rewards) - 5):
            before_mean = np.mean(rewards[i - 5 : i])
            after_mean = np.mean(rewards[i : i + 5])

            # If there's a significant improvement
            if after_mean > before_mean + 2 * np.std(rewards[i - 5 : i]):
                breakthroughs.append(i)

        return breakthroughs

    def _analyze_forgetting(self, rewards: np.ndarray) -> Dict[str, Any]:
        """Analyze catastrophic forgetting patterns"""
        if len(rewards) < 20:
            return {"forgetting_episodes": [], "total_forgetting_events": 0}

        forgetting_episodes = []

        # Look for significant drops in performance
        for i in range(10, len(rewards) - 10):
            before_max = np.max(rewards[i - 10 : i])
            current_reward = rewards[i]

            # If current performance drops significantly from recent maximum
            if current_reward < before_max - 2 * np.std(rewards[i - 10 : i]):
                forgetting_episodes.append(i)

        return {
            "forgetting_episodes": forgetting_episodes,
            "total_forgetting_events": len(forgetting_episodes),
        }

    def _detect_mean_convergence(self, rewards: np.ndarray) -> Dict[str, Any]:
        """Detect convergence based on mean stabilization"""
        if len(rewards) < 50:
            return {"converged": False, "convergence_episode": len(rewards)}

        window_size = 20
        for i in range(window_size, len(rewards) - window_size):
            window1 = rewards[i - window_size : i]
            window2 = rewards[i : i + window_size]

            # Check if means are similar
            if abs(np.mean(window1) - np.mean(window2)) < 0.1:
                return {"converged": True, "convergence_episode": i}

        return {"converged": False, "convergence_episode": len(rewards)}

    def _detect_variance_convergence(self, rewards: np.ndarray) -> Dict[str, Any]:
        """Detect convergence based on variance stabilization"""
        if len(rewards) < 50:
            return {"converged": False, "convergence_episode": len(rewards)}

        window_size = 20
        for i in range(window_size, len(rewards) - window_size):
            window_var = np.var(rewards[i : i + window_size])

            # Check if variance is low enough
            if window_var < 0.05:
                return {"converged": True, "convergence_episode": i}

        return {"converged": False, "convergence_episode": len(rewards)}

    def _detect_trend_convergence(self, rewards: np.ndarray) -> Dict[str, Any]:
        """Detect convergence based on trend stabilization"""
        if len(rewards) < 50:
            return {"converged": False, "convergence_episode": len(rewards)}

        window_size = 20
        for i in range(window_size, len(rewards) - window_size):
            window_rewards = rewards[i : i + window_size]
            x = np.arange(len(window_rewards))

            try:
                slope, _, r_value, p_value, _ = stats.linregress(x, window_rewards)

                # Check if trend is not significant (stable)
                if p_value > 0.05 or abs(slope) < 0.01:
                    return {"converged": True, "convergence_episode": i}
            except:
                continue

        return {"converged": False, "convergence_episode": len(rewards)}

    def _statistical_convergence_test(self, rewards: np.ndarray) -> Dict[str, Any]:
        """Perform statistical test for convergence"""
        if len(rewards) < 50:
            return {"test_statistic": 0, "p_value": 1.0, "converged": False}

        # Use Ljung-Box test for stationarity
        try:
            from scipy.stats import jarque_bera

            # Use a simpler test if Ljung-Box is not available
            window_size = min(30, len(rewards) // 3)
            recent_rewards = rewards[-window_size:]

            # Test for normality as a proxy for convergence
            statistic, p_value = jarque_bera(recent_rewards)

            return {
                "test_statistic": statistic,
                "p_value": p_value,
                "converged": p_value > 0.05,  # If normal, likely converged
            }
        except:
            return {"test_statistic": 0, "p_value": 1.0, "converged": False}

    def _convergence_confidence(self, rewards: np.ndarray) -> float:
        """Calculate confidence in convergence"""
        if len(rewards) < 20:
            return 0.0

        # Combine multiple convergence indicators
        mean_conv = self._detect_mean_convergence(rewards)
        var_conv = self._detect_variance_convergence(rewards)
        trend_conv = self._detect_trend_convergence(rewards)

        confidence_score = 0.0
        if mean_conv["converged"]:
            confidence_score += 0.33
        if var_conv["converged"]:
            confidence_score += 0.33
        if trend_conv["converged"]:
            confidence_score += 0.34

        return confidence_score

    def _calculate_stability_ratio(self, rewards: np.ndarray) -> float:
        """Calculate stability ratio"""
        if len(rewards) < 10:
            return 0.0

        # Ratio of episodes with rewards within one standard deviation of mean
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        stable_episodes = np.sum(np.abs(rewards - mean_reward) <= std_reward)
        return stable_episodes / len(rewards)

    def _calculate_volatility_index(self, rewards: np.ndarray) -> float:
        """Calculate volatility index"""
        if len(rewards) < 2:
            return 0.0

        # Standard deviation of first differences
        diff_rewards = np.diff(rewards)
        return np.std(diff_rewards)

    def _calculate_consistency_score(self, rewards: np.ndarray) -> float:
        """Calculate consistency score"""
        if len(rewards) < 10:
            return 0.0

        # Percentage of episodes that are improving or stable
        improving_or_stable = 0
        for i in range(1, len(rewards)):
            if rewards[i] >= rewards[i - 1] - 0.1:  # Allow small decreases
                improving_or_stable += 1

        return improving_or_stable / (len(rewards) - 1)

    def _calculate_robustness_measure(self, rewards: np.ndarray) -> float:
        """Calculate robustness measure"""
        if len(rewards) < 10:
            return 0.0

        # Robustness = 1 - (range / mean)
        reward_range = np.max(rewards) - np.min(rewards)
        mean_reward = np.mean(rewards)

        if mean_reward == 0:
            return 0.0

        return max(0.0, 1.0 - (reward_range / abs(mean_reward)))

    def _calculate_sample_efficiency(self, rewards: np.ndarray) -> float:
        """Calculate sample efficiency"""
        if len(rewards) < 10:
            return 0.0

        # Episodes needed to reach 80% of final performance
        final_performance = np.mean(rewards[-10:])
        target_performance = 0.8 * final_performance

        for i, reward in enumerate(rewards):
            if reward >= target_performance:
                return 1.0 / (i + 1)  # Inverse of episodes needed

        return 1.0 / len(rewards)  # If never reached, return minimum efficiency

    def _calculate_computational_efficiency(self, train_data: Dict) -> float:
        """Calculate computational efficiency"""
        training_time = train_data.get("training_time", 1.0)
        num_episodes = len(train_data.get("rewards", [1]))

        if training_time == 0:
            return float("inf")

        return num_episodes / training_time  # Episodes per second

    def _calculate_data_efficiency(
        self, rewards: np.ndarray, total_episodes: int
    ) -> float:
        """Calculate data efficiency"""
        if total_episodes == 0:
            return 0.0

        # Final performance divided by total episodes
        final_performance = (
            np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        )
        return final_performance / total_episodes

    def _estimate_asymptotic_performance(self, rewards: np.ndarray) -> float:
        """Estimate asymptotic performance"""
        if len(rewards) < 20:
            return np.mean(rewards) if len(rewards) > 0 else 0.0

        # Use the mean of the last 20% of episodes
        tail_size = max(10, len(rewards) // 5)
        return np.mean(rewards[-tail_size:])

    def _learning_efficiency_curve(self, rewards: np.ndarray) -> List[float]:
        """Calculate learning efficiency curve"""
        if len(rewards) < 2:
            return [0.0]

        efficiency_curve = []
        for i in range(1, len(rewards)):
            current_performance = np.mean(rewards[: i + 1])
            efficiency = current_performance / (i + 1)
            efficiency_curve.append(efficiency)

        return efficiency_curve

    def _calculate_improvement_rate(self, rewards: np.ndarray) -> float:
        """Calculate improvement rate"""
        if len(rewards) < 10:
            return 0.0

        # Compare first 25% with last 25%
        first_quarter = rewards[: len(rewards) // 4]
        last_quarter = rewards[-len(rewards) // 4 :]

        if len(first_quarter) == 0 or len(last_quarter) == 0:
            return 0.0

        first_mean = np.mean(first_quarter)
        last_mean = np.mean(last_quarter)

        if first_mean == 0:
            return float("inf") if last_mean > 0 else 0.0

        return (last_mean - first_mean) / abs(first_mean)

    def _iqr_outliers(self, values: np.ndarray) -> List[int]:
        """Detect outliers using IQR method"""
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)

        return outliers

    def _zscore_outliers(self, values: np.ndarray, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(values))
        outliers = []

        for i, z_score in enumerate(z_scores):
            if z_score > threshold:
                outliers.append(i)

        return outliers

    def _modified_zscore_outliers(
        self, values: np.ndarray, threshold: float = 3.5
    ) -> List[int]:
        """Detect outliers using modified Z-score method"""
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            return []

        modified_z_scores = 0.6745 * (values - median) / mad
        outliers = []

        for i, z_score in enumerate(modified_z_scores):
            if abs(z_score) > threshold:
                outliers.append(i)

        return outliers

    def _isolation_forest_outliers(self, values: np.ndarray) -> List[int]:
        """Detect outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest

            if len(values) < 10:
                return []

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(values.reshape(-1, 1))

            outliers = []
            for i, label in enumerate(outlier_labels):
                if label == -1:  # Outlier
                    outliers.append(i)

            return outliers
        except:
            return []

    def _perform_comparison_tests(
        self, values1: List[float], values2: List[float]
    ) -> Dict[str, Any]:
        """Perform statistical comparison tests"""
        tests = {}

        # Mann-Whitney U test
        try:
            statistic, p_value = stats.mannwhitneyu(
                values1, values2, alternative="two-sided"
            )
            tests["mann_whitney"] = {
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        except:
            tests["mann_whitney"] = {
                "statistic": 0,
                "p_value": 1.0,
                "significant": False,
            }

        # T-test
        try:
            statistic, p_value = stats.ttest_ind(values1, values2)
            tests["t_test"] = {
                "statistic": statistic,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        except:
            tests["t_test"] = {"statistic": 0, "p_value": 1.0, "significant": False}

        return tests

    def _calculate_effect_sizes(
        self, values1: List[float], values2: List[float]
    ) -> Dict[str, float]:
        """Calculate effect sizes"""
        effect_sizes = {}

        # Cohen's d
        try:
            mean1, mean2 = np.mean(values1), np.mean(values2)
            std1, std2 = np.std(values1), np.std(values2)

            pooled_std = np.sqrt(
                ((len(values1) - 1) * std1**2 + (len(values2) - 1) * std2**2)
                / (len(values1) + len(values2) - 2)
            )

            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            effect_sizes["cohens_d"] = cohens_d
        except:
            effect_sizes["cohens_d"] = 0.0

        return effect_sizes

    def _calculate_feature_importance(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        try:
            from sklearn.ensemble import RandomForestClassifier

            if len(np.unique(labels)) < 2:
                return {}

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)

            feature_names = [
                "mean_reward",
                "mean_steps",
                "win_rate",
                "efficiency_score",
            ]
            importance_dict = {}

            for i, importance in enumerate(rf.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = importance

            return importance_dict
        except:
            return {}

    # Placeholder methods for meta-analysis
    def _overall_performance_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Overall performance summary"""
        if not results:
            return {}

        return {
            "total_experiments": len(results),
            "avg_mean_reward": np.mean([r.get("mean_reward", 0) for r in results]),
            "avg_win_rate": np.mean([r.get("win_rate", 0) for r in results]),
            "best_algorithm": max(
                results, key=lambda x: x.get("efficiency_score", 0)
            ).get("algorithm", "unknown"),
        }

    def _identify_learning_patterns(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Identify common learning patterns"""
        return {"patterns_identified": 0, "common_patterns": []}

    def _identify_success_factors(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Identify factors that lead to success"""
        return {"success_factors": [], "correlation_with_success": {}}

    def _identify_failure_modes(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Identify common failure modes"""
        return {"failure_modes": [], "failure_frequency": {}}

    def _generalization_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze generalization capabilities"""
        return {"generalization_score": 0.0, "analysis": "Not implemented"}

    def _robustness_summary(
        self, results: List[Dict], training_data: List[Dict]
    ) -> Dict[str, Any]:
        """Summarize robustness across experiments"""
        return {"robustness_score": 0.0, "summary": "Not implemented"}

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive statistical report"""
        report = []
        report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Add sections for each type of analysis
        for section, data in analysis_results.items():
            report.append(f"{section.upper().replace('_', ' ')}")
            report.append("-" * 40)

            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        report.append(f"{key}: {value:.4f}")
                    elif isinstance(value, dict):
                        report.append(f"{key}:")
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                report.append(f"  {subkey}: {subvalue:.4f}")

            report.append("")

        return "\n".join(report)
