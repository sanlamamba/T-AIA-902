# 🚕 Taxi Driver RL - Streamlit Interface Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Interface
```bash
# Option 1: Use the launcher script
python run_streamlit.py

# Option 2: Direct streamlit command
streamlit run streamlit_app.py
```

### 3. Open in Browser
The interface will automatically open at `http://localhost:8501`

## Interface Features

### 🏠 Home Page
- Project overview and quick start guide
- Recent results summary
- Navigation to different features

### 🔧 Single Algorithm Testing
- **Interactive Parameter Tuning**: Adjust learning rates, discount factors, exploration parameters
- **Real-time Training Progress**: Watch training progress with live updates
- **Immediate Results**: See performance metrics instantly
- **Training Visualization**: View learning curves and progress charts

### 📊 Algorithm Comparison
- **Multi-Algorithm Testing**: Compare 2-4 algorithms simultaneously
- **Visual Comparisons**: Interactive bar charts, radar charts, and performance plots
- **Statistical Analysis**: Automatic statistical significance testing
- **Training Curves**: Compare learning progress across algorithms
- **Performance Metrics**: Side-by-side metric comparisons

### 🎯 Hyperparameter Optimization
- **Automated Parameter Search**: Grid search, random search, and Bayesian optimization
- **Interactive Parameter Ranges**: Set search boundaries with sliders
- **Real-time Progress**: Watch optimization progress with live updates
- **Parameter Space Visualization**: See parameter exploration in 2D/3D plots
- **Best Configuration Identification**: Automatic best parameter discovery

### 📈 Advanced Analysis
- **Comprehensive Statistics**: Descriptive statistics, confidence intervals
- **Performance Clustering**: Automatic grouping of similar-performing configurations
- **Hypothesis Testing**: Statistical significance testing between algorithms
- **Correlation Analysis**: Parameter correlation heatmaps
- **Distribution Analysis**: Performance distribution plots

### 💾 Results Management
- **Results History**: View all past experiments
- **Export Options**: Download results as CSV or JSON
- **Import Results**: Load previously saved results
- **Data Persistence**: Automatic saving of all experiments

## Key Features

### Visual Capabilities
- **Interactive Charts**: Plotly-based interactive visualizations
- **Real-time Updates**: Live progress tracking during training
- **Multi-dimensional Plots**: Radar charts, heatmaps, scatter plots
- **Comparison Views**: Side-by-side algorithm comparisons

### Parameter Control
- **Sliders and Input Fields**: Easy parameter adjustment
- **Range Selectors**: Set parameter search ranges
- **Algorithm-specific Options**: Tailored controls for each algorithm
- **Preset Configurations**: Quick access to optimized settings

### Data Management
- **Automatic Saving**: All results automatically saved with timestamps
- **Export Formats**: CSV, JSON, and image exports
- **Results History**: Persistent storage of all experiments
- **Import/Export**: Share results between sessions

### Performance Monitoring
- **Progress Bars**: Visual training progress indicators
- **Status Updates**: Real-time status messages
- **Error Handling**: Graceful error handling with user feedback
- **Performance Metrics**: Comprehensive metric tracking

## Usage Examples

### 1. Quick Algorithm Test
1. Go to "Single Algorithm Testing"
2. Choose Q-Learning
3. Adjust learning rate to 0.2
4. Set training episodes to 1000
5. Click "Run Training & Evaluation"
6. View results and training progress

### 2. Algorithm Comparison
1. Go to "Algorithm Comparison"
2. Select Q-Learning, SARSA, and DQN
3. Set training episodes to 1500
4. Enable statistical analysis
5. Click "Run Comparison"
6. View comparative charts and statistics

### 3. Hyperparameter Optimization
1. Go to "Hyperparameter Optimization"
2. Choose Q-Learning
3. Set learning rate range: 0.05-0.3
4. Select Grid Search with 25 configurations
5. Click "Start Optimization"
6. View parameter space exploration and best configuration

### 4. Advanced Analysis
1. First run an algorithm comparison
2. Go to "Advanced Analysis"
3. View comprehensive statistical analysis
4. Examine clustering results
5. Check hypothesis testing results

## File Outputs

### Automatic Saves
- **Results JSON**: Complete experimental data
- **Results CSV**: Tabular performance metrics
- **Visualizations**: Automatically saved plots
- **Configuration Files**: Parameter settings used

### Save Locations
```
streamlit_results/
├── taxi_rl_results_YYYYMMDD_HHMMSS.json
├── taxi_rl_results_YYYYMMDD_HHMMSS.csv
├── comparison_YYYYMMDD_HHMMSS.json
└── optimization_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   streamlit run streamlit_app.py --server.port=8502
   ```

2. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **Slow Performance**
   - Reduce training episodes
   - Use fewer configurations in optimization
   - Close other browser tabs

4. **Memory Issues**
   - Restart the Streamlit app
   - Clear results history
   - Reduce batch sizes for DQN

### Performance Tips
- Use training episodes 500-1500 for quick results
- Start with 2-3 algorithms for comparison
- Use 10-20 configurations for optimization
- Clear history periodically to free memory

## Advanced Usage

### Custom Parameter Ranges
- Adjust sliders for fine-tuned parameter control
- Use number inputs for precise values
- Combine multiple parameter variations

### Batch Experiments
- Run multiple comparisons with different settings
- Save results after each experiment
- Load and combine results from different sessions

### Statistical Analysis
- Enable statistical analysis for significance testing
- Use confidence intervals for robust comparisons
- Examine correlation patterns between parameters

## Integration with CLI

The Streamlit interface complements the CLI version:
- **CLI**: Automated batch processing, scripting
- **Streamlit**: Interactive exploration, visualization
- **Both**: Can share result files and configurations

## Best Practices

1. **Start Simple**: Begin with single algorithm tests
2. **Save Frequently**: Use auto-save and manual exports
3. **Compare Systematically**: Use consistent parameters across comparisons
4. **Analyze Results**: Leverage advanced analysis features
5. **Document Findings**: Export results and configurations for reproducibility

The Streamlit interface provides a user-friendly way to explore, compare, and optimize reinforcement learning algorithms with immediate visual feedback and comprehensive analysis capabilities.
