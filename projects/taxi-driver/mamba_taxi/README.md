

## Algorithms

- **BruteForce**: Random baseline
- **Q-Learning**: Off-policy TD control
- **SARSA**: On-policy TD control  
- **DQN**: Deep Q-Network with experience replay

## Project Structure

```
taxi_driver/
├── agents/           # Agent implementations
│   ├── base.py      # Base agent class
│   ├── bruteforce.py
│   ├── q_learning.py
│   ├── sarsa.py
│   └── dqn.py
├── trainer.py       # Training and evaluation functions
├── visualizer.py    # Results visualization
├── main.py          # Main entry point
├── requirements.txt
└── README.md
```