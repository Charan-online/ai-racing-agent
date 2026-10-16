
# Autonomous AI Racing Simulator

A 2D autonomous racing simulator built with Python, Pygame, Gymnasium, and Stable-Baselines3. The project uses Proximal Policy Optimization (PPO), a reinforcement learning algorithm, to train an AI agent to navigate a racing track.

## Features

- Custom racing environment using Gymnasium
- PPO reinforcement learning agent
- Seven simulated distance sensors
- Nine discrete driving actions
- Randomized starting positions and orientations
- Live Pygame visualization
- Automated evaluation across multiple races

## Technologies

- Python
- Pygame
- NumPy
- Gymnasium
- Stable-Baselines3
- Proximal Policy Optimization (PPO)

## Project Status

The AI agent has completed 20 out of 20 tested randomized races without collision. This result applies to the 20 evaluation scenarios tested, not every possible starting condition.

More documentation and setup instructions will be added below.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Charan-online/ai-racing-agent.git
cd ai-racing-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pygame numpy gymnasium stable-baselines3
```

### 4. Validate the environment

```bash
python validate_env.py
```

### 5. Watch the trained agent

```bash
python watch_ppo.py
```

### 6. Evaluate the trained agent

```bash
python evaluate_ppo.py
```

**Note:** The evaluation and visualization scripts expect the trained model at `models/randomized_best/best_model.zip`. Ensure this file is included in the repository before running them.

### 7. Train the agent again (optional)

```bash
python train_ppo.py
```

## Evaluation Results

The trained PPO agent was evaluated across 20 randomized starting scenarios.

| Metric | Result |
|---|---:|
| Total test races | 20 |
| Clean laps | 20 |
| Failed laps | 0 |
| Clean-lap success rate | 100% |
| Average steps per clean lap | 265.6 |
| Average simulated lap time | 4.43 seconds |
| Best simulated lap time | 4.38 seconds |
| Worst successful lap time | 4.48 seconds |

**Important:** These results apply to the 20 tested scenarios, not every possible starting condition. Simulated lap time is calculated as the number of steps divided by 60, rather than measured real-world driving time.

