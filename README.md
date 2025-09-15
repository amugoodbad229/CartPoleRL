# CartPoleRL

A reinforcement learning agent that solves the CartPole environment using the PPO algorithm.

This project uses **`uv`** for fast and reproducible setups.

## Getting Started

**Prerequisites:** 
[1] Python 3.10+
[2] NVIDIA GPU updated to the latest NVIDIA driver
[3] ProtoTwin Connect (PTC)

**1. Clone the repository:**
```bash
git clone https://github.com/amugoodbad229/CartPoleRL.git

# Change directory
cd CartpoleRL
```

**2. Install everything with one command:**
This creates a local .venv and installs all dependencies, including the correct GPU version of PyTorch

```bash
uv sync
```

**3. Activate the environment:**

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS 
source .venv/bin/activate
```
Run the Agent

**4. Monitor the agent:**

```bash
# Change directory to monitor specific PPO if you are running various PPO models (optional)
cd tensorboard/PPO_1
# or
cd tensorboard/PPO_2 
# ...
cd tensorboard/PPO_N

# Monitor the graphs
tensorboard --logdir . # Do not forget to add '.'
# or try this if that not works
python -m tensorboard.main --logdir tensorboard-vx # replace 'x' use number 0, 1, 2 to 
                                                   # know which main file you re running
```

**5. Train the agent**
Once the environment is activated, train the agent by running:

```bash
python main-vx.py # replace 'x' with number
```

#### USEFUL GIT COMMANDS

**Check current status**
```bash
git status
```

**Add all changes to staging**
```bash
git add .
```

**Commit your changes**
```bash
git commit -m "Add CartPole RL implementation with PPO and ProtoTwin integration"
```

**Push changes to GitHub**
```bash
git push
```

### Project Structure

.
├── .venv/                   # Virtual environment managed by uv
├── logs-vx/cheackpoints/    # Saved model checkpoints
├── ...                      # Other saved model checkpoints
├── tensorboard-vx/          # TensorBoard logs for monitoring training
├── ...                      # Other TensorBoard logs for monitoring training
├── .gitignore               # Files to ignore for Git
├── export_onnx.py           # convert SB3 policy to ONNX policy
├── main-vx.py               # Main script to train the agent
├── ...                      # Different versions of main script to train the agent
├── pyproject.toml           # Project dependencies and metadata
├── README.md                # Guideline for the newcomers
└── uv.lock                  # Lockfile for reproducible builds

### How It Works?

Environment: The agent operates in the CartPole environment from the Gymnasium library.

Algorithm: It uses the Proximal Policy Optimization (PPO) algorithm, a state-of-the-art reinforcement learning algorithm, implemented via the Stable Baselines3 library.

Goal: The agent's objective is to learn a policy that moves the cart left or right to keep the attached pole balanced upright for as long as possible.

### Important links:

[PDF](https://jumpshare.com/share/5R2Vt26zIvwhY93lSeQS) --> Added important understandings here
[tldraw](https://www.tldraw.com/f/T6oHe2VW4S5P4fRhE0Aqv?d=v-941.3915.2132.1013.0Nu4aCQvq1Lg7bbzkZt0N) --> Added most resources here
