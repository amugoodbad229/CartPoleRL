# 🎯 CartPole Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Reinforcement Learning](https://img.shields.io/badge/RL-PPO-orange.svg)
![Environment](https://img.shields.io/badge/Environment-CartPole-red.svg)

**A sophisticated reinforcement learning agent that masters the CartPole environment using the Proximal Policy Optimization (PPO) algorithm.**

*Experience the power of modern RL with fast, reproducible setups powered by `uv`.*

</div>

---

## 📚 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎮 Usage](#-usage)
- [📊 Monitoring](#-monitoring)
- [📁 Project Structure](#-project-structure)
- [🔧 How It Works](#-how-it-works)
- [📖 Resources](#-resources)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Overview

CartPoleRL is a state-of-the-art reinforcement learning project that demonstrates the power of the **Proximal Policy Optimization (PPO)** algorithm in solving the classic CartPole balancing problem. The agent learns to balance a pole on a cart by making strategic left and right movements, showcasing fundamental RL concepts in an intuitive environment.

## ✨ Features

- 🎯 **PPO Algorithm**: State-of-the-art policy optimization
- 🏃‍♂️ **Fast Setup**: One-command installation with `uv`
- 📊 **Real-time Monitoring**: TensorBoard integration for training visualization
- 🔧 **GPU Support**: Optimized for NVIDIA GPU acceleration
- 📈 **Model Checkpoints**: Automatic saving and loading of trained models
- 🎮 **Multiple Versions**: Support for different model configurations
- 🔄 **Reproducible**: Locked dependencies for consistent results

---

## 🚀 Quick Start

Get up and running in under 2 minutes:

```bash
# Clone and navigate
git clone https://github.com/amugoodbad229/CartPoleRL.git
cd CartPoleRL

# Install everything
uv sync

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Start training
python main-v1.py
```

---

## 📦 Installation

### 🔧 Prerequisites

Before you begin, ensure you have:

| Requirement | Version | Notes |
|------------|---------|-------|
| 🐍 Python | 3.10+ | Required for modern ML libraries |
| 🎮 NVIDIA GPU | Latest drivers | For accelerated training |
| 🔗 ProtoTwin Connect | Latest | PTC integration |

### 📥 Step-by-Step Installation

**1. Clone the Repository**
```bash
git clone https://github.com/amugoodbad229/CartPoleRL.git
cd CartPoleRL
```

**2. One-Command Setup** ⚡
```bash
uv sync
```
> This creates a local `.venv` and installs all dependencies, including the correct GPU version of PyTorch

**3. Activate the Environment**
```bash
# 🐧 Linux/macOS 
source .venv/bin/activate

# 🪟 Windows
.venv\Scripts\activate
```

---

## 🎮 Usage

### 🏋️ Training the Agent

Start training with your preferred model version:
```bash
python main-v1.py  # Version 1
python main-v2.py  # Version 2
# ... or any other version
```

---

## 📊 Monitoring

### 🔍 TensorBoard Visualization

Monitor your agent's training progress in real-time:

**Option 1: Specific Model Monitoring**
```bash
# Navigate to specific PPO model directory
cd tensorboard/PPO_1    # or PPO_2, PPO_3, etc.

# Launch TensorBoard
tensorboard --logdir .
```

**Option 2: Version-Specific Monitoring**
```bash
# Launch TensorBoard for specific version
python -m tensorboard.main --logdir tensorboard-v1  # Replace v1 with your version
```

**Option 3: All Models Overview**
```bash
# Monitor all models simultaneously
tensorboard --logdir tensorboard/
```

> 💡 **Tip**: Open your browser to `http://localhost:6006` to view the TensorBoard dashboard

---

## 📁 Project Structure

```
CartPoleRL/
│
├── 🗂️ .venv/                     # Virtual environment (managed by uv)
├── 💾 logs-v*/checkpoints/       # Saved model checkpoints
├── 📊 tensorboard-v*/            # TensorBoard logs for training visualization
├── 🚫 .gitignore                 # Git ignore rules
├── 🔄 export_onnx.py             # Convert SB3 policy to ONNX format
├── 🎯 main-v*.py                 # Training scripts (multiple versions)
├── 📋 pyproject.toml             # Project dependencies and metadata
├── 📖 README.md                  # Project documentation
└── 🔒 uv.lock                    # Lockfile for reproducible builds
```

---

## 🔧 How It Works

### 🎮 Environment
The agent operates in the **CartPole-v1** environment from the Gymnasium library, where:
- **Goal**: Balance a pole on a moving cart
- **Actions**: Move cart left (0) or right (1)
- **Observation**: Cart position, cart velocity, pole angle, pole angular velocity
- **Reward**: +1 for each timestep the pole remains upright

### 🧠 Algorithm
**Proximal Policy Optimization (PPO)** - A state-of-the-art reinforcement learning algorithm that:
- ✅ Stable and reliable training
- ⚡ Sample efficient
- 🎯 Prevents destructive policy updates
- 📚 Implemented via Stable Baselines3

### 🎯 Objective
Learn an optimal policy that maximizes the time the pole stays balanced by making strategic cart movements.

---

## 📖 Resources

### 📄 Documentation & References
- **[📋 Project Guide (PDF)](https://jumpshare.com/share/5R2Vt26zIvwhY93lSeQS)** - Important concepts and understandings
- **[🎨 Resource Collection (tldraw)](https://www.tldraw.com/f/T6oHe2VW4S5P4fRhE0Aqv?d=v-941.3915.2132.1013.0Nu4aCQvq1Lg7bbzkZt0N)** - Visual resources and diagrams

### 🛠️ Useful Git Commands

<details>
<summary>Click to expand Git reference</summary>

```bash
# Check current status
git status

# Stage all changes
git add .

# Commit with descriptive message
git commit -m "Add CartPole RL implementation with PPO and ProtoTwin integration"

# Push to remote repository
git push
```

</details>

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌟 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔀 Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the RL community**

⭐ Star this repo if you found it helpful!

</div>
