# 🏋️‍♂️ CartPoleRL  
Modern PPO-based Reinforcement Learning for solving the classic CartPole control task using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).  
Clean developer workflow powered by [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environments.  
Includes: multiple experiment variants, TensorBoard monitoring, ONNX export, and ProtoTwin inference support.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/RL-PPO-orange" />
  <img src="https://img.shields.io/badge/Framework-Stable--Baselines3-green" />
  <img src="https://img.shields.io/badge/Backend-PyTorch-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Env-uv-7834f8" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

<p align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-training--experiment-variants">Training</a> •
  <a href="#-monitoring">Monitoring</a> •
  <a href="#-onnx-export--deployment">ONNX</a> •
  <a href="#-troubleshooting">Troubleshooting</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## ✨ Features

| Area | Capability |
|------|------------|
| Algorithms | PPO (easily extensible to A2C, DQN, etc.) |
| Experimentation | Versioned training entrypoints: `main-v1.py`, `main-v2.py`, ... |
| Monitoring | TensorBoard logs per variant: `tensorboard-v1/`, `tensorboard-v2/` |
| Export | ONNX conversion via `export_onnx.py` |
| Deployment | Ready for ProtoTwin (upload ONNX or Python policy) |
| Extensibility | Clean project layout – add new envs or models fast |
| Dev UX | Minimal commands to get started |

> [!TIP]  
> Want to add new variants? Copy an existing `main-vx.py`, tweak hyperparameters, log into a new `logs-vX/` + `tensorboard-vX/` path.

---

## 📚 Resources

> [!IMPORTANT]  
> • Notes PDF: [Important understandings (PDF)](https://jumpshare.com/share/5R2Vt26zIvwhY93lSeQS)  
> • Curated Resource Board: [tldraw board](https://www.tldraw.com/f/T6oHe2VW4S5P4fRhE0Aqv?d=v-941.3915.2132.1013.0Nu4aCQvq1Lg7bbzkZt0N)

---

## ✅ Prerequisites

- Python 3.10+  
- (Optional) NVIDIA GPU + CUDA-capable PyTorch build  
- (Optional) ProtoTwin Connect (PTC) for deployment  
- Shell that can run virtual environment activation scripts

> [!NOTE]  
> PyTorch falls back to CPU automatically if CUDA is not available.

---

## ⚡ Quickstart

```bash
git clone https://github.com/amugoodbad229/CartPoleRL.git
cd CartPoleRL
```

Install `uv` (one time):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -Command "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
```

Sync environment + dependencies:

```bash
uv sync
```

Activate environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Run a training variant:

```bash
ls main-v*.py          # discover available variants
python main-v1.py      # or main-v2.py, etc.
```

---

## 🧪 Training & Experiment Variants

Each `main-vX.py` file encapsulates a slightly different configuration:
- Hyperparameters (Learning rate, gamma, entropy)
- Network architecture (Default or Custom)
- Logging folder (Agent Models)
- Callback setup

> [!TIP]  
> Duplicate an existing file to create a new experiment:  
> `cp main-v1.py main-v3.py` → edit run name, log path, and hyperparameters.

### Suggested Naming Convention

| Variant | Purpose |
|---------|---------|
| `main-v0.py` | Baseline PPO |
| `main-v1.py` | Tuned learning rate / entropy |
| `main-v2.py` | Different network width |
| `main-v3.py` | Longer training horizon |
| `main-vN.py` | Custom experiment |

---

## 📊 Monitoring

Launch TensorBoard (choose the appropriate variant path):

```bash
python -m tensorboard.main --logdir tensorboard-v1
```

Or to watch all:

```bash
python -m tensorboard.main --logdir .
```

> [!TIP]  
> If nothing appears, ensure training produced events:  
> `find tensorboard-v1 -type f -name "*tfevent*"`

---

## 📦 ONNX Export & Deployment

Generate an ONNX policy (after training):

```bash
python export_onnx.py
```

> [!NOTE]  
> If the script uses hardcoded paths, edit `export_onnx.py` or extend it with `argparse`.


ProtoTwin usage:
Upload the ONNX file to ProtoTwin or deploy the Python inference.

---

## 🧱 Project Structure

```text
.
├── main-v1.py             # Training variant 1
├── main-v2.py             # Training variant 2 (extend as needed)
├── export_onnx.py         # Convert trained model to ONNX
├── logs-v1/               # Training logs + checkpoints (variant 1)
│   └── checkpoints/
├── tensorboard-v1/        # TensorBoard event files (variant 1)
├── pyproject.toml         # Project + dependency definitions
├── uv.lock                # Locked, reproducible dependency set
└── README.md
```

> [!NOTE]  
> Additional variants (e.g., `logs-v2/`, `tensorboard-v2/`) appear after running those scripts.

---

## 🔧 Extending the Project

| Task | How |
|------|-----|
| Add a new algorithm | Replace PPO import with another SB3 algorithm |
| Add custom policy | Define `policy_kwargs` in the training script |
| Change environment | Swap `CartPole-v1` with another Gymnasium env |
| Add callbacks | Implement `BaseCallback` and register in training |
| Log extra metrics | Use custom callback + `self.logger.record()` |

---

## 🧪 Evaluating a Policy

Add (or use) a snippet like:

```python
from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean: {mean_reward:.2f} ± {std_reward:.2f}")
```
---

## 🛠 Useful Git Commands

```bash
git status
git add .
git commit -m "Experiment: tuned lr and entropy"
git push origin main
```

> [!TIP]  
> Use branches for big experiments:  
> `git checkout -b feat/entropy-sweep`

---

## 🚑 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `uv: command not found` | Reinstall `uv`, restart terminal |
| No TensorBoard data | Confirm correct `tensorboard-vX/` path |
| CPU instead of GPU | Check: `python -c "import torch; print(torch.cuda.is_available())"` |
| ImportError (SB3) | Run `uv sync` again (env might be stale) |
| Permission denied on activate | On Unix: `chmod +x .venv/bin/activate` (rare) |

> [!CAUTION]  
> Paths are case-sensitive. Use `cd CartPoleRL`, not `cd cartpolerl`.

---

## 🧭 Roadmap

- [ ] Add evaluation script (e.g., `evaluate.py`)
- [ ] Hyperparameter sweeps integration (Optuna / WandB)
- [ ] Dockerfile for containerized deployment
- [ ] Unified config system (`config/` + YAML)
- [ ] CI workflow (lint + format + smoke test)

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch: `git checkout -b feat/new-idea`  
4. Submit PR with: description, metrics, rationale

> [!TIP]  
> Keep results reproducible—note seeds if changed.

---

## 🙏 Acknowledgements

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)  
- [PyTorch](https://pytorch.org/)  
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)  
- [TensorBoard](https://www.tensorflow.org/tensorboard)  
- [ProtoTwin Platform](https://prototwin.com/)

---

## 📄 License


```text
MIT License © 2025 Ayman Khan
```

---

## ⭐ Support

If this helps you learn or prototype faster:
- Star the repo
- Share feedback
- Open issues for improvements

---

<p align="center"><strong>Happy balancing! 🛠️🧠</strong></p>
