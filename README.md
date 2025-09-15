# CartPoleRL

Train a reinforcement learning agent to balance the CartPole using Proximal Policy Optimization (PPO) with Stable-Baselines3. The project uses `uv` for fast, reproducible Python environments and includes TensorBoard logging and an ONNX export utility.

## Features

- PPO agent built with Stable-Baselines3
- TensorBoard training metrics and visualizations
- Optional GPU acceleration (PyTorch CUDA)
- ONNX export script for deployment
- Inference on ProtoTwin

---
## Resources
> [!IMPORTANT]
> - Notes PDF: [Important understandings (PDF)](https://jumpshare.com/share/5R2Vt26zIvwhY93lSeQS)
> - Curated resources: [tldraw board](https://www.tldraw.com/f/T6oHe2VW4S5P4fRhE0Aqv?d=v-941.3915.2132.1013.0Nu4aCQvq1Lg7bbzkZt0N)
---

## Prerequisites

- Python 3.10+
- Optional: NVIDIA GPU for CUDA acceleration
- ProtoTwin Connect (PTC)
> [!NOTE]
> - If you have multiple Python versions installed, make sure your shell uses Python 3.10+ within the `uv` environment.
> - The project runs on CPU if a compatible GPU is not available.

---

## Quickstart

**1) Clone the repository**
```bash
git clone https://github.com/amugoodbad229/CartPoleRL.git
cd CartPoleRL 
```

**2) Install `uv` (if you don’t have it)**
- Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -Command "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
```
- macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
> [!TIP]
> After installing, restart your terminal so `uv` is available on your PATH.

**3) Create the virtual environment and install dependencies**
```bash
uv sync
```
This creates a local `.venv` and installs all dependencies (including the appropriate PyTorch build).

**4) Activate the environment**
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```
> [!WARNING]
> If activation fails, ensure your shell has permission to run scripts and that you’re in the project directory.

**5) Train the agent**
```bash
# Explore available variants
ls main-v*.py 

# Run one variant (replace x with a number, e.g., 1)
python main-vx.py
```

**6) Monitor training with TensorBoard**
```bash
# Change directory to monitor specific PPO if you are running various PPO models (optional)
cd tensorboard/PPO_1
# or
cd tensorboard/PPO_2 
# ...
cd tensorboard/PPO_N

# Monitor the graphs
python -m tensorboard.main --logdir tensorboard-vx # replace 'x' use number 0, 1, 2 to 
                                                   # know which main file you re running
```
---
## Export to ONNX

Export a trained Stable-Baselines3 policy to ONNX:
```bash
# See available options
python export_onnx.py
```
> [!NOTE]
> Ensure you point the script to the correct checkpoint path if you trained multiple variants. Follow [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/export.html)
---

## Project Structure

```text
.
├── .venv/                   # Virtual environment managed by uv
├── logs-vx/                 # Training logs and artifacts for variant x
│   └── checkpoints/         # Saved model checkpoints
├── tensorboard-vx/          # TensorBoard logs for variant x
├── export_onnx.py           # Convert SB3 policy to ONNX
├── main-vx.py               # Training entry points (variants)
├── pyproject.toml           # Dependencies and project metadata
├── uv.lock                  # Lockfile for reproducible builds
└── README.md                # You are here
```

> [!NOTE]
> Folder names may vary by variant (x). Use the actual paths generated during your runs.


---

## Useful Git commands

```bash
# Check current status
git status

# Add all changes
git add .

# Commit your changes
git commit -m "Improve README and training instructions"

# Push to GitHub
git push
```

---

## Troubleshooting

> [!WARNING]
> uv not found  
> - Install uv (see Quickstart step 2) or ensure it’s on your PATH.  
> - Restart your terminal after installation.

> [!TIP]
> TensorBoard not showing data  
> - Confirm the correct log directory (e.g., `tensorboard-v1` or `tensorboard-v1/PPO_1`).  
> - Ensure training has produced logs.

> [!NOTE]
> Using CPU instead of GPU  
> PyTorch will fall back to CPU if CUDA is unavailable. To check:
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"
> ```

> [!CAUTION]
> Case-sensitive paths  
> Ensure you `cd CartPoleRL`

---
