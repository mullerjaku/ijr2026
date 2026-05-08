<div align="center">

# IWINAC 2026

**Intrinsic motivation methods for goal exploration and skill improvement in a simple robotic environment.**

</div>

## Videos
Experiment videos can be found in the `experiment_videos` directory.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mullerjaku/iwinac2026.git
cd iwinac2026

# Conda env
conda create -n iwinac python=3.10.12 -y
conda activate iwinac

# Install dependencies
pip install --upgrade pip
pip install -e .
```

## Running the Experiments

You can reproduce different experimental setups by running the following scripts:

| Method | Description | Command |
| :--- | :--- | :--- |
| **Motivational** | Experiment using motivational engine for goal change position. | `python cur.py` |
| **Novelty** | Experiment using novelty engine for goal change position. | `python nov.py` |
| **World Model** | Experiment using motivational engine for world model change position. | `python wm.py` |
| **Combined** | Experiment using motivational engine for goal position and world model change. | `python comb.py` |

---
Maintained by Jakub Müller.