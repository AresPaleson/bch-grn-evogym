# EvoGym GRN Bachelor Thesis Repository

This repository contains the code and experiment material for the bachelor project on
GRN-developed EvoGym robots and evolutionary crossover operators.

## Repository Layout

- `algorithms/` - GRN genome development, evolutionary algorithm classes, and optimization loops.
- `simulation/` - robot file preparation and EvoGym simulation helpers.
- `utils/` - shared configuration, metrics, drawing, and body-metric utilities.
- `run_scripts/` - main runnable experiment entry points.
- `experiments/analysis/` - analysis, plotting, consolidation, and statistical-test scripts.
- `experiments/results/final2/` - final thesis experiment outputs and analysis artifacts.
- `evogym/` - external EvoGym source dependency; 

## Dependencies

Core Python packages:

- `numpy`
- `sqlalchemy`
- `matplotlib`
- `scipy`
- `pandas`
- `opencv-python`
- `scikit-learn`
- `lxml`
- `cma`
- `gymnasium`

Plus the local EvoGym dependency in `evogym/`.

## Environment Setup

First Install EvoGym using its original instructions.

From the repository root:

```bash
python3.9 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.9 -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run or render a small smoke experiment:

```bash
python run_scripts/run_smoke_ea.py
```
