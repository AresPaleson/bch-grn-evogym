(in the instruction below, the folder referred to as 'base' holds the current repo)

# Base (GRN + EA with EvoGym simulation)

This folder contains the evolutionary pipeline:
- genome/GRN development (`algorithms/GRN_2D.py`)
- optimization loops (`algorithms/basic_EA.py`, `algorithms/cmaes.py`)
- EvoGym preparation and simulation (`simulation/prepare_robot_files.py`, `simulation/simulation_resources.py`)

## Dependencies

Core Python packages used by `base`:
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

Plus local EvoGym package from this repo:
- `../evogym` (original evogym should be cloned and installed in the same parent folder of base)

## Environment setup (recommended)

From repository root:

```bash
python3.9 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install gymnasium scikit-learn lxml cma
```

If needed, install base requirements too:

```bash
pip install -r ./base/requirements.txt
```


Run a small simulation smoke test:

```bash
cd base
source ../.venv/bin/activate
python - <<'PY'
import random
import numpy as np
from types import SimpleNamespace
from algorithms.EA_classes import Individual
from algorithms.GRN_2D import GRN, initialization
from simulation.prepare_robot_files import prepare_robot_files
from simulation.simulation_resources import simulate_evogym_batch

rng = random.Random(3)
genome = initialization(rng, ini_genome_size=80)
phenotype_cells = GRN(
    max_voxels=36,
    cube_face_size=6,
    genotype=genome,
    voxel_types="withbone",
    env_conditions="",
    plastic=0,
).develop()

phenotype_materials = np.zeros(phenotype_cells.shape, dtype=int)
for idx, value in np.ndenumerate(phenotype_cells):
    phenotype_materials[idx] = value.voxel_type if value != 0 else 0

ind = Individual(genome=genome, id_counter=1)
ind.valid = 1
ind.phenotype = phenotype_materials

args = SimpleNamespace(
    out_path="/tmp",
    study_name="demo",
    experiment_name="smoke",
    run=1,
    voxel_types="withbone",
    evogym_steps=500,
    evogym_num_workers=1,
    evogym_init_x=3,
    evogym_init_y=1,
    evogym_action_bias=1.0,
    evogym_action_amplitude=0.4,
    evogym_period_steps=20,
)

prepare_robot_files(ind, args)
simulate_evogym_batch([ind], args)
print("displacement:", ind.displacement)
PY
```

 
