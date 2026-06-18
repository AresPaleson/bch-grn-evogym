"""Generate best_robot_family_lineup.png for one completed EA run.

This is intentionally separate from run_ea.py's multirun stack analysis.
By default, the figure is written to:

<out_path>/<study_name>/<experiment_name>/run_<run>/analysis/best_robot_family_lineup.png
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from experiments.analysis.plot_best_robot_family_lineup import (
    run_best_robot_family_lineup,
)


# Edit these values, then run:
# python run_scripts/run_best_robot_family_lineup.py
OUT_PATH = "experiments/results/final2"
STUDY_NAME = "flat_crossover_stack_25pop_25off_50gen_1000steps"

# Leave EXPERIMENT_NAME empty to derive it from CROSSOVER_TYPE and the settings
# below. Set it manually if you want a custom experiment folder.
EXPERIMENT_NAME = ""
CROSSOVER_TYPE = "promoter_aligned_cut_and_splice"
RUN = 1

POPULATION_SIZE = 25
OFFSPRING_SIZE = 25
NUM_GENERATIONS = 50
EVOGYM_STEPS = 1000

MAX_VOXELS = 36
CUBE_FACE_SIZE = 6
VOXEL_TYPES = "withbone"
ENV_CONDITIONS = ""
PLASTIC = 0

OUTPUT_NAME = "best_robot_family_lineup.png"

# Leave empty to save inside the selected run's analysis folder.
ANALYSIS_DIR = ""


def derived_experiment_name() -> str:
    if EXPERIMENT_NAME:
        return EXPERIMENT_NAME
    return (
        f"flat_{CROSSOVER_TYPE}_{POPULATION_SIZE}pop_"
        f"{OFFSPRING_SIZE}off_{NUM_GENERATIONS}gen_{EVOGYM_STEPS}steps"
    )


def build_run_dir(experiment_name: str) -> Path:
    return (
        ROOT
        / OUT_PATH
        / STUDY_NAME
        / experiment_name
        / f"run_{RUN}"
    )


def main():
    experiment_name = derived_experiment_name()
    analysis_dir = Path(ANALYSIS_DIR) if ANALYSIS_DIR else build_run_dir(experiment_name) / "analysis"

    output_path = run_best_robot_family_lineup(
        out_path=OUT_PATH,
        study_name=STUDY_NAME,
        experiment_name=experiment_name,
        run=RUN,
        analysis_dir=analysis_dir,
        output_name=OUTPUT_NAME,
        voxel_types=VOXEL_TYPES,
        max_voxels=MAX_VOXELS,
        cube_face_size=CUBE_FACE_SIZE,
        env_conditions=ENV_CONDITIONS,
        plastic=PLASTIC,
    )
    print(f"Best robot family lineup saved to: {output_path}")


if __name__ == "__main__":
    main()
