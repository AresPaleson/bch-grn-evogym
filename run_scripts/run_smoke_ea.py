"""Run one tiny EA simulation to check the current setup.

Use this before launching the full crossover stack in `run_ea.py`.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Optional

from run_ea import (
    ExperimentSettings,
    build_run_settings,
    render_robot,
    run_analysis,
    run_single_experiment,
)


ROOT = Path(__file__).resolve().parent.parent

# Common modes:
# - Run one fresh smoke simulation: RUN_SMOKE=1, RENDER_LATEST_SMOKE=0
# - Render the best robot found below: RUN_SMOKE=0, RENDER_SPECIFIC_SMOKE=1
# - Render the newest existing smoke run: RUN_SMOKE=0, RENDER_LATEST_SMOKE=1
# - Run a smoke simulation and immediately render it: RUN_SMOKE=1, RENDER_AFTER_SMOKE=1
RUN_SMOKE = 0
RENDER_SPECIFIC_SMOKE = 1
RENDER_LATEST_SMOKE = 0
RENDER_AFTER_SMOKE = 0
RUN_ANALYSIS_AFTER_SMOKE = 0

SPECIFIC_EXPERIMENT_NAME = "REQUEST_len()"
SPECIFIC_ROBOT_ID = None


def latest_smoke_experiment_name() -> str:
    base_settings = smoke_settings(experiment_name="")
    smoke_root = ROOT / base_settings.out_path / base_settings.study_name
    candidates = [
        path
        for path in smoke_root.glob("_ea_smoke_flat_pit_single*/run_1/run_1")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No smoke run database found under {smoke_root}. "
            "Set RUN_SMOKE = 1 first, or update smoke_settings().out_path "
            "to the folder containing your smoke runs."
        )

    renderable = []
    skipped = []
    for path in candidates:
        con = sqlite3.connect(path)
        try:
            survivor_count = con.execute(
                "SELECT COUNT(*) FROM generation_survivors"
            ).fetchone()[0]
        except sqlite3.Error as exc:
            skipped.append(f"{path.parents[1].name} ({exc})")
            continue
        finally:
            con.close()

        if survivor_count > 0:
            renderable.append(path)
        else:
            skipped.append(f"{path.parents[1].name} (0 survivors)")

    if not renderable:
        skipped_text = ", ".join(skipped) if skipped else "none"
        raise RuntimeError(
            f"No renderable smoke run database found under {smoke_root}. "
            f"Skipped: {skipped_text}. Set RUN_SMOKE = 1 to create a new run."
        )

    latest = max(renderable, key=lambda path: path.stat().st_mtime)
    if skipped:
        print(f"Skipped non-renderable smoke run(s): {', '.join(skipped)}")
    return latest.parents[1].name


def smoke_settings(*, experiment_name: Optional[str] = None) -> ExperimentSettings:
    if experiment_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"_ea_smoke_flat_pit_single_{stamp}"

    return ExperimentSettings(
        run_experiment=1,
        render_only=0,
        population_size=20,
        offspring_size=20,
        num_generations=10,
        evogym_steps=500,
        max_voxels=36,
        cube_face_size=6,
        out_path="experiments/results/tmp",
        study_name="_ea_smoke_flat_pit",
        experiment_name=experiment_name,
        run=1,
        num_runs=1,
        crossover_type="cut_and_splice",
        crossover_types="",
        analyze_after_run=RUN_ANALYSIS_AFTER_SMOKE,
        render_after_run=RENDER_AFTER_SMOKE,
        record_video=0,
        evogym_num_workers=1,
        evogym_isolate_tasks=0,
        render_headless=0,
        render_render_mode="screen",
    )


def main():
    if RUN_SMOKE:
        settings = build_run_settings(
            smoke_settings(),
            crossover_type="cut_and_splice",
            run=1,
        )
        print(
            "Running one smoke EA simulation: "
            f"{settings.population_size} pop, {settings.offspring_size} offspring, "
            f"{settings.num_generations} generation, {settings.evogym_steps} steps."
        )
        run_single_experiment(settings)

        if RUN_ANALYSIS_AFTER_SMOKE:
            output_paths = run_analysis(settings)
            print(f"Analysis figure saved to: {output_paths['morphology_progression']}")
            print(f"Fitness-over-generations figure saved to: {output_paths['fitness_over_generations']}")
            print(f"Parent-child morphology/fitness figure saved to: {output_paths['parent_child_fitness_density']}")
            if output_paths.get("statistical_report") is not None:
                print(f"Statistical report saved to: {output_paths['statistical_report']}")

        if RENDER_AFTER_SMOKE:
            render_robot(settings)

        print("Smoke run completed.")
        return

    if RENDER_LATEST_SMOKE:
        experiment_name = latest_smoke_experiment_name()
        settings = build_run_settings(
            smoke_settings(experiment_name=experiment_name),
            crossover_type="cut_and_splice",
            run=1,
        )
        print(f"Rendering latest smoke run: {experiment_name}")
        render_robot(settings)
        return

    if RENDER_SPECIFIC_SMOKE:
        if SPECIFIC_ROBOT_ID is None:
            raise ValueError("Set SPECIFIC_ROBOT_ID before rendering a specific smoke robot.")
        settings = build_run_settings(
            smoke_settings(experiment_name=SPECIFIC_EXPERIMENT_NAME),
            crossover_type="cut_and_splice",
            run=1,
        )
        settings.render_robot_id = SPECIFIC_ROBOT_ID
        print(
            f"Rendering specific smoke robot: "
            f"{SPECIFIC_EXPERIMENT_NAME} robot_id={SPECIFIC_ROBOT_ID}"
        )
        render_robot(settings)
        return

    print("Nothing to do. Set RUN_SMOKE, RENDER_SPECIFIC_SMOKE, or RENDER_LATEST_SMOKE to 1.")


if __name__ == "__main__":
    main()
