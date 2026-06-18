"""Run the full EA crossover stack on the current EvoGym world.

This is the long experiment runner. For a quick single-run smoke test, use
`run_scripts/run_smoke_ea.py`.
"""

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Optional
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from algorithms.GRN import GRN
from algorithms.EA import EA
from experiments.analysis.consolidate import Analysis
from experiments.analysis.plot_fitness_progress import run_plot as run_fitness_plot
from experiments.analysis.plot_parent_child_fitness_density import run_density_plot
from experiments.analysis.plot_morphology_progress import run_plot
from simulation.prepare_robot_files import prepare_robot_files
from simulation.simulation_resources import simulate_evogym_batch
from experiments.analysis.statistical_tests_final_generation_population import (
    run_final_generation_population_tests,
)
from experiments.analysis.statistical_tests_parent_child_morphology import run_statistical_tests


EVOGYM_ENVIRONMENT_OPTIONS = (
    "FlatCeiling-v0",
    "DownStepper-v0",
)

EVOGYM_ENVIRONMENT_ALIASES = {
    "flatceiling": "FlatCeiling-v0",
    "flatceilingv0": "FlatCeiling-v0",
    "ceiling": "FlatCeiling-v0",
    "corridor": "FlatCeiling-v0",
    "downstepper": "DownStepper-v0",
    "downstepperv0": "DownStepper-v0",
}

CROSSOVER_TYPE_ALIASES = {
    "promoter_aligned_cut_and_splice": "promoter_aligned_cut_and_splice",
    "cut_and_splice": "promoter_aligned_cut_and_splice",
    "cutpoint": "promoter_aligned_cut_and_splice",
    "one_point": "promoter_aligned_cut_and_splice",
    "old": "promoter_aligned_cut_and_splice",
    "proportional": "promoter_aligned_cut_and_splice",
    "unequal_prop": "promoter_aligned_cut_and_splice",
    "arithmetic_recombination": "arithmetic_recombination",
    "arithmetic_crossover": "arithmetic_recombination",
    "intermediate_recombination": "arithmetic_recombination",
    "blended": "arithmetic_recombination",
    "blend": "arithmetic_recombination",
    "weighted_average": "arithmetic_recombination",
    "weighted": "arithmetic_recombination",
    "arithmetic": "arithmetic_recombination",
    "homologous_gene_block_recombination": "homologous_gene_block_recombination",
    "homologous_gene_recombination": "homologous_gene_block_recombination",
    "homologous_gene_block": "homologous_gene_block_recombination",
    "homologous": "homologous_gene_block_recombination",
    "aligned_gene_block": "homologous_gene_block_recombination",
    "gene_block": "homologous_gene_block_recombination",
}
CROSSOVER_TYPE_OPTIONS = (
    "promoter_aligned_cut_and_splice",
    "arithmetic_recombination",
    "homologous_gene_block_recombination",
)


def _normalize_environment_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def resolve_evogym_env_name(settings) -> str:
    selected = str(
        getattr(settings, "simulation_environment", "")
        or getattr(settings, "evogym_env_name", "")
        or ""
    ).strip()
    if not selected:
        return "FlatCeiling-v0"

    if selected in EVOGYM_ENVIRONMENT_OPTIONS:
        return selected

    alias = EVOGYM_ENVIRONMENT_ALIASES.get(_normalize_environment_key(selected))
    if alias is not None:
        return alias

    options = ", ".join(EVOGYM_ENVIRONMENT_OPTIONS)
    raise ValueError(
        f"Unknown simulation environment {selected!r}. "
        f"Use: {options}."
    )


def normalize_crossover_type(crossover_type: str) -> str:
    key = str(crossover_type).strip().lower()
    if key not in CROSSOVER_TYPE_ALIASES:
        valid = ", ".join(CROSSOVER_TYPE_OPTIONS)
        raise ValueError(
            f"Unsupported crossover_type {crossover_type!r}. Use one of: {valid}."
        )
    return CROSSOVER_TYPE_ALIASES[key]


def parse_crossover_types(raw_value: str) -> list[str]:
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return []

    if _normalize_environment_key(raw_value) == "all":
        return list(CROSSOVER_TYPE_OPTIONS)

    crossovers = []
    for item in raw_value.split(","):
        if not item.strip():
            continue
        crossover = normalize_crossover_type(item)
        if crossover not in crossovers:
            crossovers.append(crossover)
    return crossovers


def selected_crossover_types(settings) -> list[str]:
    crossovers = parse_crossover_types(getattr(settings, "crossover_types", ""))
    if crossovers:
        return crossovers
    return [normalize_crossover_type(settings.crossover_type)]


def run_numbers(settings) -> list[int]:
    count = int(getattr(settings, "num_runs", 1))
    if count < 1:
        raise ValueError("num_runs must be at least 1.")
    return list(range(1, count + 1))


def derived_experiment_name(settings, crossover_type: str) -> str:
    if settings.experiment_name:
        return settings.experiment_name
    return (
        f"flat_{crossover_type}_{settings.population_size}pop_"
        f"{settings.offspring_size}off_{settings.num_generations}gen_"
        f"{settings.evogym_steps}steps"
    )


def build_run_settings(
    settings: "ExperimentSettings",
    *,
    crossover_type: str,
    run: int,
) -> "ExperimentSettings":
    crossover_type = normalize_crossover_type(crossover_type)
    return replace(
        settings,
        crossover_type=crossover_type,
        experiment_name=derived_experiment_name(settings, crossover_type),
        run=run,
    )


@dataclass
class ExperimentSettings:
    
    run_experiment: int = 0
    render_only: int = not run_experiment

    population_size: int = 25
    offspring_size: int = 25
    num_generations: int = 50
    evogym_steps: int = 1000
    record_video: int = 1
    
    
    # Results root folder. Output will land under:
    # <out_path>/<study_name>/<experiment_name>/run_<run>/
    out_path: str = "experiments/results/final2"
    study_name: str = f"flat_crossover_stack_{population_size}pop_{offspring_size}off_{num_generations}gen_{evogym_steps}steps"
    # Leave empty to derive one experiment folder per crossover type.
    experiment_name: str = ""
    algorithm: str = "EA"
    run: int = 1
    # Number of independent runs per selected crossover type.
    num_runs: int = 10

    # Parent selection pressure.
    tournament_k: int = 4
    parent_tournament_k: Optional[int] = None

    # Body-development limits.
    # cube_face_size=6 means the GRN grows inside a 6x6 matrix before trimming.
    # max_voxels=36 means at most 36 cells can be occupied.
    max_voxels: int = 36
    cube_face_size: int = 6
    voxel_types: str = "withbone"
    plastic: int = 0
    env_conditions: str = ""

    # Variation operators.
    crossover_prob: float = 1.0
    # Options:
    # - "arithmetic_recombination": random-alpha weighted genome average.
    # - "promoter_aligned_cut_and_splice": variable-length promoter-aligned cut/splice.
    # - "homologous_gene_block_recombination": similarity-matched whole-gene recombination.
    # For batch experiments, edit crossover_types. Use "all" or a comma-separated
    # subset such as "arithmetic_recombination,homologous_gene_block_recombination".
    crossover_type: str = "promoter_aligned_cut_and_splice"
    crossover_types: str = "all"
    mutation_prob: float = 0.9

    # Optimization target used to set `individual.fitness`.
    # Common choices are "displacement", "novelty", "novelty_weighted", "num_voxels".
    fitness_metric: str = "displacement"

    # Kept for compatibility with older config/analysis code. The simple EA
    # uses direct fitness selection.
    ea_objectives: str = "fitness,novelty"
    novelty_archive_max_size: int = 100
    novelty_archive_add_k: int = 1

    # Physics / terrain settings.
    simulation_environment: str = "flatceiling"
    evogym_env_name: str = "FlatCeiling-v0"
    ustatic: float = 0.5
    udynamic: float = 0.2
    evogym_left_wall: int = 0
    evogym_left_wall_height: int = 8
    evogym_flat_ceiling_gap_blocks: int = 6
    evogym_flat_ceiling_width: int = 100

    # If 1, simulate robots in EvoGym and compute behavior metrics like displacement.
    # If 0, the run only develops bodies and cannot optimize locomotion meaningfully.
    run_simulation: int = 1

    # EvoGym execution defaults.
    # Windows + EvoGym native bindings are more reliable in a single worker by default.
    evogym_num_workers: int = 6
    # Evaluate each robot in a fresh subprocess to prevent long-run native leaks/crashes.
    evogym_isolate_tasks: int = 1
    evogym_init_x: int = 1
    evogym_init_y: int = 7
    evogym_action_bias: float = 1.0
    evogym_action_amplitude: float = 0.6
    evogym_ann_hidden_size: int = 8
    evogym_sine_period: float = 40.0
    evogym_sine_mix: float = 0.35
    evogym_headless: int = 1
    evogym_render_mode: str = "screen"

    # Present for compatibility with the shared config object.
    generations: str = ""
    final_gen: str = ""
    experiments: str = ""
    runs: str = ""

    # Windows-native pipeline controls.
    
    analyze_after_run: int = 1
    analysis_metrics: str = (
        "size,proportion,coverage,symmetry,relative_number_of_joints,"
        "relative_number_of_limbs,total_voxel_volume,bounding_box_area,"
        "actuation_energy_cost,"
        "environmental_contact_area,"
        "material_ratios,muscle_phase_ratios"
    )
    analysis_output_name: str = "morphology_metrics_progression.png"
    fitness_analysis_output_name: str = "fitness_over_generations.png"
    stack_analysis_dir_name: str = "analysis"
    run_statistical_tests_after_analysis: int = 1
    statistical_alpha: float = 0.05

    render_after_run: int = 0
    render_generation: Optional[int] = None
    render_robot_id: Optional[int] = None
    # Comma-separated render jobs in the form:
    # crossover_type:run:robot_id:video_name
    # Leave empty to render the single crossover_type/run/render_robot_id above.
    render_targets: str = (
        "arithmetic_recombination:10:744:best_arithmetic_robot.gif,"
        "homologous_gene_block_recombination:6:1198:best_homologous_robot.gif"
    )
    
    render_metric: str = "fitness"
    render_headless: int = 1
    render_render_mode: str = "screen"
    record_video_fps: int = 50
    record_video_stride: int = 1
    record_video_name: str = "best_robot.gif"

    # `EA.py` reads this with `getattr`, so we expose it here too.
    elitism: int = 3


def build_args(settings: ExperimentSettings) -> SimpleNamespace:
    # Convert the editable dataclass into the attribute-style object expected by the EA.
    args = SimpleNamespace(**settings.__dict__)
    args.evogym_env_name = resolve_evogym_env_name(settings)
    return args


def build_run_dir(settings: ExperimentSettings) -> Path:
    return (
        ROOT
        / settings.out_path
        / settings.study_name
        / settings.experiment_name
        / f"run_{settings.run}"
    )


def _format_timestamp(value: Optional[datetime]) -> str:
    return value.isoformat(timespec="seconds") if value is not None else ""


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return ""
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours}h {minutes}m {seconds:.1f}s ({value:.1f} seconds)"


def _format_parameter_value(value) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _fetch_one_dict(cur: sqlite3.Cursor, query: str, params=()):
    row = cur.execute(query, params).fetchone()
    return dict(row) if row is not None else None


def load_run_db_summary(settings: ExperimentSettings) -> dict:
    db_path = build_db_path(settings)
    summary = {
        "database_path": db_path,
        "database_exists": db_path.exists(),
    }
    if not db_path.exists():
        return summary

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        summary["seed"] = cur.execute(
            "SELECT seed FROM experiment_info ORDER BY id LIMIT 1"
        ).fetchone()["seed"]
        summary["completed_generation"] = cur.execute(
            "SELECT MAX(generation) FROM generation_survivors"
        ).fetchone()[0]
        summary["robots_saved"] = cur.execute(
            "SELECT COUNT(*) FROM all_robots"
        ).fetchone()[0]
        summary["survivor_rows_saved"] = cur.execute(
            "SELECT COUNT(*) FROM generation_survivors"
        ).fetchone()[0]

        final_generation = summary["completed_generation"]
        if final_generation is not None:
            summary["best_final_generation_robot"] = _fetch_one_dict(
                cur,
                """
                SELECT gs.generation, gs.robot_id, gs.fitness,
                       r.displacement, r.valid, r.born_generation, r.num_voxels
                FROM generation_survivors gs
                JOIN all_robots r ON r.robot_id = gs.robot_id
                WHERE gs.generation = ?
                ORDER BY gs.fitness DESC
                LIMIT 1
                """,
                (final_generation,),
            )

        summary["best_saved_robot"] = _fetch_one_dict(
            cur,
            """
            SELECT gs.generation, gs.robot_id, gs.fitness,
                   r.displacement, r.valid, r.born_generation, r.num_voxels
            FROM generation_survivors gs
            JOIN all_robots r ON r.robot_id = gs.robot_id
            ORDER BY gs.fitness DESC
            LIMIT 1
            """,
        )
    finally:
        con.close()

    return summary


def _append_robot_summary(lines: list[str], label: str, robot: Optional[dict]):
    lines.append(f"{label}:")
    if not robot:
        lines.append("  none")
        return
    for key, value in robot.items():
        lines.append(f"  {key}: {value}")


def write_run_info(
    settings: ExperimentSettings,
    *,
    status: str,
    started_at: datetime,
    ended_at: Optional[datetime] = None,
    elapsed_seconds: Optional[float] = None,
    error: Optional[str] = None,
):
    run_dir = build_run_dir(settings)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info_path = run_dir / "run_info.txt"

    args = build_args(settings)
    db_summary = load_run_db_summary(settings)

    lines = [
        "Experiment Run Info",
        "===================",
        "",
        "Run",
        "---",
        f"status: {status}",
        f"started_at: {_format_timestamp(started_at)}",
        f"ended_at: {_format_timestamp(ended_at)}",
        f"elapsed: {_format_seconds(elapsed_seconds)}",
        f"algorithm: {settings.algorithm}",
        f"study_name: {settings.study_name}",
        f"experiment_name: {settings.experiment_name}",
        f"run: {settings.run}",
        f"crossover_type: {settings.crossover_type}",
        "",
        "Paths",
        "-----",
        f"workspace_root: {ROOT}",
        f"run_directory: {run_dir}",
        f"database_path: {db_summary['database_path']}",
        f"database_exists: {db_summary['database_exists']}",
        f"manual_stop_file: {run_dir / 'STOP'}",
        "",
        "Database Summary",
        "----------------",
        f"seed: {db_summary.get('seed', '')}",
        f"completed_generation: {db_summary.get('completed_generation', '')}",
        f"robots_saved: {db_summary.get('robots_saved', '')}",
        f"survivor_rows_saved: {db_summary.get('survivor_rows_saved', '')}",
        "",
    ]
    _append_robot_summary(
        lines,
        "best_final_generation_robot",
        db_summary.get("best_final_generation_robot"),
    )
    lines.append("")
    _append_robot_summary(lines, "best_saved_robot", db_summary.get("best_saved_robot"))

    lines.extend(
        [
            "",
            "Resolved Settings",
            "-----------------",
            f"resolved_evogym_env_name: {args.evogym_env_name}",
            "",
            "Parameters",
            "----------",
        ]
    )
    for key, value in settings.__dict__.items():
        lines.append(f"{key}: {_format_parameter_value(value)}")

    if error:
        lines.extend(["", "Error", "-----", error])

    run_info_path.write_text("\n".join(lines) + "\n")


def run_single_experiment(settings: ExperimentSettings):
    started_at = datetime.now().astimezone()
    start_perf = time.perf_counter()
    write_run_info(settings, status="running", started_at=started_at)

    try:
        args = build_args(settings)
        EA(args).run()
    except Exception as exc:
        ended_at = datetime.now().astimezone()
        write_run_info(
            settings,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=time.perf_counter() - start_perf,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    ended_at = datetime.now().astimezone()
    write_run_info(
        settings,
        status="completed",
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=time.perf_counter() - start_perf,
    )


def run_analysis(settings: ExperimentSettings) -> dict[str, Path]:
    args = build_args(settings)
    args.final_gen = str(settings.num_generations)
    args.experiments = settings.experiment_name
    args.runs = str(settings.run)
    analysis_dir = build_run_dir(settings) / "analysis"
    args.analysis_dir = str(analysis_dir)

    Analysis(args).consolidate()

    morphology_plot_path = analysis_dir / settings.analysis_output_name
    run_plot(
        analysis_dir=analysis_dir,
        experiments_raw=settings.experiment_name,
        metrics_raw=settings.analysis_metrics,
        output_name=settings.analysis_output_name,
    )
    fitness_plot_path = run_fitness_plot(
        analysis_dir=analysis_dir,
        experiments_raw=settings.experiment_name,
        output_name=settings.fitness_analysis_output_name,
    )
    density_plot_path = run_density_plot(
        out_path=settings.out_path,
        study_name=settings.study_name,
        analysis_dir=analysis_dir,
        experiments_raw=settings.experiment_name,
        runs_raw=str(settings.run),
        valid_only=1,
        max_voxels=settings.max_voxels,
        cube_face_size=settings.cube_face_size,
        voxel_types=settings.voxel_types,
        env_conditions=settings.env_conditions,
        plastic=settings.plastic,
        fitness_metric=settings.fitness_metric,
    )
    statistical_report_path = None
    final_generation_statistical_report_path = None
    if settings.run_statistical_tests_after_analysis:
        final_generation_statistical_paths = run_final_generation_population_tests(
            analysis_dir=analysis_dir,
            alpha=settings.statistical_alpha,
            experiments_raw=settings.experiment_name,
        )
        final_generation_statistical_report_path = final_generation_statistical_paths[
            "statistical_report"
        ]
        statistical_paths = run_statistical_tests(
            analysis_dir=analysis_dir,
            alpha=settings.statistical_alpha,
            parent_slot="closest_parent",
        )
        statistical_report_path = statistical_paths["statistical_report"]
    return {
        "morphology_progression": morphology_plot_path,
        "fitness_over_generations": fitness_plot_path,
        "parent_child_fitness_density": density_plot_path,
        "statistical_report": statistical_report_path,
        "final_generation_statistical_report": final_generation_statistical_report_path,
    }


def build_stack_analysis_dir(settings: ExperimentSettings) -> Path:
    return ROOT / settings.out_path / settings.study_name / settings.stack_analysis_dir_name


def iter_stack_run_settings(settings: ExperimentSettings):
    if settings.experiment_name and len(selected_crossover_types(settings)) > 1:
        raise ValueError(
            "Leave experiment_name empty when running multiple crossover types, "
            "or select a single crossover_type."
        )

    for crossover_type in selected_crossover_types(settings):
        for run in run_numbers(settings):
            yield build_run_settings(
                settings,
                crossover_type=crossover_type,
                run=run,
            )


def run_crossover_stack(settings: ExperimentSettings):
    runs = list(iter_stack_run_settings(settings))
    print(
        f"Starting crossover stack: {len(selected_crossover_types(settings))} "
        f"crossover types x {len(run_numbers(settings))} runs = {len(runs)} runs."
    )
    for index, run_settings in enumerate(runs, start=1):
        print(
            f"[{index}/{len(runs)}] Running {run_settings.experiment_name} "
            f"run {run_settings.run}/{settings.num_runs} "
            f"({run_settings.crossover_type})"
        )
        run_single_experiment(run_settings)


def run_stack_analysis(settings: ExperimentSettings) -> dict[str, Path]:
    crossovers = selected_crossover_types(settings)
    runs = run_numbers(settings)
    experiment_names = [
        derived_experiment_name(settings, crossover_type)
        for crossover_type in crossovers
    ]

    args = build_args(settings)
    args.final_gen = str(settings.num_generations)
    args.experiments = ",".join(experiment_names)
    args.runs = ",".join(str(run) for run in runs)

    analysis_dir = build_stack_analysis_dir(settings)
    args.analysis_dir = str(analysis_dir)

    Analysis(args).consolidate()

    morphology_plot_path = analysis_dir / settings.analysis_output_name
    run_plot(
        analysis_dir=analysis_dir,
        experiments_raw=args.experiments,
        metrics_raw=settings.analysis_metrics,
        output_name=settings.analysis_output_name,
    )
    fitness_plot_path = run_fitness_plot(
        analysis_dir=analysis_dir,
        experiments_raw=args.experiments,
        output_name=settings.fitness_analysis_output_name,
    )
    density_plot_path = run_density_plot(
        out_path=settings.out_path,
        study_name=settings.study_name,
        analysis_dir=analysis_dir,
        experiments_raw=args.experiments,
        runs_raw=args.runs,
        valid_only=1,
        max_voxels=settings.max_voxels,
        cube_face_size=settings.cube_face_size,
        voxel_types=settings.voxel_types,
        env_conditions=settings.env_conditions,
        plastic=settings.plastic,
        fitness_metric=settings.fitness_metric,
    )
    statistical_report_path = None
    final_generation_statistical_report_path = None
    if settings.run_statistical_tests_after_analysis:
        final_generation_statistical_paths = run_final_generation_population_tests(
            analysis_dir=analysis_dir,
            alpha=settings.statistical_alpha,
            experiments_raw=args.experiments,
        )
        final_generation_statistical_report_path = final_generation_statistical_paths[
            "statistical_report"
        ]
        statistical_paths = run_statistical_tests(
            analysis_dir=analysis_dir,
            alpha=settings.statistical_alpha,
            parent_slot="closest_parent",
        )
        statistical_report_path = statistical_paths["statistical_report"]

    return {
        "morphology_progression": morphology_plot_path,
        "fitness_over_generations": fitness_plot_path,
        "parent_child_fitness_density": density_plot_path,
        "statistical_report": statistical_report_path,
        "final_generation_statistical_report": final_generation_statistical_report_path,
    }


def build_db_path(settings: ExperimentSettings) -> Path:
    return (
        ROOT
        / settings.out_path
        / settings.study_name
        / settings.experiment_name
        / f"run_{settings.run}"
        / f"run_{settings.run}"
    )


def load_robot_record(
    db_path: Path,
    generation: Optional[int] = None,
    robot_id: Optional[int] = None,
    metric: str = "fitness",
):
    if not db_path.exists():
        raise FileNotFoundError(
            "Could not render a robot because the experiment database was not found:\n"
            f"  {db_path}\n"
            "Run the experiment first with `run_experiment = 1`, or set "
            "`study_name`, `experiment_name`, and `run` to an existing run."
        )

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if robot_id is not None:
        row = cur.execute(
            """
            SELECT r.robot_id, r.genome, r.born_generation, r.valid, r.displacement,
                   gs.generation, gs.fitness, gs.novelty, gs.novelty_weighted
            FROM all_robots r
            LEFT JOIN generation_survivors gs ON gs.robot_id = r.robot_id
            WHERE r.robot_id = ?
            ORDER BY gs.generation DESC
            LIMIT 1
            """,
            (robot_id,),
        ).fetchone()
    else:
        if generation is None:
            generation = cur.execute(
                "SELECT MAX(generation) FROM generation_survivors"
            ).fetchone()[0]

        if generation is None:
            raise RuntimeError(f"No survivors found in DB: {db_path}")

        metric_expr = {
            "fitness": "gs.fitness",
            "novelty": "gs.novelty",
            "novelty_weighted": "gs.novelty_weighted",
            "displacement": "r.displacement",
        }[metric]

        row = cur.execute(
            f"""
            SELECT r.robot_id, r.genome, r.born_generation, r.valid, r.displacement,
                   gs.generation, gs.fitness, gs.novelty, gs.novelty_weighted
            FROM generation_survivors gs
            JOIN all_robots r ON r.robot_id = gs.robot_id
            WHERE gs.generation = ?
            ORDER BY {metric_expr} DESC
            LIMIT 1
            """,
            (generation,),
        ).fetchone()

    con.close()

    if row is None:
        raise RuntimeError("Could not find a robot matching the requested selection.")

    record = dict(row)
    if isinstance(record["genome"], str):
        record["genome"] = json.loads(record["genome"])
    return record


def phenotype_to_materials(cells):
    materials = np.zeros(cells.shape, dtype=int)
    for idx, value in np.ndenumerate(cells):
        materials[idx] = value.voxel_type if value != 0 else 0
    return materials


def build_render_args(settings: ExperimentSettings) -> SimpleNamespace:
    return SimpleNamespace(
        out_path=str(ROOT / settings.out_path),
        study_name=settings.study_name,
        experiment_name=settings.experiment_name,
        run=settings.run,
        voxel_types=settings.voxel_types,
        evogym_steps=settings.evogym_steps,
        evogym_env_name=resolve_evogym_env_name(settings),
        evogym_num_workers=1,
        evogym_init_x=settings.evogym_init_x,
        evogym_init_y=settings.evogym_init_y,
        evogym_action_bias=settings.evogym_action_bias,
        evogym_action_amplitude=settings.evogym_action_amplitude,
        evogym_ann_hidden_size=settings.evogym_ann_hidden_size,
        evogym_sine_period=settings.evogym_sine_period,
        evogym_sine_mix=settings.evogym_sine_mix,
        evogym_headless=settings.render_headless,
        evogym_render_mode=settings.render_render_mode,
        evogym_isolate_tasks=0,
        evogym_record_video=settings.record_video,
        evogym_record_video_fps=settings.record_video_fps,
        evogym_record_video_stride=settings.record_video_stride,
        evogym_record_video_path="",
        evogym_left_wall=settings.evogym_left_wall,
        evogym_left_wall_height=settings.evogym_left_wall_height,
        evogym_flat_ceiling_gap_blocks=settings.evogym_flat_ceiling_gap_blocks,
        evogym_flat_ceiling_width=settings.evogym_flat_ceiling_width,
        ustatic=settings.ustatic,
        udynamic=settings.udynamic,
    )


def render_robot(settings: ExperimentSettings):
    db_path = build_db_path(settings)
    record = load_robot_record(
        db_path=db_path,
        generation=settings.render_generation,
        robot_id=settings.render_robot_id,
        metric=settings.render_metric,
    )

    cells = GRN(
        max_voxels=settings.max_voxels,
        cube_face_size=settings.cube_face_size,
        genotype=record["genome"],
        voxel_types=settings.voxel_types,
        env_conditions="",
        plastic=0,
    ).develop()
    phenotype = phenotype_to_materials(cells)

    individual = SimpleNamespace(
        id=int(record["robot_id"]),
        genome=record["genome"],
        phenotype=phenotype,
        valid=bool(record["valid"]),
    )

    args = build_render_args(settings)
    if settings.record_video:
        video_dir = (
            ROOT
            / settings.out_path
            / settings.study_name
            / settings.experiment_name
            / f"run_{settings.run}"
            / "videos"
        )
        video_name = settings.record_video_name or f"robot_{record['robot_id']}.gif"
        args.evogym_record_video_path = str(video_dir / video_name)
    prepare_robot_files(individual, args)

    print(
        f"Rendering robot_id={record['robot_id']} generation={record['generation']} "
        f"fitness={record['fitness']} displacement={record['displacement']}"
    )
    simulate_evogym_batch([individual], args)
    print(f"Replay displacement: {getattr(individual, 'displacement', None)}")
    return individual


def iter_render_settings(settings: ExperimentSettings):
    raw_targets = str(getattr(settings, "render_targets", "") or "").strip()
    if not raw_targets:
        yield build_run_settings(
            settings,
            crossover_type=settings.crossover_type,
            run=settings.run,
        )
        return

    for raw_target in raw_targets.split(","):
        raw_target = raw_target.strip()
        if not raw_target:
            continue
        parts = [part.strip() for part in raw_target.split(":")]
        if len(parts) not in (3, 4):
            raise ValueError(
                "render_targets entries must be "
                "crossover_type:run:robot_id[:video_name]"
            )
        crossover_type, run, robot_id = parts[:3]
        video_name = parts[3] if len(parts) == 4 and parts[3] else settings.record_video_name
        yield replace(
            build_run_settings(
                settings,
                crossover_type=crossover_type,
                run=int(run),
            ),
            render_robot_id=int(robot_id),
            record_video_name=video_name,
        )


def main():
    # Edit the values above, then run this file from your IDE or with:
    # python run_scripts/run_ea.py
    # This file is the full crossover-stack runner. Use run_smoke_ea.py for a
    # quick one-run check.
    settings = ExperimentSettings()

    if settings.run_experiment:
        run_crossover_stack(settings)

    if settings.run_experiment and settings.analyze_after_run:
        output_paths = run_stack_analysis(settings)
        print(f"Analysis figure saved to: {output_paths['morphology_progression']}")
        print(f"Fitness-over-generations figure saved to: {output_paths['fitness_over_generations']}")
        print(f"Parent-child morphology/fitness figure saved to: {output_paths['parent_child_fitness_density']}")
        if output_paths["final_generation_statistical_report"] is not None:
            print(
                "Final-generation population statistical report saved to: "
                f"{output_paths['final_generation_statistical_report']}"
            )
        if output_paths["statistical_report"] is not None:
            print(f"Statistical report saved to: {output_paths['statistical_report']}")

    if settings.render_only or settings.render_after_run:
        for render_settings in iter_render_settings(settings):
            render_robot(render_settings)


if __name__ == "__main__":
    main()
