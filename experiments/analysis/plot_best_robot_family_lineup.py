#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from algorithms.EA_classes import GenerationSurvivor, Robot
from utils.body_metrics import develop_body_from_genome


COLORS = {
    "bg": "#FCFCFA",
    "ink": "#1E2430",
    "grid": "#D9DDE3",
    "gray": "#9AA3AD",
}

def build_voxel_palette(voxel_types: str):
    # Match EvoGym's renderer colors after simulation.prepare_robot_files maps
    # GRN material IDs to EvoGym material IDs.
    renderer_colors = {
        "empty": "#FFFFFF",
        "rigid": (0.15, 0.15, 0.15),
        "soft": (0.75, 0.75, 0.75),
        "h_act": (0.99215, 0.56862, 0.25763),
        "v_act": (0.25763, 0.56862, 0.99215),
    }

    if voxel_types == "nobone":
        material_colors = {
            0: renderer_colors["empty"],
            1: renderer_colors["soft"],
            2: renderer_colors["soft"],
            3: renderer_colors["h_act"],
            4: renderer_colors["h_act"],
        }
    else:
        material_colors = {
            0: renderer_colors["empty"],
            1: renderer_colors["rigid"],
            2: renderer_colors["soft"],
            3: renderer_colors["h_act"],
            4: renderer_colors["h_act"],
        }

    max_material_id = max(material_colors)
    colors = [material_colors.get(material_id, renderer_colors["empty"]) for material_id in range(max_material_id + 1)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, max_material_id + 1.5, 1.0), cmap.N)
    return cmap, norm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a family lineup image for the best robot in the final generation."
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
    parser.add_argument("--experiment-name", default="defaultexperiment", type=str)
    parser.add_argument("--run", default=1, type=int)
    parser.add_argument("--analysis-dir", default="", type=str)
    parser.add_argument("--output-name", default="best_robot_family_lineup.png", type=str)
    parser.add_argument("--voxel-types", default="withbone", type=str)
    parser.add_argument("--max-voxels", default=64, type=int)
    parser.add_argument("--cube-face-size", default=10, type=int)
    parser.add_argument("--env-conditions", default="", type=str)
    parser.add_argument("--plastic", default=0, type=int)
    return parser.parse_args()


def build_db_path(out_path: str, study_name: str, experiment_name: str, run: int) -> Path:
    return Path(out_path) / study_name / experiment_name / f"run_{run}" / f"run_{run}"


def load_rows(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.connect() as conn:
        robots_df = pd.read_sql(
            select(
                Robot.robot_id,
                Robot.born_generation,
                Robot.parent1_id,
                Robot.parent2_id,
                Robot.genome,
                Robot.valid,
            ),
            conn,
        )
        survivors_df = pd.read_sql(
            select(
                GenerationSurvivor.generation,
                GenerationSurvivor.robot_id,
                GenerationSurvivor.fitness,
            ),
            conn,
        )
    return robots_df, survivors_df


def choose_better_parent(robot_row, best_fitness_by_robot):
    candidates = []
    for parent_col in ("parent1_id", "parent2_id"):
        parent_id = robot_row[parent_col]
        if parent_id is None or (isinstance(parent_id, float) and math.isnan(parent_id)):
            continue
        candidates.append((int(parent_id), best_fitness_by_robot.get(int(parent_id), float("-inf"))))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return candidates[0][0]


def build_lineage(robots_df, survivors_df):
    if survivors_df.empty:
        raise RuntimeError("No generation survivors found in the database.")

    final_gen = int(survivors_df["generation"].max())
    final_df = survivors_df[survivors_df["generation"] == final_gen].copy()
    if final_df.empty:
        raise RuntimeError(f"No survivors found for final generation {final_gen}.")

    best_final = final_df.sort_values(["fitness", "robot_id"], ascending=[False, True]).iloc[0]
    best_fitness_by_robot = (
        survivors_df.groupby("robot_id", as_index=True)["fitness"]
        .max()
        .to_dict()
    )
    robots_by_id = robots_df.set_index("robot_id", drop=False)

    chain = []
    current_id = int(best_final["robot_id"])
    visited = set()
    while current_id not in visited and current_id in robots_by_id.index:
        visited.add(current_id)
        robot_row = robots_by_id.loc[current_id]
        best_fit = best_fitness_by_robot.get(current_id, np.nan)
        chain.append(
            {
                "robot_id": current_id,
                "generation": int(robot_row["born_generation"]),
                "fitness": float(best_fit) if best_fit == best_fit else np.nan,
                "genome": json.loads(robot_row["genome"]) if isinstance(robot_row["genome"], str) else robot_row["genome"],
            }
        )
        parent_id = choose_better_parent(robot_row, best_fitness_by_robot)
        if parent_id is None:
            break
        current_id = parent_id

    chain.reverse()
    return chain


def render_body_image(ax, phenotype, voxel_types: str):
    body = np.asarray(phenotype, dtype=int)
    occupied_rows = np.where((body != 0).any(axis=1))[0]
    occupied_cols = np.where((body != 0).any(axis=0))[0]
    if occupied_rows.size and occupied_cols.size:
        body = body[occupied_rows[0] : occupied_rows[-1] + 1, occupied_cols[0] : occupied_cols[-1] + 1]
    else:
        body = np.zeros((1, 1), dtype=int)

    voxel_cmap, voxel_norm = build_voxel_palette(voxel_types)
    ax.imshow(body.T, origin="lower", cmap=voxel_cmap, norm=voxel_norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": "white",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "savefig.facecolor": COLORS["bg"],
            "savefig.dpi": 300,
        }
    )


def run_best_robot_family_lineup(
    *,
    out_path: str = "experiments/results/tmp",
    study_name: str = "defaultstudy",
    experiment_name: str = "defaultexperiment",
    run: int = 1,
    analysis_dir: Optional[Path] = None,
    output_name: str = "best_robot_family_lineup.png",
    voxel_types: str = "withbone",
    max_voxels: int = 64,
    cube_face_size: int = 10,
    env_conditions: str = "",
    plastic: int = 0,
):
    style()

    db_path = build_db_path(out_path, study_name, experiment_name, run)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    robots_df, survivors_df = load_rows(db_path)
    lineage = build_lineage(robots_df, survivors_df)
    if not lineage:
        raise RuntimeError("Could not build a lineage for the final best robot.")

    analysis_dir = analysis_dir if analysis_dir else Path(out_path) / study_name / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / output_name

    fig, axes = plt.subplots(1, len(lineage), figsize=(3.2 * len(lineage), 4.3))
    axes = np.atleast_1d(axes)

    for idx, (ax, item) in enumerate(zip(axes, lineage)):
        phenotype = develop_body_from_genome(
            item["genome"],
            max_voxels=max_voxels,
            cube_face_size=cube_face_size,
            voxel_types=voxel_types,
            env_conditions=env_conditions,
            plastic=plastic,
        )
        render_body_image(ax, phenotype, voxel_types)
        role = "Best Robot" if idx == len(lineage) - 1 else "Ancestor"
        fit_text = f"{item['fitness']:.3f}" if item["fitness"] == item["fitness"] else "NA"
        ax.set_title(
            f"Gen {item['generation']}\nID {item['robot_id']}\nFitness {fit_text}\n{role}",
            fontsize=10,
            pad=10,
        )

    fig.suptitle("Family Lineup of the Final Best Robot", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lineup figure to: {output_path}")
    return output_path


def main():
    args = parse_args()
    run_best_robot_family_lineup(
        out_path=args.out_path,
        study_name=args.study_name,
        experiment_name=args.experiment_name,
        run=args.run,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
        output_name=args.output_name,
        voxel_types=args.voxel_types,
        max_voxels=args.max_voxels,
        cube_face_size=args.cube_face_size,
        env_conditions=args.env_conditions,
        plastic=args.plastic,
    )


if __name__ == "__main__":
    main()
