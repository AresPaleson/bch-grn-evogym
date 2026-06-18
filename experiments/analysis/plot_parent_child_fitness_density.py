#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

try:
    from sqlalchemy import create_engine, select
    from algorithms.EA_classes import Robot
    DB_DEPENDENCY_ERROR = None
except ModuleNotFoundError as exc:
    create_engine = None
    select = None
    Robot = None
    DB_DEPENDENCY_ERROR = exc

from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE
from utils.body_metrics import compute_body_metrics_from_phenotype, develop_body_from_genome
from crossover_labels import (
    CROSSOVER_COLORS as CROSSOVER_COLOR_MAP,
    CROSSOVER_ORDER,
    display_crossover_name,
    infer_crossover_type,
)


COLORS = {
    "ink": "#1E2430",
    "red": "#C53A3A",
    "bg": "#FCFCFA",
    "grid": "#D9DDE3",
    "gray": "#9AA3AD",
}
MIN_RELEVANT_DISPLACEMENT = -5.0

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "robot_density",
    ["#F3F1EC", "#C2D0DD", "#6E879D", "#26445A", "#0E1F2D"],
)

CROSSOVER_TYPES = CROSSOVER_ORDER

CROSSOVER_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#9C755F",
]

CROSSOVER_DENSITY_CMAPS = {
    crossover: LinearSegmentedColormap.from_list(
        f"{crossover}_density",
        ["#F7F7F2", CROSSOVER_COLOR_MAP[crossover]],
    )
    for crossover in CROSSOVER_ORDER
}

MORPH_DISTANCE_METRICS = (
    "size",
    "proportion",
    "coverage",
    "symmetry",
    "relative_number_of_joints",
    "relative_number_of_limbs",
)

TRAIT_DISTANCE_METRICS = (
    "num_voxels",
    "proportion",
    "coverage",
    "symmetry",
    "relative_number_of_joints",
    "relative_number_of_limbs",
    "bounding_box_area",
    "environmental_contact_area",
    "bone_prop",
    "fat_prop",
    "fat2_prop",
    "phase_muscle_prop",
    "offphase_muscle_prop",
)

SUMMARY_VALUE_METRICS = (
    "morph_distance",
    "shape_distance",
    "occupied_union_material_distance",
    "material_substitution_distance",
    "trait_profile_distance",
    "parent_pair_morph_distance",
    "child_fitness",
    "fitness_delta",
)

def parse_csv_arg(raw, cast=str):
    if raw is None or raw == "":
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def discover_experiments(study_root):
    return sorted(
        entry.name
        for entry in study_root.iterdir()
        if entry.is_dir() and entry.name != "analysis"
    )


def discover_runs(experiment_root):
    runs = []
    for entry in experiment_root.iterdir():
        if entry.is_dir() and entry.name.startswith("run_"):
            try:
                runs.append(int(entry.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(runs)


def resolve_db_path(study_root, experiment, run):
    db_path = study_root / experiment / f"run_{run}" / f"run_{run}"
    return db_path if db_path.exists() else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fitness versus parent-child morphological distance as density panels."
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
    parser.add_argument("--analysis-dir", default="", type=str)
    parser.add_argument(
        "--experiments",
        default="",
        type=str,
        help="Comma-separated experiment names. Empty means: analyze every experiment in the study.",
    )
    parser.add_argument(
        "--runs",
        default="",
        type=str,
        help="Comma-separated run numbers. Empty means: analyze every discovered run per experiment.",
    )
    parser.add_argument("--valid-only", default=1, type=int)
    parser.add_argument("--max-voxels", default=64, type=int)
    parser.add_argument("--cube-face-size", default=10, type=int)
    parser.add_argument("--voxel-types", default="withbone", type=str)
    parser.add_argument("--env-conditions", default="", type=str)
    parser.add_argument("--plastic", default=0, type=int)
    parser.add_argument("--fitness-metric", default="displacement", type=str)
    return parser.parse_args()


def resolve_robot_fitness_metric(fitness_metric):
    requested = (fitness_metric or "displacement").strip()
    if requested in Robot.__table__.columns:
        return requested

    print(
        f"[warn] Fitness metric '{requested}' is not stored for every robot; "
        "using displacement for the parent-child plot."
    )
    return "displacement"


def metric_label(metric):
    return metric.replace("_", " ").title()


def voxel_type_map(voxel_types: str):
    if voxel_types == "withbone":
        return VOXEL_TYPES
    if voxel_types == "nobone":
        return VOXEL_TYPES_NOBONE
    raise ValueError(f"Unsupported voxel_types: {voxel_types}")


def material_profile_from_phenotype(phenotype, voxel_types: str) -> dict[str, float]:
    grid = np.asarray(phenotype, dtype=int)
    occupied_total = int((grid != 0).sum())
    profile = {}
    for name, material_id in voxel_type_map(voxel_types).items():
        count = int((grid == material_id).sum())
        profile[f"{name}_count"] = float(count)
        profile[f"{name}_prop"] = float(count / occupied_total) if occupied_total else 0.0
    for name in ("bone", "fat", "fat2", "phase_muscle", "offphase_muscle"):
        profile.setdefault(f"{name}_count", 0.0)
        profile.setdefault(f"{name}_prop", 0.0)
    profile["muscle_prop"] = profile["phase_muscle_prop"] + profile["offphase_muscle_prop"]
    return profile


def body_trait_profile(
    phenotype,
    voxel_types: str,
    max_voxels: Optional[int] = None,
) -> dict[str, float]:
    grid = np.asarray(phenotype, dtype=int)
    traits = compute_body_metrics_from_phenotype(
        grid,
        voxel_types=voxel_types,
        max_voxels=max_voxels,
    )
    traits["num_voxels"] = float((grid != 0).sum())
    traits.update(material_profile_from_phenotype(grid, voxel_types=voxel_types))
    return traits


def normalized_abs_diff(child_value, parent_value):
    if child_value is None or parent_value is None:
        return np.nan
    try:
        child_value = float(child_value)
        parent_value = float(parent_value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(child_value) or not np.isfinite(parent_value):
        return np.nan
    denominator = max(abs(child_value), abs(parent_value), 1e-9)
    return abs(child_value - parent_value) / denominator


def trait_profile_distance(child_traits: dict[str, float], parent_traits: dict[str, float]):
    distances = [
        normalized_abs_diff(child_traits.get(metric), parent_traits.get(metric))
        for metric in TRAIT_DISTANCE_METRICS
    ]
    distances = [value for value in distances if np.isfinite(value)]
    if not distances:
        return np.nan
    return float(np.mean(distances))


def shape_distance(child_phenotype, parent_phenotype):
    child_grid = np.asarray(child_phenotype, dtype=int)
    parent_grid = np.asarray(parent_phenotype, dtype=int)
    if child_grid.shape != parent_grid.shape:
        raise ValueError(f"Shape mismatch: {child_grid.shape} vs {parent_grid.shape}")

    child_occupied = child_grid != 0
    parent_occupied = parent_grid != 0
    scale = float(child_grid.size)
    raw_distance = float((child_occupied != parent_occupied).sum())
    return (raw_distance / scale if scale else 0.0), raw_distance, scale


def occupied_union_material_distance(child_phenotype, parent_phenotype):
    child_grid = np.asarray(child_phenotype, dtype=int)
    parent_grid = np.asarray(parent_phenotype, dtype=int)
    if child_grid.shape != parent_grid.shape:
        raise ValueError(f"Shape mismatch: {child_grid.shape} vs {parent_grid.shape}")

    occupied_union = (child_grid != 0) | (parent_grid != 0)
    scale = float(occupied_union.sum())
    if scale <= 0:
        return 0.0, 0.0, 0.0
    raw_distance = float((child_grid[occupied_union] != parent_grid[occupied_union]).sum())
    return raw_distance / scale, raw_distance, scale


def material_substitution_distance(child_phenotype, parent_phenotype):
    child_grid = np.asarray(child_phenotype, dtype=int)
    parent_grid = np.asarray(parent_phenotype, dtype=int)
    if child_grid.shape != parent_grid.shape:
        raise ValueError(f"Shape mismatch: {child_grid.shape} vs {parent_grid.shape}")

    occupied_overlap = (child_grid != 0) & (parent_grid != 0)
    scale = float(occupied_overlap.sum())
    if scale <= 0:
        return np.nan, 0.0, 0.0
    raw_distance = float((child_grid[occupied_overlap] != parent_grid[occupied_overlap]).sum())
    return raw_distance / scale, raw_distance, scale


def load_run_tables(db_path, experiment, run, fitness_metric):
    resolved_fitness_metric = resolve_robot_fitness_metric(fitness_metric)
    fitness_col = getattr(Robot, resolved_fitness_metric).label("child_fitness")

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.connect() as conn:
        robots_df = pd.read_sql(
            select(
                Robot.robot_id,
                Robot.born_generation,
                Robot.valid,
                Robot.parent1_id,
                Robot.parent2_id,
                Robot.genome,
                Robot.displacement,
                fitness_col,
            ),
            conn,
        )

    robots_df["experiment"] = experiment
    robots_df["run"] = run
    robots_df["crossover_type"] = infer_crossover_type(experiment)
    robots_df["fitness_metric"] = resolved_fitness_metric
    return robots_df


def build_robot_state_map(robots_df, args):
    robot_state = {}
    for row in robots_df.itertuples(index=False):
        genome = row.genome
        if isinstance(genome, str):
            genome = json.loads(genome)

        phenotype = develop_body_from_genome(
            genome,
            max_voxels=args.max_voxels,
            cube_face_size=args.cube_face_size,
            voxel_types=args.voxel_types,
            env_conditions=args.env_conditions,
            plastic=args.plastic,
        )
        traits = body_trait_profile(
            phenotype,
            voxel_types=args.voxel_types,
            max_voxels=args.max_voxels,
        )
        num_voxels = int((phenotype != 0).sum())
        fitness_value = row.child_fitness
        if fitness_value is not None and not np.isfinite(fitness_value):
            fitness_value = np.nan
        robot_state[(row.experiment, row.run, int(row.robot_id))] = {
            "phenotype": phenotype,
            "num_voxels": num_voxels,
            "traits": traits,
            "valid": float(row.valid) if row.valid is not None else np.nan,
            "fitness": fitness_value,
            "born_generation": row.born_generation,
            "parent1_id": row.parent1_id,
            "parent2_id": row.parent2_id,
        }
    return robot_state


def morphology_distance(child_traits: dict[str, float], parent_traits: dict[str, float]):
    child_vector = []
    parent_vector = []
    for metric in MORPH_DISTANCE_METRICS:
        child_value = child_traits.get(metric)
        parent_value = parent_traits.get(metric)
        if child_value is None or parent_value is None:
            return np.nan, np.nan, np.nan
        try:
            child_value = float(child_value)
            parent_value = float(parent_value)
        except (TypeError, ValueError):
            return np.nan, np.nan, np.nan
        if not np.isfinite(child_value) or not np.isfinite(parent_value):
            return np.nan, np.nan, np.nan
        child_vector.append(child_value)
        parent_vector.append(parent_value)

    raw_distance = float(
        np.linalg.norm(np.asarray(child_vector) - np.asarray(parent_vector))
    )
    max_distance = float(np.sqrt(len(MORPH_DISTANCE_METRICS)))
    return raw_distance / max_distance, raw_distance, max_distance


def fitness_delta(child_fitness, parent_fitness):
    if pd.isna(child_fitness) or pd.isna(parent_fitness):
        return np.nan
    if not np.isfinite(child_fitness) or not np.isfinite(parent_fitness):
        return np.nan
    return child_fitness - parent_fitness


def parent_pair_distances(row, robot_state, valid_only):
    parent1_id = getattr(row, "parent1_id")
    parent2_id = getattr(row, "parent2_id")
    if pd.isna(parent1_id) or pd.isna(parent2_id):
        return {
            "parent_pair_morph_distance": np.nan,
            "parent_pair_shape_distance": np.nan,
            "parent_pair_trait_profile_distance": np.nan,
        }

    parent1 = robot_state.get((row.experiment, row.run, int(parent1_id)))
    parent2 = robot_state.get((row.experiment, row.run, int(parent2_id)))
    if parent1 is None or parent2 is None:
        return {
            "parent_pair_morph_distance": np.nan,
            "parent_pair_shape_distance": np.nan,
            "parent_pair_trait_profile_distance": np.nan,
        }
    if valid_only and (parent1["valid"] != 1 or parent2["valid"] != 1):
        return {
            "parent_pair_morph_distance": np.nan,
            "parent_pair_shape_distance": np.nan,
            "parent_pair_trait_profile_distance": np.nan,
        }

    morph_distance, _, _ = morphology_distance(parent1["traits"], parent2["traits"])
    parent_shape_distance, _, _ = shape_distance(parent1["phenotype"], parent2["phenotype"])
    parent_trait_distance = trait_profile_distance(parent1["traits"], parent2["traits"])
    return {
        "parent_pair_morph_distance": morph_distance,
        "parent_pair_shape_distance": parent_shape_distance,
        "parent_pair_trait_profile_distance": parent_trait_distance,
    }


def build_parent_child_links(robots_df, robot_state, valid_only):
    rows = []

    for row in robots_df.itertuples(index=False):
        child_key = (row.experiment, row.run, int(row.robot_id))
        child = robot_state.get(child_key)
        if child is None:
            continue
        if valid_only and child["valid"] != 1:
            continue

        pair_distances = parent_pair_distances(row, robot_state, valid_only)

        for slot in ("parent1_id", "parent2_id"):
            parent_id = getattr(row, slot)
            if pd.isna(parent_id):
                continue

            parent_key = (row.experiment, row.run, int(parent_id))
            parent = robot_state.get(parent_key)
            if parent is None:
                continue
            if valid_only and parent["valid"] != 1:
                continue

            morph_distance, raw_distance, distance_scale = morphology_distance(
                child["traits"],
                parent["traits"],
            )
            child_shape_distance, child_shape_distance_raw, child_shape_distance_scale = shape_distance(
                child["phenotype"],
                parent["phenotype"],
            )
            child_union_material_distance, child_union_material_raw, child_union_material_scale = (
                occupied_union_material_distance(
                    child["phenotype"],
                    parent["phenotype"],
                )
            )
            child_material_substitution_distance, child_material_substitution_raw, child_material_substitution_scale = (
                material_substitution_distance(
                    child["phenotype"],
                    parent["phenotype"],
                )
            )
            child_trait_distance = trait_profile_distance(child["traits"], parent["traits"])
            size_abs_diff = abs(child["traits"]["num_voxels"] - parent["traits"]["num_voxels"])
            size_norm_abs_diff = normalized_abs_diff(
                child["traits"]["num_voxels"],
                parent["traits"]["num_voxels"],
            )

            rows.append(
                {
                    "experiment": row.experiment,
                    "run": row.run,
                    "crossover_type": row.crossover_type,
                    "fitness_metric": row.fitness_metric,
                    "child_id": int(row.robot_id),
                    "child_generation": child["born_generation"],
                    "parent_id": int(parent_id),
                    "parent_slot": "parent1" if slot == "parent1_id" else "parent2",
                    "child_fitness": child["fitness"],
                    "parent_fitness": parent["fitness"],
                    "fitness_delta": fitness_delta(child["fitness"], parent["fitness"]),
                    "morph_distance": morph_distance,
                    "morph_distance_raw": raw_distance,
                    "morph_distance_scale": distance_scale,
                    "shape_distance": child_shape_distance,
                    "shape_distance_raw": child_shape_distance_raw,
                    "shape_distance_scale": child_shape_distance_scale,
                    "occupied_union_material_distance": child_union_material_distance,
                    "occupied_union_material_distance_raw": child_union_material_raw,
                    "occupied_union_material_distance_scale": child_union_material_scale,
                    "material_substitution_distance": child_material_substitution_distance,
                    "material_substitution_distance_raw": child_material_substitution_raw,
                    "material_substitution_distance_scale": child_material_substitution_scale,
                    "trait_profile_distance": child_trait_distance,
                    "size_abs_diff": size_abs_diff,
                    "size_norm_abs_diff": size_norm_abs_diff,
                    "symmetry_abs_diff": abs(child["traits"]["symmetry"] - parent["traits"]["symmetry"]),
                    "coverage_abs_diff": abs(
                        child["traits"]["coverage"]
                        - parent["traits"]["coverage"]
                    ),
                    "proportion_abs_diff": abs(
                        child["traits"]["proportion"]
                        - parent["traits"]["proportion"]
                    ),
                    "relative_joints_abs_diff": abs(
                        child["traits"]["relative_number_of_joints"]
                        - parent["traits"]["relative_number_of_joints"]
                    ),
                    "relative_limbs_abs_diff": abs(
                        child["traits"]["relative_number_of_limbs"]
                        - parent["traits"]["relative_number_of_limbs"]
                    ),
                    "muscle_prop_abs_diff": abs(
                        child["traits"]["muscle_prop"] - parent["traits"]["muscle_prop"]
                    ),
                    **pair_distances,
                }
            )

    links_df = pd.DataFrame(rows)
    if links_df.empty:
        return links_df

    closest_df = (
        links_df.sort_values(
            ["experiment", "run", "child_id", "morph_distance", "parent_slot", "parent_id"]
        )
        .groupby(["experiment", "run", "child_id"], as_index=False)
        .first()
    )
    closest_df["closest_parent_slot"] = closest_df["parent_slot"]
    closest_df["parent_slot"] = "closest_parent"
    links_df["closest_parent_slot"] = ""
    return pd.concat([links_df, closest_df], ignore_index=True)


def style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["gray"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.45,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "savefig.facecolor": COLORS["bg"],
            "savefig.dpi": 300,
        }
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def plot_regression(ax, df, x_col, y_col):
    fit_df = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(fit_df) < 2 or fit_df[x_col].nunique() < 2:
        return

    slope, intercept = np.polyfit(fit_df[x_col], fit_df[y_col], 1)
    x_vals = np.linspace(fit_df[x_col].min(), fit_df[x_col].max(), 200)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, color=COLORS["red"], linewidth=2.3)


def clean_plot_frame(df, x_col, y_col):
    plot_df = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        return plot_df

    finite_mask = np.isfinite(plot_df[x_col].to_numpy()) & np.isfinite(plot_df[y_col].to_numpy())
    return plot_df.loc[finite_mask].copy()


def clip_plot_frame(plot_df, x_col, y_col, xlim, ylim):
    clipped = plot_df[
        (plot_df[x_col] >= xlim[0])
        & (plot_df[x_col] <= xlim[1])
    ].copy()
    if ylim is not None:
        clipped = clipped[
            (clipped[y_col] >= ylim[0])
            & (clipped[y_col] <= ylim[1])
        ].copy()
    return clipped


def smooth_density(density, passes=2):
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel = kernel / kernel.sum()
    smoothed = density.astype(float)
    for _ in range(passes):
        padded_x = np.pad(smoothed, ((2, 2), (0, 0)), mode="edge")
        smoothed = np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"),
            axis=0,
            arr=padded_x,
        )
        padded_y = np.pad(smoothed, ((0, 0), (2, 2)), mode="edge")
        smoothed = np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"),
            axis=1,
            arr=padded_y,
        )
    return smoothed


def plot_density_axis(
    ax,
    df,
    x_col,
    y_col,
    title,
    ylabel,
    *,
    xlim=(0, 0.65),
    ylim=None,
    show_colorbar=True,
    xlabel="Euclidean distance",
    title_fontsize=12,
    label_fontsize=11,
    tick_fontsize=10,
    density_cmap=DENSITY_CMAP,
):
    plot_df = clip_plot_frame(
        clean_plot_frame(df, x_col, y_col),
        x_col,
        y_col,
        xlim,
        ylim,
    )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)
    clean_axes(ax)
    ax.tick_params(labelsize=tick_fontsize)

    if plot_df.empty:
        ax.set_title(f"{title} (Corr: NA, n=0)", fontsize=title_fontsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=COLORS["gray"])
        return

    x = plot_df[x_col].to_numpy()
    y = plot_df[y_col].to_numpy()

    bins = 32
    hist_range = [xlim, ylim] if ylim is not None else [xlim, [float(np.min(y)), float(np.max(y))]]
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=hist_range)
    if np.nanmax(hist) > 0:
        density = smooth_density(hist.T, passes=2)
        density = density / np.nanmax(density)
        density = np.ma.masked_where(density <= 0.015, density)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        levels = np.linspace(0.0, 1.0, 9)
        mesh = ax.contourf(
            x_centers,
            y_centers,
            density,
            levels=levels,
            cmap=density_cmap,
            vmin=0,
            vmax=1,
            alpha=0.92,
        )
        if show_colorbar:
            cbar = plt.colorbar(mesh, ax=ax, shrink=0.94, pad=0.02)
            cbar.set_label("Relative density", fontsize=label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)

    plot_regression(ax, plot_df, x_col, y_col)

    corr = plot_df[x_col].corr(plot_df[y_col])
    corr_text = f"{corr:.2f}" if pd.notna(corr) else "NA"
    ax.set_title(f"{title} (Corr: {corr_text}, n={len(plot_df)})", fontsize=title_fontsize)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_all_crossover_density_grid(links_df, output_path, fitness_metric="displacement"):
    if fitness_metric == "displacement":
        links_df = links_df[
            (links_df["child_fitness"] >= MIN_RELEVANT_DISPLACEMENT)
            & (links_df["parent_fitness"] >= MIN_RELEVANT_DISPLACEMENT)
        ].copy()

    y_label = metric_label(fitness_metric)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(CROSSOVER_TYPES),
        figsize=(21.6, 14.2),
        sharex=True,
    )

    for col, crossover in enumerate(CROSSOVER_TYPES):
        crossover_df = links_df[links_df["crossover_type"] == crossover].copy()
        all_parent_df = crossover_df[
            crossover_df["parent_slot"].isin(["parent1", "parent2"])
        ].copy()
        closest_df = crossover_df[crossover_df["parent_slot"] == "closest_parent"].copy()
        crossover_label = display_crossover_name(crossover)

        plot_density_axis(
            axes[0, col],
            all_parent_df,
            "morph_distance",
            "child_fitness",
            f"{crossover_label}\nChild {y_label} vs Both Parents",
            f"Child {y_label}",
            ylim=(0, 100),
            xlabel="",
            title_fontsize=18,
            label_fontsize=17,
            tick_fontsize=15,
            density_cmap=CROSSOVER_DENSITY_CMAPS[crossover],
        )
        plot_density_axis(
            axes[1, col],
            closest_df,
            "morph_distance",
            "child_fitness",
            "Child vs Closest Parent",
            f"Child {y_label}",
            ylim=(0, 100),
            xlabel="",
            title_fontsize=18,
            label_fontsize=17,
            tick_fontsize=15,
            density_cmap=CROSSOVER_DENSITY_CMAPS[crossover],
        )
        plot_density_axis(
            axes[2, col],
            closest_df,
            "morph_distance",
            "fitness_delta",
            "Child-Parent Change vs Closest Parent",
            f"Child - Parent {y_label}",
            ylim=(-90, 40),
            xlabel="",
            title_fontsize=18,
            label_fontsize=17,
            tick_fontsize=15,
            density_cmap=CROSSOVER_DENSITY_CMAPS[crossover],
        )

    for col in range(1, len(CROSSOVER_TYPES)):
        for row in range(3):
            axes[row, col].set_ylabel("")

    fig.suptitle(
        "Parent-Child Fitness-Distance Density by Crossover Type",
        fontsize=24,
        y=0.995,
    )
    fig.supxlabel("Euclidean distance", fontsize=20, color=COLORS["ink"], y=0.018)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def finite_frame(df, columns):
    clean = df.copy()
    clean[columns] = clean[columns].replace([np.inf, -np.inf], np.nan)
    return clean


def summarize_distance_links(df, group_cols):
    base_columns = [
        *group_cols,
        "pair_count",
        "child_count",
        "run_count",
    ]
    metric_columns = []
    for metric in SUMMARY_VALUE_METRICS:
        metric_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_std",
                f"{metric}_q25",
                f"{metric}_q75",
            ]
        )
    extra_columns = [
        "morph_distance_min",
        "morph_distance_max",
        "morph_distance_raw_mean",
        "shape_distance_raw_mean",
        "occupied_union_material_distance_raw_mean",
        "material_substitution_distance_raw_mean",
        "size_norm_abs_diff_mean",
        "symmetry_abs_diff_mean",
        "proportion_abs_diff_mean",
        "coverage_abs_diff_mean",
        "relative_joints_abs_diff_mean",
        "relative_limbs_abs_diff_mean",
        "muscle_prop_abs_diff_mean",
    ]
    columns = base_columns + metric_columns + extra_columns
    if df.empty:
        return pd.DataFrame(columns=columns)

    clean = finite_frame(
        df,
        [
            *SUMMARY_VALUE_METRICS,
            "morph_distance_raw",
            "shape_distance_raw",
            "occupied_union_material_distance_raw",
            "material_substitution_distance_raw",
            "size_norm_abs_diff",
            "symmetry_abs_diff",
            "proportion_abs_diff",
            "coverage_abs_diff",
            "relative_joints_abs_diff",
            "relative_limbs_abs_diff",
            "muscle_prop_abs_diff",
        ],
    )
    agg_spec = {
        "pair_count": ("morph_distance", "count"),
        "child_count": ("child_id", "nunique"),
        "run_count": ("run", "nunique"),
    }
    for metric in SUMMARY_VALUE_METRICS:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_median"] = (metric, "median")
        agg_spec[f"{metric}_std"] = (metric, lambda x: x.dropna().std(ddof=0))
        agg_spec[f"{metric}_q25"] = (metric, lambda x: x.dropna().quantile(0.25))
        agg_spec[f"{metric}_q75"] = (metric, lambda x: x.dropna().quantile(0.75))

    agg_spec.update(
        {
            "morph_distance_min": ("morph_distance", "min"),
            "morph_distance_max": ("morph_distance", "max"),
            "morph_distance_raw_mean": ("morph_distance_raw", "mean"),
            "shape_distance_raw_mean": ("shape_distance_raw", "mean"),
            "occupied_union_material_distance_raw_mean": ("occupied_union_material_distance_raw", "mean"),
            "material_substitution_distance_raw_mean": ("material_substitution_distance_raw", "mean"),
            "size_norm_abs_diff_mean": ("size_norm_abs_diff", "mean"),
            "symmetry_abs_diff_mean": ("symmetry_abs_diff", "mean"),
            "proportion_abs_diff_mean": ("proportion_abs_diff", "mean"),
            "coverage_abs_diff_mean": ("coverage_abs_diff", "mean"),
            "relative_joints_abs_diff_mean": ("relative_joints_abs_diff", "mean"),
            "relative_limbs_abs_diff_mean": ("relative_limbs_abs_diff", "mean"),
            "muscle_prop_abs_diff_mean": ("muscle_prop_abs_diff", "mean"),
        }
    )
    grouped = clean.groupby(group_cols, dropna=False)
    summary = grouped.agg(**agg_spec).reset_index()
    return summary[columns]


def write_distance_summaries(links_df, analysis_dir):
    summary_by_crossover = summarize_distance_links(
        links_df,
        ["crossover_type", "parent_slot"],
    )
    summary_by_experiment = summarize_distance_links(
        links_df,
        ["experiment", "crossover_type", "parent_slot"],
    )
    summary_by_run = summarize_distance_links(
        links_df,
        ["experiment", "crossover_type", "run", "parent_slot"],
    )
    summary_by_generation = summarize_distance_links(
        links_df,
        ["crossover_type", "child_generation", "parent_slot"],
    )

    paths = {
        "summary_by_crossover": analysis_dir / "parent_child_morphology_summary_by_crossover.csv",
        "summary_by_experiment": analysis_dir / "parent_child_morphology_summary_by_experiment.csv",
        "summary_by_run": analysis_dir / "parent_child_morphology_summary_by_run.csv",
        "summary_by_generation": analysis_dir / "parent_child_morphology_summary_by_generation.csv",
    }
    summary_by_crossover.to_csv(paths["summary_by_crossover"], index=False)
    summary_by_experiment.to_csv(paths["summary_by_experiment"], index=False)
    summary_by_run.to_csv(paths["summary_by_run"], index=False)
    summary_by_generation.to_csv(paths["summary_by_generation"], index=False)
    return paths


def plot_distance_by_crossover(links_df, output_path):
    closest_df = links_df[links_df["parent_slot"] == "closest_parent"].copy()
    closest_df["morph_distance"] = pd.to_numeric(closest_df["morph_distance"], errors="coerce")
    closest_df = closest_df[np.isfinite(closest_df["morph_distance"])]

    fig, ax = plt.subplots(figsize=(9.8, 5.3))
    clean_axes(ax)
    ax.grid(True, axis="y")
    ax.set_ylabel("Normalized morphological distance", fontsize=15)
    ax.set_xlabel("Variation operator", fontsize=15)
    ax.set_title("Closest Parent-Child Morphological Distance by Operator", fontsize=17)
    ax.tick_params(labelsize=13)

    if closest_df.empty:
        ax.text(0.5, 0.5, "No parent-child links", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    order = list(CROSSOVER_TYPES)
    values = []
    positions = []
    for idx, crossover in enumerate(order, start=1):
        series = closest_df.loc[closest_df["crossover_type"] == crossover, "morph_distance"].dropna()
        vals = series.to_numpy()
        if vals.size > 0:
            values.append(vals)
            positions.append(idx)
    labels = [display_crossover_name(crossover) for crossover in order]

    if not values:
        ax.text(
            0.5,
            0.5,
            "No data for configured crossover functions",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    box = ax.boxplot(
        values,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=0.34,
    )
    for idx, patch in enumerate(box["boxes"]):
        color_idx = positions[idx] - 1
        patch.set_facecolor(CROSSOVER_COLORS[color_idx % len(CROSSOVER_COLORS)])
        patch.set_alpha(0.72)
        patch.set_edgecolor(COLORS["ink"])
    for median in box["medians"]:
        median.set_color(COLORS["ink"])
        median.set_linewidth(1.8)

    rng = np.random.default_rng(7)
    for position, vals in zip(positions, values):
        if vals.size == 0:
            continue
        sample = vals if vals.size <= 450 else rng.choice(vals, size=450, replace=False)
        jitter = rng.normal(0, 0.028, size=sample.size)
        ax.scatter(
            np.full(sample.size, position) + jitter,
            sample,
            s=7,
            color=COLORS["ink"],
            alpha=0.16,
            linewidths=0,
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(0.45, len(order) + 0.55)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distance_by_generation(summary_path, output_path):
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[summary_df["parent_slot"] == "closest_parent"].copy()

    fig, ax = plt.subplots(figsize=(9.8, 5.3))
    clean_axes(ax)
    ax.grid(True)
    ax.set_xlabel("Child generation", fontsize=15)
    ax.set_ylabel("Mean normalized morphological distance", fontsize=15)
    ax.set_title("Closest Parent-Child Morphological Distance Over Generations", fontsize=17)
    ax.tick_params(labelsize=13)

    if summary_df.empty:
        ax.text(0.5, 0.5, "No generation summary", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    order = list(CROSSOVER_TYPES)

    for idx, crossover in enumerate(order):
        data = summary_df[summary_df["crossover_type"] == crossover].sort_values("child_generation")
        if data.empty:
            continue
        color = CROSSOVER_COLORS[idx % len(CROSSOVER_COLORS)]
        center = data["morph_distance_mean"]
        lower = data["morph_distance_q25"]
        upper = data["morph_distance_q75"]
        ax.plot(
            data["child_generation"],
            center,
            color=color,
            linewidth=2.6,
            label=display_crossover_name(crossover),
        )
        if lower.notna().any() and upper.notna().any():
            ax.fill_between(
                data["child_generation"],
                lower,
                upper,
                color=color,
                alpha=0.14,
            )

    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def regenerate_plots_from_links(analysis_dir: Path, fitness_metric: str):
    links_path = analysis_dir / "parent_child_fitness_distance_links.csv"
    if not links_path.exists():
        raise FileNotFoundError(f"Missing parent-child link CSV: {links_path}")

    links_df = pd.read_csv(links_path, low_memory=False)
    if "fitness_metric" in links_df.columns and links_df["fitness_metric"].notna().any():
        fitness_metric = links_df["fitness_metric"].dropna().iloc[0]

    crossover_grid_path = analysis_dir / "parent_child_fitness_distance_density_by_crossover.png"
    crossover_distance_path = analysis_dir / "parent_child_morphology_distance_by_crossover.png"
    generation_distance_path = analysis_dir / "parent_child_morphology_distance_by_generation.png"

    summary_paths = write_distance_summaries(links_df, analysis_dir)
    plot_all_crossover_density_grid(
        links_df,
        crossover_grid_path,
        fitness_metric=fitness_metric,
    )
    plot_distance_by_crossover(links_df, crossover_distance_path)
    plot_distance_by_generation(summary_paths["summary_by_generation"], generation_distance_path)

    for summary_path in summary_paths.values():
        print(f"Saved summary to: {summary_path}")
    print(f"Saved figure to: {crossover_grid_path}")
    print(f"Saved figure to: {crossover_distance_path}")
    print(f"Saved figure to: {generation_distance_path}")
    return crossover_grid_path


def run_density_plot(
    *,
    out_path: str = "experiments/results/tmp",
    study_name: str = "defaultstudy",
    analysis_dir: Optional[Path] = None,
    experiments_raw: str = "",
    runs_raw: str = "",
    valid_only: int = 1,
    max_voxels: int = 64,
    cube_face_size: int = 10,
    voxel_types: str = "withbone",
    env_conditions: str = "",
    plastic: int = 0,
    fitness_metric: str = "displacement",
):
    style()

    study_root = Path(out_path) / study_name
    if not study_root.exists():
        raise FileNotFoundError(f"Study directory not found: {study_root}")

    analysis_dir = analysis_dir if analysis_dir else study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    links_path = analysis_dir / "parent_child_fitness_distance_links.csv"

    if DB_DEPENDENCY_ERROR is not None:
        if links_path.exists():
            return regenerate_plots_from_links(analysis_dir, fitness_metric)
        raise RuntimeError(
            "Database dependencies are unavailable and no existing parent-child "
            f"link CSV was found. Original import error: {DB_DEPENDENCY_ERROR}"
        )

    experiments = parse_csv_arg(experiments_raw)
    if not experiments:
        experiments = discover_experiments(study_root)

    requested_runs = parse_csv_arg(runs_raw, cast=int)
    frames = []

    for experiment in experiments:
        experiment_root = study_root / experiment
        if not experiment_root.exists():
            print(f"[warn] Experiment not found, skipping: {experiment_root}")
            continue

        runs = requested_runs or discover_runs(experiment_root)
        for run in runs:
            db_path = resolve_db_path(study_root, experiment, run)
            if db_path is None:
                print(f"[warn] DB not found for {experiment} run {run}, skipping.")
                continue
            frames.append(load_run_tables(db_path, experiment, run, fitness_metric))

    if not frames and links_path.exists():
        return regenerate_plots_from_links(analysis_dir, fitness_metric)
    if not frames:
        raise RuntimeError("No experiment databases were found for the requested study/experiments/runs.")

    robots_df = pd.concat(frames, ignore_index=True)
    class Settings:
        pass

    settings = Settings()
    settings.max_voxels = max_voxels
    settings.cube_face_size = cube_face_size
    settings.voxel_types = voxel_types
    settings.env_conditions = env_conditions
    settings.plastic = plastic

    robot_state = build_robot_state_map(robots_df, settings)
    links_df = build_parent_child_links(robots_df, robot_state, valid_only=bool(valid_only))

    if links_df.empty:
        raise RuntimeError("No valid parent-child links were found to plot.")

    crossover_grid_path = analysis_dir / "parent_child_fitness_distance_density_by_crossover.png"
    crossover_distance_path = analysis_dir / "parent_child_morphology_distance_by_crossover.png"
    generation_distance_path = analysis_dir / "parent_child_morphology_distance_by_generation.png"

    links_df.to_csv(links_path, index=False)
    summary_paths = write_distance_summaries(links_df, analysis_dir)

    plot_fitness_metric = robots_df["fitness_metric"].dropna().iloc[0]
    plot_all_crossover_density_grid(
        links_df,
        crossover_grid_path,
        fitness_metric=plot_fitness_metric,
    )
    plot_distance_by_crossover(links_df, crossover_distance_path)
    plot_distance_by_generation(summary_paths["summary_by_generation"], generation_distance_path)

    print(f"Saved data to: {links_path}")
    for summary_path in summary_paths.values():
        print(f"Saved summary to: {summary_path}")
    print(f"Saved figure to: {crossover_grid_path}")
    print(f"Saved figure to: {crossover_distance_path}")
    print(f"Saved figure to: {generation_distance_path}")
    return crossover_grid_path


def main():
    args = parse_args()
    run_density_plot(
        out_path=args.out_path,
        study_name=args.study_name,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
        experiments_raw=args.experiments,
        runs_raw=args.runs,
        valid_only=args.valid_only,
        max_voxels=args.max_voxels,
        cube_face_size=args.cube_face_size,
        voxel_types=args.voxel_types,
        env_conditions=args.env_conditions,
        plastic=args.plastic,
        fitness_metric=args.fitness_metric,
    )


if __name__ == "__main__":
    main()
