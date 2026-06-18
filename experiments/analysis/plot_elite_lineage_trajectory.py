#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crossover_labels import CROSSOVER_COLORS, CROSSOVER_LABELS, CROSSOVER_ORDER

ROOT = Path(__file__).resolve().parent.parent.parent

INK = "#1E2430"
GRID = "#D9DDE3"
BG = "#FCFCFA"


def parse_args():
    default_analysis_dir = (
        ROOT
        / "experiments"
        / "results"
        / "final2"
        / "flat_crossover_stack_25pop_25off_50gen_1000steps"
        / "analysis"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean parent-child morphology distance along the elite lineage "
            "of each run's final-generation best robot."
        )
    )
    parser.add_argument("--analysis-dir", default=str(default_analysis_dir), type=str)
    parser.add_argument("--survivors-name", default="gens_robots.csv", type=str)
    parser.add_argument(
        "--links-name",
        default="parent_child_fitness_distance_links.csv",
        type=str,
    )
    parser.add_argument(
        "--output-name",
        default="elite_lineage_morphological_trajectory.png",
        type=str,
    )
    parser.add_argument(
        "--lineage-name",
        default="elite_lineage_parent_child_distances.csv",
        type=str,
    )
    parser.add_argument(
        "--summary-name",
        default="elite_lineage_morphological_trajectory_summary.csv",
        type=str,
    )
    parser.add_argument(
        "--distance-column",
        default="morph_distance_raw",
        choices=("morph_distance_raw", "morph_distance"),
        help=(
            "morph_distance_raw is the unnormalized Euclidean distance; "
            "morph_distance is normalized by sqrt(number of morphology metrics)."
        ),
    )
    parser.add_argument(
        "--parent-choice",
        default="elite_parent",
        choices=("elite_parent", "closest_parent"),
        help=(
            "elite_parent follows the higher-fitness parent at each step; "
            "closest_parent follows the morphologically closest parent from the existing analysis."
        ),
    )
    return parser.parse_args()


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def load_inputs(analysis_dir: Path, survivors_name: str, links_name: str, distance_column: str):
    survivors_path = analysis_dir / survivors_name
    links_path = analysis_dir / links_name
    if not survivors_path.exists():
        raise FileNotFoundError(f"Survivor CSV not found: {survivors_path}")
    if not links_path.exists():
        raise FileNotFoundError(f"Parent-child link CSV not found: {links_path}")

    survivors_cols = [
        "experiment",
        "run",
        "generation",
        "robot_id",
        "fitness",
        "born_generation",
    ]
    links_cols = [
        "experiment",
        "run",
        "crossover_type",
        "child_id",
        "child_generation",
        "parent_id",
        "parent_slot",
        "parent_fitness",
        distance_column,
    ]
    survivors_df = pd.read_csv(survivors_path, usecols=survivors_cols)
    links_df = pd.read_csv(links_path, usecols=links_cols, low_memory=False)

    for col in ("run", "generation", "robot_id", "born_generation"):
        survivors_df[col] = pd.to_numeric(survivors_df[col], errors="coerce")
    for col in ("run", "child_id", "child_generation", "parent_id", distance_column):
        links_df[col] = pd.to_numeric(links_df[col], errors="coerce")

    survivors_df = survivors_df.dropna(subset=["experiment", "run", "generation", "robot_id", "fitness"])
    links_df = links_df.dropna(
        subset=[
            "experiment",
            "run",
            "crossover_type",
            "child_id",
            "child_generation",
            "parent_id",
            "parent_fitness",
            distance_column,
        ]
    )
    return survivors_df, links_df


def final_best_robots(survivors_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, run), group in survivors_df.groupby(["experiment", "run"], sort=True):
        final_generation = group["generation"].max()
        final_group = group[group["generation"] == final_generation].copy()
        final_group = final_group.sort_values(
            ["fitness", "robot_id"],
            ascending=[False, True],
        )
        best = final_group.iloc[0]
        rows.append(
            {
                "experiment": experiment,
                "run": int(run),
                "final_generation": int(final_generation),
                "final_best_robot_id": int(best["robot_id"]),
                "final_best_fitness": float(best["fitness"]),
                "final_best_born_generation": int(best["born_generation"]),
            }
        )
    return pd.DataFrame(rows)


def trace_lineages(
    final_best_df: pd.DataFrame,
    links_df: pd.DataFrame,
    distance_column: str,
    parent_choice: str,
) -> pd.DataFrame:
    link_map = build_lineage_link_map(links_df, parent_choice)

    rows = []
    for best in final_best_df.itertuples(index=False):
        current_id = int(best.final_best_robot_id)
        visited = set()
        step_from_best = 0

        while current_id not in visited:
            visited.add(current_id)
            link = link_map.get((best.experiment, int(best.run), current_id))
            if link is None:
                break

            original_generation = int(link.child_generation)
            rows.append(
                {
                    "experiment": best.experiment,
                    "run": int(best.run),
                    "crossover_type": link.crossover_type,
                    "final_generation": int(best.final_generation),
                    "final_best_robot_id": int(best.final_best_robot_id),
                    "final_best_fitness": float(best.final_best_fitness),
                    "final_best_born_generation": int(best.final_best_born_generation),
                    "child_id": int(link.child_id),
                    "parent_id": int(link.parent_id),
                    "parent_choice": parent_choice,
                    "chosen_parent_slot": link.parent_slot,
                    "original_child_generation": original_generation,
                    "generation_zero_based": original_generation - 1,
                    "step_from_final_best": step_from_best,
                    "distance": float(getattr(link, distance_column)),
                }
            )
            current_id = int(link.parent_id)
            step_from_best += 1

    lineage_df = pd.DataFrame(rows)
    if not lineage_df.empty:
        lineage_df = lineage_df.sort_values(
            ["crossover_type", "run", "generation_zero_based", "child_id"]
        )
    return lineage_df


def build_lineage_link_map(links_df: pd.DataFrame, parent_choice: str) -> dict:
    if parent_choice == "closest_parent":
        selected_df = links_df[links_df["parent_slot"] == "closest_parent"].copy()
    else:
        candidate_df = links_df[links_df["parent_slot"].isin(["parent1", "parent2"])].copy()
        candidate_df["parent_fitness"] = pd.to_numeric(
            candidate_df["parent_fitness"],
            errors="coerce",
        )
        candidate_df = candidate_df.sort_values(
            [
                "experiment",
                "run",
                "child_id",
                "parent_fitness",
                "parent_id",
            ],
            ascending=[True, True, True, False, True],
        )
        selected_df = candidate_df.drop_duplicates(
            subset=["experiment", "run", "child_id"],
            keep="first",
        )

    link_map = {}
    for row in selected_df.itertuples(index=False):
        key = (row.experiment, int(row.run), int(row.child_id))
        link_map[key] = row
    return link_map


def summarize_lineages(lineage_df: pd.DataFrame) -> pd.DataFrame:
    if lineage_df.empty:
        return pd.DataFrame(
            columns=[
                "crossover_type",
                "crossover_label",
                "generation_zero_based",
                "original_child_generation",
                "lineage_edge_count",
                "run_count",
                "parent_choice",
                "mean_distance",
                "std_distance",
                "median_distance",
                "q25_distance",
                "q75_distance",
            ]
        )

    summary = (
        lineage_df.groupby(
            [
                "crossover_type",
                "parent_choice",
                "generation_zero_based",
                "original_child_generation",
            ],
            as_index=False,
        )
        .agg(
            lineage_edge_count=("distance", "size"),
            run_count=("run", "nunique"),
            mean_distance=("distance", "mean"),
            std_distance=("distance", "std"),
            median_distance=("distance", "median"),
            q25_distance=("distance", lambda s: s.quantile(0.25)),
            q75_distance=("distance", lambda s: s.quantile(0.75)),
        )
    )
    summary["crossover_label"] = summary["crossover_type"].map(CROSSOVER_LABELS).fillna(
        summary["crossover_type"]
    )
    return summary[
        [
            "crossover_type",
            "crossover_label",
            "generation_zero_based",
            "original_child_generation",
            "lineage_edge_count",
            "run_count",
            "parent_choice",
            "mean_distance",
            "std_distance",
            "median_distance",
            "q25_distance",
            "q75_distance",
        ]
    ].sort_values(["crossover_type", "generation_zero_based"])


def plot_summary(summary_df: pd.DataFrame, output_path: Path, distance_column: str):
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    clean_axes(ax)
    ax.grid(True, axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    y_label = (
        "Mean Parent-Child Morphological Euclidean Distance"
        if distance_column == "morph_distance_raw"
        else "Mean Normalized Parent-Child Morphological Distance"
    )

    if summary_df.empty:
        ax.text(
            0.5,
            0.5,
            "No elite lineage parent-child distances found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=INK,
        )
    else:
        for crossover in CROSSOVER_ORDER:
            data = summary_df[summary_df["crossover_type"] == crossover].sort_values(
                "generation_zero_based"
            )
            if data.empty:
                continue
            color = CROSSOVER_COLORS[crossover]
            x = data["generation_zero_based"].to_numpy(dtype=float)
            y = data["mean_distance"].to_numpy(dtype=float)
            q25 = data["q25_distance"].to_numpy(dtype=float)
            q75 = data["q75_distance"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.2,
                label=CROSSOVER_LABELS[crossover],
            )
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)

    ax.set_title("Phenotypic Lineage Trajectory", fontsize=15, color=INK, pad=12)
    ax.set_xlabel("Generation Number", fontsize=11, color=INK)
    ax.set_ylabel(y_label, fontsize=11, color=INK)
    ax.tick_params(colors=INK)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_plot(
    *,
    analysis_dir: Path,
    survivors_name: str = "gens_robots.csv",
    links_name: str = "parent_child_fitness_distance_links.csv",
    output_name: str = "elite_lineage_morphological_trajectory.png",
    lineage_name: str = "elite_lineage_parent_child_distances.csv",
    summary_name: str = "elite_lineage_morphological_trajectory_summary.csv",
    distance_column: str = "morph_distance_raw",
    parent_choice: str = "elite_parent",
):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    survivors_df, links_df = load_inputs(
        analysis_dir,
        survivors_name,
        links_name,
        distance_column,
    )
    final_best_df = final_best_robots(survivors_df)
    lineage_df = trace_lineages(final_best_df, links_df, distance_column, parent_choice)
    summary_df = summarize_lineages(lineage_df)

    lineage_path = analysis_dir / lineage_name
    summary_path = analysis_dir / summary_name
    output_path = analysis_dir / output_name
    lineage_df.to_csv(lineage_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_summary(summary_df, output_path, distance_column)

    print(f"Saved lineage data to: {lineage_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved figure to: {output_path}")
    return {
        "lineage": lineage_path,
        "summary": summary_path,
        "figure": output_path,
    }


def main():
    args = parse_args()
    run_plot(
        analysis_dir=Path(args.analysis_dir),
        survivors_name=args.survivors_name,
        links_name=args.links_name,
        output_name=args.output_name,
        lineage_name=args.lineage_name,
        summary_name=args.summary_name,
        distance_column=args.distance_column,
        parent_choice=args.parent_choice,
    )


if __name__ == "__main__":
    main()
