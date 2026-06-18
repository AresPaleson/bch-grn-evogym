#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crossover_labels import CROSSOVER_COLORS, CROSSOVER_LABELS, CROSSOVER_ORDER, infer_crossover_type

ROOT = Path(__file__).resolve().parent.parent.parent

MORPHOLOGY_METRICS = (
    ("size", "Size"),
    ("symmetry", "Symmetry"),
    ("number_of_limbs", "Limbs"),
    ("bounding_box_area", "BBox Area"),
    ("morphological_density", "Coverage"),
    ("muscle_to_tissue_ratio", "Muscle Ratio"),
    ("actuation_energy_cost", "Actuation"),
    ("environmental_contact_area", "Contact"),
)

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
            "Plot mean population morphology diversity over generations for "
            "each crossover method."
        )
    )
    parser.add_argument("--analysis-dir", default=str(default_analysis_dir), type=str)
    parser.add_argument("--input-name", default="gens_robots.csv", type=str)
    parser.add_argument(
        "--output-name",
        default="mean_population_diversity_over_time.png",
        type=str,
    )
    parser.add_argument(
        "--run-summary-name",
        default="mean_population_diversity_by_run_generation.csv",
        type=str,
    )
    parser.add_argument(
        "--summary-name",
        default="mean_population_diversity_over_time_summary.csv",
        type=str,
    )
    parser.add_argument(
        "--include-diagonal",
        default=1,
        type=int,
        help=(
            "If 1, average every value in the N x N distance matrix, including "
            "zero self-distances. If 0, average only unique off-diagonal pairs."
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


def available_metrics(columns) -> list[tuple[str, str]]:
    metrics = [(metric, label) for metric, label in MORPHOLOGY_METRICS if metric in columns]
    if len(metrics) < 2:
        raise ValueError("Need at least two morphology metrics to compute pairwise diversity.")
    return metrics


def load_generation_rows(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0)
    metrics = available_metrics(header.columns)
    metric_cols = [metric for metric, _ in metrics]
    usecols = ["experiment", "run", "generation", "robot_id", *metric_cols]
    df = pd.read_csv(input_path, usecols=usecols)

    for column in ("run", "generation", "robot_id", *metric_cols):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["experiment", "run", "generation", "robot_id", *metric_cols])
    df["crossover_type"] = df["experiment"].map(infer_crossover_type)
    df["crossover_label"] = df["crossover_type"].map(CROSSOVER_LABELS).fillna(df["crossover_type"])

    first_generation = df.groupby(["experiment", "run"])["generation"].transform("min")
    df["generation_zero_based"] = (df["generation"] - first_generation).astype(int)
    return df, metrics


def add_normalized_metric_columns(df: pd.DataFrame, metrics: list[tuple[str, str]]) -> pd.DataFrame:
    df = df.copy()
    for metric, _ in metrics:
        values = df[metric].to_numpy(dtype=float)
        low = np.nanmin(values)
        high = np.nanmax(values)
        norm_col = f"{metric}_normalized"
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            df[norm_col] = 0.5
        else:
            df[norm_col] = (values - low) / (high - low)
    return df


def mean_pairwise_distance(matrix: np.ndarray, include_diagonal: bool) -> float:
    n = matrix.shape[0]
    if n < 2:
        return np.nan

    diff = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(diff, axis=2) / np.sqrt(matrix.shape[1])
    if include_diagonal:
        return float(distances.mean())

    upper = distances[np.triu_indices(n, k=1)]
    return float(upper.mean()) if upper.size else np.nan


def compute_run_generation_diversity(
    df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    include_diagonal: bool,
) -> pd.DataFrame:
    norm_cols = [f"{metric}_normalized" for metric, _ in metrics]
    rows = []

    group_cols = [
        "experiment",
        "run",
        "crossover_type",
        "crossover_label",
        "generation",
        "generation_zero_based",
    ]
    for keys, group in df.groupby(group_cols, sort=True):
        matrix = group[norm_cols].to_numpy(dtype=float)
        rows.append(
            {
                "experiment": keys[0],
                "run": int(keys[1]),
                "crossover_type": keys[2],
                "crossover_label": keys[3],
                "generation": int(keys[4]),
                "generation_zero_based": int(keys[5]),
                "population_size": int(len(group)),
                "metric_count": int(len(norm_cols)),
                "distance_matrix_average": (
                    "full_n_by_n_with_diagonal"
                    if include_diagonal
                    else "unique_off_diagonal_pairs"
                ),
                "mean_pairwise_morphological_distance": mean_pairwise_distance(
                    matrix,
                    include_diagonal=include_diagonal,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_over_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        run_df.groupby(
            ["crossover_type", "crossover_label", "generation_zero_based"],
            as_index=False,
        )
        .agg(
            run_count=("run", "nunique"),
            mean_population_size=("population_size", "mean"),
            mean_pairwise_morphological_distance=(
                "mean_pairwise_morphological_distance",
                "mean",
            ),
            std_pairwise_morphological_distance=(
                "mean_pairwise_morphological_distance",
                "std",
            ),
            q25_pairwise_morphological_distance=(
                "mean_pairwise_morphological_distance",
                lambda s: s.quantile(0.25),
            ),
            q75_pairwise_morphological_distance=(
                "mean_pairwise_morphological_distance",
                lambda s: s.quantile(0.75),
            ),
        )
    )
    return summary.sort_values(["crossover_type", "generation_zero_based"])


def plot_diversity(summary_df: pd.DataFrame, output_path: Path, include_diagonal: bool):
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    clean_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for crossover in CROSSOVER_ORDER:
        data = summary_df[summary_df["crossover_type"] == crossover].sort_values(
            "generation_zero_based"
        )
        if data.empty:
            continue

        color = CROSSOVER_COLORS[crossover]
        x = data["generation_zero_based"].to_numpy(dtype=float)
        y = data["mean_pairwise_morphological_distance"].to_numpy(dtype=float)
        q25 = data["q25_pairwise_morphological_distance"].to_numpy(dtype=float)
        q75 = data["q75_pairwise_morphological_distance"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.4,
            label=CROSSOVER_LABELS[crossover],
        )
        ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)

    average_note = "full N x N matrix" if include_diagonal else "unique off-diagonal pairs"
    ax.set_title("Mean Population Diversity Over Time", fontsize=15, color=INK, pad=12)
    ax.set_xlabel("Generation Number", fontsize=11, color=INK)
    ax.set_ylabel("Mean Pairwise Morphological Distance", fontsize=11, color=INK)
    ax.text(
        0.995,
        0.02,
        f"Averaged over {average_note}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        color=INK,
        alpha=0.75,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors=INK)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_plot(
    *,
    analysis_dir: Path,
    input_name: str = "gens_robots.csv",
    output_name: str = "mean_population_diversity_over_time.png",
    run_summary_name: str = "mean_population_diversity_by_run_generation.csv",
    summary_name: str = "mean_population_diversity_over_time_summary.csv",
    include_diagonal: bool = True,
):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df, metrics = load_generation_rows(analysis_dir / input_name)
    df = add_normalized_metric_columns(df, metrics)
    run_df = compute_run_generation_diversity(df, metrics, include_diagonal=include_diagonal)
    summary_df = summarize_over_runs(run_df)

    output_path = analysis_dir / output_name
    run_summary_path = analysis_dir / run_summary_name
    summary_path = analysis_dir / summary_name

    run_df.to_csv(run_summary_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_diversity(summary_df, output_path, include_diagonal=include_diagonal)

    print(f"Saved run-generation diversity to: {run_summary_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved figure to: {output_path}")
    return {
        "figure": output_path,
        "run_summary": run_summary_path,
        "summary": summary_path,
    }


def main():
    args = parse_args()
    run_plot(
        analysis_dir=Path(args.analysis_dir),
        input_name=args.input_name,
        output_name=args.output_name,
        run_summary_name=args.run_summary_name,
        summary_name=args.summary_name,
        include_diagonal=bool(args.include_diagonal),
    )


if __name__ == "__main__":
    main()
