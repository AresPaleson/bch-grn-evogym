#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crossover_labels import display_experiment_name


EXPERIMENT_COLORS = [
    "#3F6EA8",
    "#C99227",
    "#5FA8A2",
    "#B85757",
    "#7C6F9B",
    "#7E9C45",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fitness over generations from consolidated analysis CSVs."
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
    parser.add_argument("--analysis-dir", default="", type=str)
    parser.add_argument(
        "--experiments",
        default="",
        type=str,
        help="Comma-separated experiment names. Empty means: all experiments in the CSV.",
    )
    parser.add_argument(
        "--output-name",
        default="fitness_over_generations.png",
        type=str,
    )
    return parser.parse_args()


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def require_fitness_columns(df: pd.DataFrame):
    columns = [
        "fitness_mean_mean",
        "fitness_max_mean",
    ]
    fallback_columns = [
        "fitness_mean_median",
        "fitness_max_median",
        "fitness_max_std",
    ]
    missing = [col for col in columns if col not in df.columns]
    fallback_missing = [col for col in fallback_columns if col not in df.columns]
    if missing and fallback_missing:
        raise RuntimeError(
            "Cannot plot fitness over generations because gens_robots_outer.csv "
            f"is missing columns: {', '.join(missing)}"
        )
    available = [col for col in columns + fallback_columns if col in df.columns]
    if df[available].isna().all().all():
        raise RuntimeError("Fitness columns are present, but all values are empty.")


def set_plot_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFA",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9AA3AD",
            "axes.labelcolor": "#1E2430",
            "axes.titlecolor": "#1E2430",
            "xtick.color": "#1E2430",
            "ytick.color": "#1E2430",
            "grid.color": "#D9DDE3",
            "grid.alpha": 0.55,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "savefig.facecolor": "#FCFCFA",
            "savefig.dpi": 300,
        }
    )


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise RuntimeError(f"Missing expected plot column. Tried: {', '.join(candidates)}")


def plot_series(
    ax,
    data: pd.DataFrame,
    center_candidates: list[str],
    spread_candidates: list[str],
    label: str,
    color: str,
    linestyle: str,
):
    center_column = first_existing_column(data, center_candidates)
    spread_column = first_existing_column(data, spread_candidates)
    center = data[center_column]
    spread = data[spread_column].fillna(0.0)

    line = ax.plot(
        data["generation"],
        center,
        color=color,
        linestyle=linestyle,
        linewidth=2.4,
        label=label,
    )[0]
    if (spread > 0).any():
        ax.fill_between(
            data["generation"],
            center - spread,
            center + spread,
            color=color,
            alpha=0.16,
        )
    return line


def plot_metric_panel(
    ax,
    df: pd.DataFrame,
    experiments,
    *,
    title: str,
    center_candidates: list[str],
    spread_candidates: list[str],
):
    legend_handles = []
    legend_labels = []

    for idx_experiment, experiment in enumerate(experiments):
        data = df[df["experiment"] == experiment].sort_values("generation")
        if data.empty:
            continue

        label = display_experiment_name(experiment)
        line = plot_series(
            ax,
            data,
            center_candidates,
            spread_candidates,
            label,
            EXPERIMENT_COLORS[idx_experiment % len(EXPERIMENT_COLORS)],
            "-",
        )
        legend_handles.append(line)
        legend_labels.append(label)

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    clean_axes(ax)
    return legend_handles, legend_labels


def run_plot(*, analysis_dir: Path, experiments_raw: str = "", output_name: str = "fitness_over_generations.png"):
    outer_path = analysis_dir / "gens_robots_outer.csv"
    if not outer_path.exists():
        raise FileNotFoundError(f"Missing consolidated CSV: {outer_path}")

    df = pd.read_csv(outer_path)
    require_fitness_columns(df)

    if experiments_raw:
        experiments = [item.strip() for item in experiments_raw.split(",") if item.strip()]
    else:
        experiments = list(df["experiment"].dropna().unique())

    set_plot_style()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9.4, 7.1), sharex=True)
    handles, labels = plot_metric_panel(
        axes[0],
        df,
        experiments,
        title="Average Best Fitness Over Generations",
        center_candidates=["fitness_max_mean", "fitness_max_median"],
        spread_candidates=["fitness_max_std"],
    )
    plot_metric_panel(
        axes[1],
        df,
        experiments,
        title="Average Mean Fitness Over Generations",
        center_candidates=["fitness_mean_mean", "fitness_mean_median"],
        spread_candidates=["fitness_mean_std", "fitness_std_median"],
    )

    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, 0.995),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path = analysis_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")
    return output_path


def main():
    args = parse_args()
    analysis_dir = (
        Path(args.analysis_dir)
        if args.analysis_dir
        else Path(args.out_path) / args.study_name / "analysis"
    )
    run_plot(
        analysis_dir=analysis_dir,
        experiments_raw=args.experiments,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
