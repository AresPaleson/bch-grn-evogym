#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crossover_labels import (
    CROSSOVER_COLORS,
    CROSSOVER_LABELS,
    CROSSOVER_ORDER,
    infer_crossover_type,
)


INK = "#1E2430"
GRID = "#D9DDE3"
BG = "#FCFCFA"
CROSSOVER_LINESTYLES = {
    "promoter_aligned_cut_and_splice": "-",
    "arithmetic_recombination": "--",
    "homologous_gene_block_recombination": "-.",
}
CROSSOVER_MARKERS = {
    "promoter_aligned_cut_and_splice": "o",
    "arithmetic_recombination": "s",
    "homologous_gene_block_recombination": "^",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw a four-panel chart with population diversity, morphology step "
            "size, symmetry, and environmental contact area."
        )
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
    parser.add_argument("--analysis-dir", default="", type=str)
    parser.add_argument(
        "--output-name",
        default="diversity_step_symmetry_contact_panel_chart.png",
        type=str,
    )
    return parser.parse_args()


def set_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": "white",
            "axes.edgecolor": "#9AA3AD",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.alpha": 0.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "savefig.facecolor": BG,
            "savefig.dpi": 300,
        }
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def add_panel_label(ax, label: str):
    ax.text(
        0.018,
        0.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=24,
        fontweight="bold",
        color=INK,
        bbox={
            "boxstyle": "square,pad=0.2",
            "facecolor": BG,
            "edgecolor": INK,
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        clip_on=False,
    )


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise RuntimeError(f"Missing expected column. Tried: {', '.join(candidates)}")


def ordered_experiments(df: pd.DataFrame) -> list[str]:
    experiments = list(df["experiment"].dropna().unique())
    order_index = {crossover: idx for idx, crossover in enumerate(CROSSOVER_ORDER)}
    return sorted(
        experiments,
        key=lambda experiment: (
            order_index.get(infer_crossover_type(experiment), len(order_index)),
            str(experiment),
        ),
    )


def plot_population_diversity(ax, summary_df: pd.DataFrame):
    handles = []
    labels = []

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

        line = ax.plot(
            x,
            y,
            color=color,
            linestyle=CROSSOVER_LINESTYLES[crossover],
            linewidth=2.4,
            marker=CROSSOVER_MARKERS[crossover],
            markersize=4.2,
            markevery=5,
            label=CROSSOVER_LABELS[crossover],
        )[0]
        ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)
        handles.append(line)
        labels.append(CROSSOVER_LABELS[crossover])

    ax.set_title("Mean Population Diversity Over Time")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean pairwise distance")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    clean_axes(ax)
    return handles, labels


def plot_elite_lineage_trajectory(ax, summary_df: pd.DataFrame):
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
                linestyle=CROSSOVER_LINESTYLES[crossover],
                linewidth=2.4,
                marker=CROSSOVER_MARKERS[crossover],
                markersize=4.2,
            )
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)

    ax.set_title("Phenotypic Lineage Trajectory")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean parent-child distance")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 0.8)
    ax.grid(True)
    clean_axes(ax)


def plot_metric_line(ax, outer_df: pd.DataFrame, metric: str, title: str, ylabel: str):
    for experiment in ordered_experiments(outer_df):
        data = outer_df[outer_df["experiment"] == experiment].sort_values("generation")
        if data.empty:
            continue

        crossover = infer_crossover_type(experiment)
        if crossover not in CROSSOVER_COLORS:
            continue

        color = CROSSOVER_COLORS[crossover]
        center = pd.to_numeric(
            data[first_existing_column(data, [f"{metric}_mean_mean", f"{metric}_mean_median"])],
            errors="coerce",
        )
        spread = pd.to_numeric(
            data[first_existing_column(data, [f"{metric}_mean_std", f"{metric}_std_median"])],
            errors="coerce",
        ).fillna(0.0)

        ax.plot(
            data["generation"],
            center,
            color=color,
            linestyle=CROSSOVER_LINESTYLES[crossover],
            linewidth=2.4,
            marker=CROSSOVER_MARKERS[crossover],
            markersize=4.2,
            markevery=5,
        )
        if (spread > 0).any():
            ax.fill_between(
                data["generation"],
                (center - spread).clip(lower=0.0),
                center + spread,
                color=color,
                alpha=0.16,
            )

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    clean_axes(ax)


def load_inputs(analysis_dir: Path):
    paths = {
        "diversity": analysis_dir / "mean_population_diversity_over_time_summary.csv",
        "elite_lineage": analysis_dir / "elite_lineage_morphological_trajectory_summary.csv",
        "outer": analysis_dir / "gens_robots_outer.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required analysis files:\n" + "\n".join(missing))

    return (
        pd.read_csv(paths["diversity"]),
        pd.read_csv(paths["elite_lineage"]),
        pd.read_csv(paths["outer"]),
    )


def run_plot(
    *,
    analysis_dir: Path,
    output_name: str = "diversity_step_symmetry_contact_panel_chart.png",
):
    set_style()
    diversity_df, elite_lineage_df, outer_df = load_inputs(analysis_dir)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.4, 8.8))

    handles, labels = plot_population_diversity(axes[0, 0], diversity_df)
    plot_elite_lineage_trajectory(axes[1, 0], elite_lineage_df)
    plot_metric_line(axes[0, 1], outer_df, "symmetry", "Symmetry Over Generations", "Symmetry")
    plot_metric_line(
        axes[1, 1],
        outer_df,
        "environmental_contact_area",
        "Environmental Contact Area Over Generations",
        "Environmental contact area",
    )

    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncol=min(len(handles), 3),
            bbox_to_anchor=(0.5, 0.925),
        )

    fig.suptitle(
        "Population Diversity and Morphology Metrics",
        fontsize=24,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.tight_layout(rect=(0.035, 0.02, 1, 0.935), h_pad=2.2, w_pad=2.0)

    for label, ax in (
        ("A", axes[0, 0]),
        ("B", axes[1, 0]),
        ("C", axes[0, 1]),
        ("D", axes[1, 1]),
    ):
        add_panel_label(ax, label)

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
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
