#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crossover_labels import (
    CROSSOVER_COLORS,
    CROSSOVER_ORDER,
    display_crossover_name,
    display_experiment_name,
    infer_crossover_type,
)


FITNESS_CENTER_COLUMNS = {
    "best": ["fitness_max_mean", "fitness_max_median"],
    "mean": ["fitness_mean_mean", "fitness_mean_median"],
}
FITNESS_SPREAD_COLUMNS = {
    "best": ["fitness_max_std"],
    "mean": ["fitness_mean_std", "fitness_std_median"],
}
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
CROSSOVER_HATCHES = {
    "promoter_aligned_cut_and_splice": "",
    "arithmetic_recombination": "//",
    "homologous_gene_block_recombination": "\\\\",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw a four-panel chart with best fitness, mean fitness, "
            "parent-child morphology over generations, and parent-child "
            "morphology by crossover operator."
        )
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
    parser.add_argument("--analysis-dir", default="", type=str)
    parser.add_argument(
        "--output-name",
        default="fitness_parent_child_morphology_panel_chart.png",
        type=str,
    )
    return parser.parse_args()


def set_style():
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
            "grid.alpha": 0.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "savefig.facecolor": "#FCFCFA",
            "savefig.dpi": 300,
        }
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise RuntimeError(f"Missing expected column. Tried: {', '.join(candidates)}")


def ordered_experiments(df: pd.DataFrame):
    experiments = list(df["experiment"].dropna().unique())
    order_index = {crossover: idx for idx, crossover in enumerate(CROSSOVER_ORDER)}
    return sorted(
        experiments,
        key=lambda experiment: (
            order_index.get(infer_crossover_type(experiment), len(order_index)),
            display_experiment_name(experiment),
        ),
    )


def plot_fitness_panel(ax, fitness_df: pd.DataFrame, metric: str, title: str):
    center_column = first_existing_column(fitness_df, FITNESS_CENTER_COLUMNS[metric])
    spread_column = first_existing_column(fitness_df, FITNESS_SPREAD_COLUMNS[metric])
    handles = []
    labels = []

    for experiment in ordered_experiments(fitness_df):
        data = fitness_df[fitness_df["experiment"] == experiment].sort_values("generation")
        if data.empty:
            continue

        crossover = infer_crossover_type(experiment)
        color = CROSSOVER_COLORS.get(crossover, "#4E79A7")
        label = display_experiment_name(experiment)
        center = pd.to_numeric(data[center_column], errors="coerce")
        spread = pd.to_numeric(data[spread_column], errors="coerce").fillna(0.0)

        line = ax.plot(
            data["generation"],
            center,
            color=color,
            linestyle=CROSSOVER_LINESTYLES.get(crossover, "-"),
            linewidth=2.4,
            marker=CROSSOVER_MARKERS.get(crossover, "o"),
            markersize=4.2,
            markevery=5,
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
        handles.append(line)
        labels.append(label)

    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    clean_axes(ax)
    return handles, labels


def plot_generation_distance_panel(ax, summary_df: pd.DataFrame):
    summary_df = summary_df[summary_df["parent_slot"] == "closest_parent"].copy()

    for crossover in CROSSOVER_ORDER:
        data = summary_df[summary_df["crossover_type"] == crossover].sort_values(
            "child_generation"
        )
        if data.empty:
            continue
        color = CROSSOVER_COLORS[crossover]
        center = pd.to_numeric(data["morph_distance_mean"], errors="coerce")
        lower = pd.to_numeric(data["morph_distance_q25"], errors="coerce")
        upper = pd.to_numeric(data["morph_distance_q75"], errors="coerce")

        ax.plot(
            data["child_generation"],
            center,
            color=color,
            linestyle=CROSSOVER_LINESTYLES[crossover],
            linewidth=2.4,
            marker=CROSSOVER_MARKERS[crossover],
            markersize=4.2,
            markevery=5,
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

    ax.set_title("Parent-Child Morphological Distance Over Generations")
    ax.set_xlabel("Child generation")
    ax.set_ylabel("Mean normalized distance")
    ax.set_ylim(0, 0.6)
    ax.grid(True)
    clean_axes(ax)


def plot_crossover_distance_panel(ax, links_df: pd.DataFrame):
    closest_df = links_df[links_df["parent_slot"] == "closest_parent"].copy()
    closest_df["morph_distance"] = pd.to_numeric(
        closest_df["morph_distance"],
        errors="coerce",
    )
    closest_df = closest_df[np.isfinite(closest_df["morph_distance"])]

    values = []
    positions = []
    labels = []
    for idx, crossover in enumerate(CROSSOVER_ORDER, start=1):
        series = closest_df.loc[
            closest_df["crossover_type"] == crossover,
            "morph_distance",
        ].dropna()
        if not series.empty:
            values.append(series.to_numpy())
            positions.append(idx)
        labels.append(display_crossover_name(crossover))

    if values:
        box = ax.boxplot(
            values,
            positions=positions,
            patch_artist=True,
            showfliers=False,
            widths=0.28,
        )
        for idx, patch in enumerate(box["boxes"]):
            crossover = CROSSOVER_ORDER[positions[idx] - 1]
            patch.set_facecolor(CROSSOVER_COLORS[crossover])
            patch.set_alpha(0.72)
            patch.set_edgecolor("#1E2430")
            patch.set_hatch(CROSSOVER_HATCHES[crossover])
        for median in box["medians"]:
            median.set_color("#1E2430")
            median.set_linewidth(1.8)

        rng = np.random.default_rng(7)
        for position, vals in zip(positions, values):
            sample = vals if vals.size <= 450 else rng.choice(vals, size=450, replace=False)
            jitter = rng.normal(0, 0.022, size=sample.size)
            ax.scatter(
                np.full(sample.size, position) + jitter,
                sample,
                s=7,
                color="#1E2430",
                alpha=0.14,
                linewidths=0,
            )
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Parent-Child Morphological Distance by Operator")
    ax.set_xlabel("")
    ax.set_ylabel("Normalized distance")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0.45, len(CROSSOVER_ORDER) + 0.55)
    ax.set_xticks(range(1, len(CROSSOVER_ORDER) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, axis="y")
    clean_axes(ax)


def add_panel_label(ax, label: str):
    ax.text(
        0.018,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="#1E2430",
        bbox={
            "boxstyle": "square,pad=0.2",
            "facecolor": "#FCFCFA",
            "edgecolor": "#1E2430",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        clip_on=False,
    )


def load_inputs(analysis_dir: Path):
    paths = {
        "fitness": analysis_dir / "gens_robots_outer.csv",
        "generation_summary": analysis_dir / "parent_child_morphology_summary_by_generation.csv",
        "links": analysis_dir / "parent_child_fitness_distance_links.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required analysis files:\n" + "\n".join(missing))

    return (
        pd.read_csv(paths["fitness"]),
        pd.read_csv(paths["generation_summary"]),
        pd.read_csv(paths["links"], low_memory=False),
    )


def run_plot(
    *,
    analysis_dir: Path,
    output_name: str = "fitness_parent_child_morphology_panel_chart.png",
):
    set_style()
    fitness_df, generation_summary_df, links_df = load_inputs(analysis_dir)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.4, 8.8))

    handles, labels = plot_fitness_panel(
        axes[0, 0],
        fitness_df,
        "best",
        "Best Fitness Over Generations",
    )
    plot_fitness_panel(
        axes[1, 0],
        fitness_df,
        "mean",
        "Mean Fitness Over Generations",
    )
    plot_generation_distance_panel(axes[0, 1], generation_summary_df)
    plot_crossover_distance_panel(axes[1, 1], links_df)

    if handles:
        unique = dict(zip(labels, handles))
        fig.legend(
            unique.values(),
            unique.keys(),
            frameon=False,
            loc="upper center",
            ncol=min(len(unique), 3),
            bbox_to_anchor=(0.5, 0.925),
        )

    fig.suptitle(
        "Fitness and Parent-Child Morphological Distance",
        fontsize=24,
        fontweight="bold",
        color="#1E2430",
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
    run_plot(analysis_dir=analysis_dir, output_name=args.output_name)


if __name__ == "__main__":
    main()
