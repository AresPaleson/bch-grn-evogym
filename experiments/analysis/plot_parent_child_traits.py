#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


BODY_TRAITS = [
    "num_voxels",
    "bone_count",
    "bone_prop",
    "fat_count",
    "fat_prop",
    "fat2_count",
    "fat2_prop",
    "phase_muscle_count",
    "phase_muscle_prop",
    "offphase_muscle_count",
    "offphase_muscle_prop",
]

TRAIT_LABELS = {
    "num_voxels": "Num voxels",
    "bone_count": "Bone count",
    "bone_prop": "Bone proportion",
    "fat_count": "Fat count",
    "fat_prop": "Fat proportion",
    "fat2_count": "Fat2 count",
    "fat2_prop": "Fat2 proportion",
    "phase_muscle_count": "Phase muscle count",
    "phase_muscle_prop": "Phase muscle proportion",
    "offphase_muscle_count": "Offphase muscle count",
    "offphase_muscle_prop": "Offphase muscle proportion",
}

COLORS = {
    "ink": "#1E2430",
    "blue": "#3F6EA8",
    "teal": "#5FA8A2",
    "gold": "#C99227",
    "red": "#B85757",
    "gray": "#9AA3AD",
    "bg": "#FCFCFA",
    "grid": "#D9DDE3",
}


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
            "grid.linestyle": "-",
            "grid.alpha": 0.55,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "savefig.facecolor": COLORS["bg"],
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def finish_figure(fig, output_path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def annotate_barh(ax, values, y_positions, pad=0.012):
    xmax = ax.get_xlim()[1]
    xmin = ax.get_xlim()[0]
    span = xmax - xmin
    for y, value in zip(y_positions, values):
        if pd.isna(value):
            continue
        if value >= 0:
            x = min(value + span * pad, xmax - span * 0.02)
            ha = "left"
        else:
            x = max(value - span * pad, xmin + span * 0.02)
            ha = "right"
        ax.text(x, y, f"{value:.3f}", va="center", ha=ha, color=COLORS["ink"], fontsize=8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create visualizations for parent-child body trait differences."
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
    parser.add_argument("--analysis-dir", default="", type=str)
    parser.add_argument("--top-k", default=6, type=int)
    return parser.parse_args()


def load_data(analysis_dir):
    all_links = pd.read_csv(analysis_dir / "parent_child_links.csv")
    all_summary = pd.read_csv(analysis_dir / "parent_child_trait_summary.csv")
    closest_links = pd.read_csv(analysis_dir / "closest_parent_links.csv")
    closest_summary = pd.read_csv(analysis_dir / "closest_parent_trait_summary.csv")
    return all_links, all_summary, closest_links, closest_summary


def plot_distance_distribution(closest_links, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    values = closest_links["trait_profile_distance"].dropna()
    ax.hist(values, bins=20, color=COLORS["blue"], edgecolor="white", alpha=0.9)
    ax.axvline(values.mean(), color=COLORS["red"], linewidth=2, label=f"Mean = {values.mean():.3f}")
    ax.axvline(values.median(), color=COLORS["gold"], linewidth=2, label=f"Median = {values.median():.3f}")
    ax.set_title("Closest Parent-Child Trait Distance")
    ax.set_xlabel("Trait-profile distance")
    ax.set_ylabel("Number of children")
    ax.grid(axis="y")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    clean_axes(ax)
    finish_figure(fig, output_dir / "distance_distribution.png")


def plot_trait_bar_summary(closest_summary, output_dir):
    df = closest_summary[closest_summary["metric"].isin(BODY_TRAITS)].copy()
    df = df[df["mean_norm_abs_diff"].notna()].sort_values("mean_norm_abs_diff", ascending=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    labels = [TRAIT_LABELS.get(metric, metric) for metric in df["metric"]]
    colors = [COLORS["teal"] if value < df["mean_norm_abs_diff"].median() else COLORS["blue"] for value in df["mean_norm_abs_diff"]]
    y = np.arange(len(df))
    ax.barh(y, df["mean_norm_abs_diff"], color=colors, height=0.72)
    ax.set_yticks(y, labels=labels)
    ax.set_title("Average Normalized Parent-Child Difference by Trait")
    ax.set_xlabel("Mean normalized absolute difference")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.set_xlim(0, max(df["mean_norm_abs_diff"].max() * 1.22, 0.6))
    annotate_barh(ax, df["mean_norm_abs_diff"].tolist(), y)
    clean_axes(ax)
    finish_figure(fig, output_dir / "trait_difference_bar.png")


def plot_trait_correlation_bars(closest_summary, output_dir):
    df = closest_summary[closest_summary["metric"].isin(BODY_TRAITS)].copy()
    df = df[df["parent_child_correlation"].notna()].sort_values("parent_child_correlation", ascending=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    labels = [TRAIT_LABELS.get(metric, metric) for metric in df["metric"]]
    colors = [COLORS["gold"] if value < 0.35 else COLORS["blue"] for value in df["parent_child_correlation"]]
    y = np.arange(len(df))
    ax.barh(y, df["parent_child_correlation"], color=colors, height=0.72)
    ax.set_yticks(y, labels=labels)
    ax.set_title("Parent-Child Trait Correlation")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("")
    ax.set_xlim(-0.05, 1.0)
    ax.grid(axis="x")
    annotate_barh(ax, df["parent_child_correlation"].tolist(), y)
    clean_axes(ax)
    finish_figure(fig, output_dir / "trait_correlation_bar.png")


def plot_generation_distance(closest_links, output_dir):
    grouped = (
        closest_links.groupby("child_generation", as_index=False)["trait_profile_distance"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_distance", "median": "median_distance"})
    )

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(grouped["child_generation"], grouped["mean_distance"], color=COLORS["blue"], linewidth=2.5, label="Mean")
    ax.plot(grouped["child_generation"], grouped["median_distance"], color=COLORS["gold"], linewidth=2.5, label="Median")
    ax.set_title("Closest Parent-Child Distance Across Child Generations")
    ax.set_xlabel("Child generation")
    ax.set_ylabel("Trait-profile distance")
    ax.grid(True)
    ax.legend(frameon=False)

    ax2 = ax.twinx()
    ax2.bar(grouped["child_generation"], grouped["count"], color=COLORS["gray"], alpha=0.18, width=0.8)
    ax2.set_ylabel("Children compared")
    ax2.tick_params(colors=COLORS["ink"])
    clean_axes(ax)
    ax2.spines["top"].set_visible(False)
    finish_figure(fig, output_dir / "distance_by_generation.png")


def plot_top_trait_scatter(closest_links, closest_summary, output_dir, top_k):
    summary = closest_summary[closest_summary["metric"].isin(BODY_TRAITS)].copy()
    summary = summary[summary["mean_norm_abs_diff"].notna()]
    top_traits = summary.sort_values("mean_norm_abs_diff", ascending=False).head(top_k)["metric"].tolist()

    if not top_traits:
        return

    ncols = 2
    nrows = int(np.ceil(len(top_traits) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.2, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, top_traits):
        xcol = f"parent_{metric}"
        ycol = f"child_{metric}"
        plot_df = closest_links[[xcol, ycol]].dropna().copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        ax.scatter(plot_df[xcol], plot_df[ycol], s=28, alpha=0.7, color=COLORS["blue"], edgecolors="none")
        low = min(plot_df[xcol].min(), plot_df[ycol].min())
        high = max(plot_df[xcol].max(), plot_df[ycol].max())
        ax.plot([low, high], [low, high], color=COLORS["red"], linewidth=1.5, linestyle="--")
        ax.set_title(TRAIT_LABELS.get(metric, metric))
        ax.set_xlabel("Parent value")
        ax.set_ylabel("Child value")
        ax.grid(True)
        clean_axes(ax)

    for ax in axes[len(top_traits):]:
        ax.set_visible(False)

    fig.suptitle("Traits With Largest Parent-Child Differences", y=1.01, fontsize=15)
    finish_figure(fig, output_dir / "top_trait_scatter_grid.png", rect=(0, 0, 1, 0.98))


def plot_signed_shift(closest_summary, output_dir):
    df = closest_summary[closest_summary["metric"].isin(BODY_TRAITS)].copy()
    df = df[df["mean_signed_diff"].notna()].sort_values("mean_signed_diff")
    labels = [TRAIT_LABELS.get(metric, metric) for metric in df["metric"]]

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    colors = [COLORS["red"] if value > 0 else COLORS["teal"] for value in df["mean_signed_diff"]]
    y = np.arange(len(df))
    ax.barh(y, df["mean_signed_diff"], color=colors, height=0.72)
    ax.set_yticks(y, labels=labels)
    ax.axvline(0, color=COLORS["ink"], linewidth=1.2)
    ax.set_title("Average Direction of Change: Child Minus Closest Parent")
    ax.set_xlabel("Mean signed difference")
    ax.set_ylabel("")
    ax.grid(axis="x")
    span = max(abs(df["mean_signed_diff"].min()), abs(df["mean_signed_diff"].max()), 0.1)
    ax.set_xlim(-span * 1.3, span * 1.3)
    annotate_barh(ax, df["mean_signed_diff"].tolist(), y)
    clean_axes(ax)
    finish_figure(fig, output_dir / "signed_shift_bar.png")


def plot_overview_dashboard(all_links, closest_links, closest_summary, output_dir):
    trait_df = closest_summary[closest_summary["metric"].isin(BODY_TRAITS)].copy()
    trait_df = trait_df[trait_df["mean_norm_abs_diff"].notna()].sort_values("mean_norm_abs_diff", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    fig.suptitle("Parent-Child Body Trait Similarity Dashboard", fontsize=14, y=0.99)

    values = closest_links["trait_profile_distance"].dropna()
    axes[0, 0].hist(values, bins=20, color=COLORS["blue"], edgecolor="white")
    axes[0, 0].axvline(values.mean(), color=COLORS["red"], linewidth=2)
    axes[0, 0].set_title("Closest-parent distance")
    axes[0, 0].set_xlabel("Distance")
    axes[0, 0].set_ylabel("Children")

    top_traits = trait_df.head(6).sort_values("mean_norm_abs_diff", ascending=True)
    axes[0, 1].barh(
        [TRAIT_LABELS.get(metric, metric) for metric in top_traits["metric"]],
        top_traits["mean_norm_abs_diff"],
        color=COLORS["gold"],
    )
    axes[0, 1].set_title("Largest trait differences")
    axes[0, 1].set_xlabel("Mean normalized difference")

    grouped = closest_links.groupby("child_generation", as_index=False)["trait_profile_distance"].mean()
    axes[1, 0].plot(grouped["child_generation"], grouped["trait_profile_distance"], color=COLORS["teal"], linewidth=2.5)
    axes[1, 0].set_title("Distance by generation")
    axes[1, 0].set_xlabel("Child generation")
    axes[1, 0].set_ylabel("Mean distance")

    slot_means = all_links.groupby("parent_slot", as_index=False)["trait_profile_distance"].mean()
    axes[1, 1].bar(slot_means["parent_slot"], slot_means["trait_profile_distance"], color=[COLORS["blue"], COLORS["teal"]])
    axes[1, 1].set_title("Average distance by parent slot")
    axes[1, 1].set_xlabel("Recorded parent")
    axes[1, 1].set_ylabel("Mean distance")

    for ax in axes.ravel():
        ax.grid(True, axis="y", alpha=0.4)
        clean_axes(ax)

    finish_figure(fig, output_dir / "dashboard_overview.png", rect=(0, 0, 1, 0.97))


def main():
    args = parse_args()
    style()

    analysis_dir = (
        Path(args.analysis_dir)
        if args.analysis_dir
        else Path(args.out_path) / args.study_name / "analysis" / "parent_child_traits"
    )
    output_dir = analysis_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_links, _, closest_links, closest_summary = load_data(analysis_dir)

    plot_distance_distribution(closest_links, output_dir)
    plot_trait_bar_summary(closest_summary, output_dir)
    plot_trait_correlation_bars(closest_summary, output_dir)
    plot_generation_distance(closest_links, output_dir)
    plot_top_trait_scatter(closest_links, closest_summary, output_dir, top_k=args.top_k)
    plot_signed_shift(closest_summary, output_dir)
    plot_overview_dashboard(all_links, closest_links, closest_summary, output_dir)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
