#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crossover_labels import display_experiment_name


DEFAULT_METRICS = [
    ("size", "Size"),
    ("proportion", "Proportion"),
    ("coverage", "Coverage"),
    ("symmetry", "Symmetry"),
    ("bounding_box_area", "Bounding Box Area"),
    ("relative_number_of_joints", "Relative Number Of Joints"),
    ("relative_number_of_limbs", "Relative Number Of Limbs"),
    ("actuation_energy_cost", "Actuation Energy Cost"),
    ("environmental_contact_area", "Environmental Contact Area"),
    ("material_ratios", "Material Ratios"),
    ("muscle_phase_ratios", "Muscle Phase Ratios"),
]

DEFAULT_METRICS_RAW = ",".join(metric for metric, _ in DEFAULT_METRICS)

COLORS = [
    "#3F6EA8",
    "#C99227",
    "#5FA8A2",
    "#B85757",
    "#7C6F9B",
    "#7E9C45",
]

SERIES_COLORS = {
    "bone_prop": "#1E2430",
    "bone_count": "#1E2430",
    "fat_prop": "#8C8C8C",
    "fat_count": "#8C8C8C",
    "fat2_prop": "#B5A642",
    "fat2_count": "#B5A642",
    "horizontal_muscle_prop": "#C99227",
    "horizontal_muscle_count": "#C99227",
    "vertical_muscle_prop": "#3F6EA8",
    "vertical_muscle_count": "#3F6EA8",
    "phase_muscle_prop": "#5FA8A2",
    "phase_muscle_count": "#5FA8A2",
    "offphase_muscle_prop": "#B85757",
    "offphase_muscle_count": "#B85757",
    "muscle_prop": "#7C6F9B",
    "muscle_count": "#7C6F9B",
    "horizontal_phase_muscle_prop": "#D18F1F",
    "horizontal_offphase_muscle_prop": "#E7B969",
    "vertical_phase_muscle_prop": "#2F6FB3",
    "vertical_offphase_muscle_prop": "#75A9E0",
    "total_voxel_volume": "#3F6EA8",
    "bounding_box_area": "#C99227",
    "coverage": "#5FA8A2",
    "proportion": "#3F6EA8",
    "relative_number_of_joints": "#B85757",
    "relative_number_of_limbs": "#7E9C45",
    "environmental_contact_area": "#7C6F9B",
}

EXPERIMENT_LINESTYLES = ["-", "--", "-.", ":"]

COMPOSITE_METRICS = {
    "body_structure_metrics": {
        "label": "Body Structure Metrics",
        "ylabel": "Mean Value",
        "series": [
            ("bounding_box_area", "Bounding Box Area"),
            ("proportion", "Proportion"),
            ("coverage", "Coverage"),
            ("relative_number_of_joints", "Relative Number Of Joints"),
            ("relative_number_of_limbs", "Relative Number Of Limbs"),
            ("environmental_contact_area", "Environmental Contact Area"),
        ],
    },
    "material_ratios": {
        "label": "Material Ratios",
        "ylabel": "Mean Ratio",
        "series": [
            ("bone_prop", "Bone"),
            ("fat_prop", "Fat"),
            ("fat2_prop", "Fat2"),
            ("muscle_prop", "Muscle"),
        ],
    },
    "voxel_type_ratios": {
        "label": "Detailed Voxel Type Ratios",
        "ylabel": "Mean Ratio",
        "series": [
            ("bone_prop", "Bone"),
            ("fat_prop", "Fat"),
            ("fat2_prop", "Fat2"),
            ("horizontal_phase_muscle_prop", "Horizontal Phase Muscle"),
            ("horizontal_offphase_muscle_prop", "Horizontal Offphase Muscle"),
            ("vertical_phase_muscle_prop", "Vertical Phase Muscle"),
            ("vertical_offphase_muscle_prop", "Vertical Offphase Muscle"),
        ],
    },
    "muscle_orientation_ratios": {
        "label": "Muscle Orientation Ratios",
        "ylabel": "Mean Ratio",
        "series": [
            ("horizontal_muscle_prop", "Horizontal Muscle"),
            ("vertical_muscle_prop", "Vertical Muscle"),
        ],
    },
    "muscle_phase_ratios": {
        "label": "Muscle Phase Ratios",
        "ylabel": "Mean Ratio",
        "series": [
            ("phase_muscle_prop", "Phase Muscle"),
            ("offphase_muscle_prop", "Offphase Muscle"),
        ],
    },
    "material_counts": {
        "label": "Material Counts",
        "ylabel": "Mean Count",
        "series": [
            ("bone_count", "Bone"),
            ("fat_count", "Fat"),
            ("fat2_count", "Fat2"),
            ("phase_muscle_count", "Phase Muscle"),
            ("offphase_muscle_count", "Offphase Muscle"),
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot morphology metrics over generations as a multi-panel figure."
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
        "--metrics",
        default=DEFAULT_METRICS_RAW,
        type=str,
        help="Comma-separated metric ids to plot.",
    )
    parser.add_argument(
        "--output-name",
        default="morphology_metrics_progression.png",
        type=str,
    )
    return parser.parse_args()


def metric_label_map():
    labels = dict(DEFAULT_METRICS)
    labels.update(
        {
            "total_voxel_volume": "Total Voxel Volume",
            "bounding_box_area": "Bounding Box Area",
            "proportion": "Proportion",
            "coverage": "Coverage",
            "relative_number_of_joints": "Relative Number Of Joints",
            "relative_number_of_limbs": "Relative Number Of Limbs",
            "environmental_contact_area": "Environmental Contact Area",
            "bone_prop": "Bone Ratio",
            "fat_prop": "Fat Ratio",
            "fat2_prop": "Fat2 Ratio",
            "horizontal_phase_muscle_prop": "Horizontal Phase Muscle Ratio",
            "horizontal_offphase_muscle_prop": "Horizontal Offphase Muscle Ratio",
            "vertical_phase_muscle_prop": "Vertical Phase Muscle Ratio",
            "vertical_offphase_muscle_prop": "Vertical Offphase Muscle Ratio",
            "horizontal_muscle_prop": "Horizontal Muscle Ratio",
            "vertical_muscle_prop": "Vertical Muscle Ratio",
            "phase_muscle_prop": "Phase Muscle Ratio",
            "offphase_muscle_prop": "Offphase Muscle Ratio",
            "muscle_prop": "Muscle Ratio",
            "bone_count": "Bone Count",
            "fat_count": "Fat Count",
            "fat2_count": "Fat2 Count",
            "horizontal_muscle_count": "Horizontal Muscle Count",
            "vertical_muscle_count": "Vertical Muscle Count",
            "phase_muscle_count": "Phase Muscle Count",
            "offphase_muscle_count": "Offphase Muscle Count",
            "muscle_count": "Muscle Count",
        }
    )
    return labels


def parse_metric_list(raw_metrics: str):
    labels = metric_label_map()
    metrics = [metric.strip() for metric in raw_metrics.split(",") if metric.strip()]
    return [(metric, labels.get(metric, metric.replace("_", " ").title())) for metric in metrics]


def add_derived_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    add_summed_metric(
        df,
        target_metric="muscle_prop",
        source_metrics=["phase_muscle_prop", "offphase_muscle_prop"],
    )
    return df


def add_summed_metric(df: pd.DataFrame, *, target_metric: str, source_metrics):
    for suffix in ("mean", "median"):
        mean_cols = [f"{metric}_mean_{suffix}" for metric in source_metrics]
        target_mean_col = f"{target_metric}_mean_{suffix}"
        if target_mean_col not in df.columns and all(col in df.columns for col in mean_cols):
            df[target_mean_col] = df[mean_cols].sum(axis=1)

    std_sets = [
        ("mean_std", [f"{metric}_mean_std" for metric in source_metrics]),
        ("std_median", [f"{metric}_std_median" for metric in source_metrics]),
    ]
    for target_suffix, std_cols in std_sets:
        target_std_col = f"{target_metric}_{target_suffix}"
        if target_std_col not in df.columns and all(col in df.columns for col in std_cols):
            df[target_std_col] = (df[std_cols].pow(2).sum(axis=1)) ** 0.5


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def metric_columns_available(df: pd.DataFrame, metric: str) -> bool:
    center_candidates = [f"{metric}_mean_mean", f"{metric}_mean_median"]
    spread_candidates = [f"{metric}_mean_std", f"{metric}_std_median"]
    center_cols = [col for col in center_candidates if col in df.columns]
    spread_cols = [col for col in spread_candidates if col in df.columns]
    if not center_cols or not spread_cols:
        return False
    return not df[center_cols + spread_cols].isna().all().all()


def composite_series_has_signal(df: pd.DataFrame, metric: str) -> bool:
    columns = [
        col
        for col in (
            f"{metric}_mean_mean",
            f"{metric}_mean_median",
            f"{metric}_mean_std",
            f"{metric}_std_median",
        )
        if col in df.columns
    ]
    values = df[columns].replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
    if values.empty:
        return False
    return (values.abs().sum(axis=1) > 0).any()


def resolve_panels(df: pd.DataFrame, requested_metrics):
    panels = []
    for metric, label in requested_metrics:
        if metric in COMPOSITE_METRICS:
            spec = COMPOSITE_METRICS[metric]
            series = [
                (series_metric, series_label)
                for series_metric, series_label in spec["series"]
                if metric_columns_available(df, series_metric)
                and composite_series_has_signal(df, series_metric)
            ]
            if series:
                panels.append(
                    {
                        "kind": "composite",
                        "metric": metric,
                        "label": spec["label"],
                        "ylabel": spec["ylabel"],
                        "series": series,
                    }
                )
            continue

        if metric_columns_available(df, metric):
            panels.append(
                {
                    "kind": "single",
                    "metric": metric,
                    "label": label,
                    "ylabel": label,
                }
            )
    return panels


def first_existing_column(df: pd.DataFrame, candidates) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise RuntimeError(f"Missing expected plot column. Tried: {', '.join(candidates)}")


def plot_single_panel(ax, df: pd.DataFrame, experiments, metric: str, label: str, legend_handles, legend_labels):
    for idx_experiment, experiment in enumerate(experiments):
        data = df[df["experiment"] == experiment].sort_values("generation")
        if data.empty:
            continue

        experiment_label = display_experiment_name(experiment)
        color = COLORS[idx_experiment % len(COLORS)]
        center = data[first_existing_column(data, [f"{metric}_mean_mean", f"{metric}_mean_median"])]
        spread = data[first_existing_column(data, [f"{metric}_mean_std", f"{metric}_std_median"])].fillna(0.0)
        line = ax.plot(
            data["generation"],
            center,
            color=color,
            linewidth=2.2,
            label=experiment_label,
        )[0]
        if (spread > 0).any():
            ax.fill_between(
                data["generation"],
                (center - spread).clip(lower=0.0),
                center + spread,
                color=color,
                alpha=0.22,
            )
        if experiment_label not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(experiment_label)

    ax.set_title(label)
    ax.set_xlabel("Generation")
    ax.set_ylabel(label)
    ax.grid(True)
    clean_axes(ax)


def plot_composite_panel(ax, df: pd.DataFrame, experiments, panel):
    one_experiment = len(experiments) == 1

    for idx_experiment, experiment in enumerate(experiments):
        data = df[df["experiment"] == experiment].sort_values("generation")
        if data.empty:
            continue

        linestyle = EXPERIMENT_LINESTYLES[idx_experiment % len(EXPERIMENT_LINESTYLES)]
        experiment_label = display_experiment_name(experiment)
        for series_metric, series_label in panel["series"]:
            color = SERIES_COLORS.get(series_metric, COLORS[idx_experiment % len(COLORS)])
            label = series_label if one_experiment else f"{experiment_label}: {series_label}"
            center = data[first_existing_column(data, [f"{series_metric}_mean_mean", f"{series_metric}_mean_median"])]
            spread = data[first_existing_column(data, [f"{series_metric}_mean_std", f"{series_metric}_std_median"])].fillna(0.0)
            ax.plot(
                data["generation"],
                center,
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
            if (spread > 0).any():
                ax.fill_between(
                    data["generation"],
                    (center - spread).clip(lower=0.0),
                    center + spread,
                    color=color,
                    alpha=0.12 if one_experiment else 0.08,
                )

    ax.set_title(panel["label"])
    ax.set_xlabel("Generation")
    ax.set_ylabel(panel["ylabel"])
    ax.set_ylim(bottom=0)
    ax.grid(True)
    clean_axes(ax)
    ax.legend(frameon=False, loc="best", fontsize=8)


def run_plot(*, analysis_dir: Path, experiments_raw: str = "", metrics_raw: str = DEFAULT_METRICS_RAW, output_name: str = "morphology_metrics_progression.png"):
    outer_path = analysis_dir / "gens_robots_outer.csv"
    if not outer_path.exists():
        raise FileNotFoundError(f"Missing consolidated CSV: {outer_path}")

    df = add_derived_plot_columns(pd.read_csv(outer_path))
    requested_metrics = parse_metric_list(metrics_raw)

    if experiments_raw:
        experiments = [item.strip() for item in experiments_raw.split(",") if item.strip()]
    else:
        experiments = list(df["experiment"].dropna().unique())

    panels = resolve_panels(df, requested_metrics)

    if not panels:
        raise RuntimeError("None of the requested morphology metrics were found in gens_robots_outer.csv.")

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
            "savefig.facecolor": "#FCFCFA",
            "savefig.dpi": 300,
        }
    )

    n_metrics = len(panels)
    ncols = 2 if n_metrics > 1 else 1
    nrows = math.ceil(n_metrics / ncols)
    fig = plt.figure(figsize=(7.2 * ncols, 4.8 * nrows))
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols)
    axes = []
    for idx_metric in range(n_metrics):
        row = idx_metric // ncols
        col = idx_metric % ncols
        if ncols == 2 and n_metrics % 2 == 1 and idx_metric == n_metrics - 1:
            axes.append(fig.add_subplot(grid[row, :]))
        else:
            axes.append(fig.add_subplot(grid[row, col]))

    legend_handles = []
    legend_labels = []

    for idx_metric, panel in enumerate(panels):
        ax = axes[idx_metric]
        if panel["kind"] == "composite":
            plot_composite_panel(ax, df, experiments, panel)
        else:
            plot_single_panel(
                ax,
                df,
                experiments,
                panel["metric"],
                panel["label"],
                legend_handles,
                legend_labels,
            )

    fig.suptitle("Morphology Metrics Over Generations", fontsize=15, y=0.995)
    if legend_handles and len(legend_labels) > 1:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )

    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    output_path = analysis_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


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
        metrics_raw=args.metrics,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
