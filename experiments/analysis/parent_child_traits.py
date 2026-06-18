#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from algorithms.EA_classes import Robot


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


def safe_mean(series):
    values = pd.Series(series).dropna()
    return values.mean() if not values.empty else np.nan


def safe_median(series):
    values = pd.Series(series).dropna()
    return values.median() if not values.empty else np.nan


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


def load_robot_table(db_path, experiment, run):
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    columns = [
        Robot.robot_id,
        Robot.born_generation,
        Robot.valid,
        Robot.parent1_id,
        Robot.parent2_id,
        Robot.genome_size,
    ]
    columns.extend(getattr(Robot, metric) for metric in BODY_TRAITS)

    with engine.connect() as conn:
        df = pd.read_sql(select(*columns), conn)

    df["experiment"] = experiment
    df["run"] = run
    return df


def build_parent_links(robots_df):
    children = robots_df.copy()
    parents = robots_df.copy()

    child_cols = {
        "robot_id": "child_id",
        "born_generation": "child_generation",
        "valid": "child_valid",
        "genome_size": "child_genome_size",
    }
    child_cols.update({metric: f"child_{metric}" for metric in BODY_TRAITS})
    children = children.rename(columns=child_cols)

    parents_cols = {
        "robot_id": "parent_id",
        "born_generation": "parent_generation",
        "valid": "parent_valid",
        "genome_size": "parent_genome_size",
    }
    parents_cols.update({metric: f"parent_{metric}" for metric in BODY_TRAITS})
    parents = parents.rename(columns=parents_cols)

    links = []
    for slot in ("parent1", "parent2"):
        slot_df = children[
            ["experiment", "run", "child_id", "child_generation", "child_valid", "child_genome_size"]
            + [f"child_{metric}" for metric in BODY_TRAITS]
            + [f"{slot}_id"]
        ].copy()
        slot_df = slot_df.rename(columns={f"{slot}_id": "parent_id"})
        slot_df["parent_slot"] = slot
        slot_df = slot_df.dropna(subset=["parent_id"])
        links.append(slot_df)

    links_df = pd.concat(links, ignore_index=True)
    links_df["parent_id"] = links_df["parent_id"].astype(int)

    merged = links_df.merge(
        parents,
        on=["experiment", "run", "parent_id"],
        how="left",
        validate="many_to_one",
    )
    return merged


def add_trait_differences(df):
    trait_distance_cols = []

    for metric in BODY_TRAITS:
        child_col = f"child_{metric}"
        parent_col = f"parent_{metric}"

        signed_col = f"{metric}_signed_diff"
        abs_col = f"{metric}_abs_diff"
        norm_col = f"{metric}_norm_abs_diff"

        df[child_col] = pd.to_numeric(df[child_col], errors="coerce")
        df[parent_col] = pd.to_numeric(df[parent_col], errors="coerce")

        df[signed_col] = df[child_col] - df[parent_col]
        df[abs_col] = (df[child_col] - df[parent_col]).abs()

        denom = np.maximum(np.maximum(df[child_col].abs(), df[parent_col].abs()), 1e-9)
        df[norm_col] = df[abs_col] / denom
        trait_distance_cols.append(norm_col)

    trait_matrix = df[trait_distance_cols]
    trait_counts = trait_matrix.notna().sum(axis=1)
    trait_sums = trait_matrix.sum(axis=1, skipna=True)
    df["trait_profile_distance"] = np.where(trait_counts > 0, trait_sums / trait_counts, np.nan)
    df["trait_profile_similarity"] = 1.0 - df["trait_profile_distance"]
    df["generation_gap"] = df["child_generation"] - df["parent_generation"]
    return df


def summarize_links(df, label):
    rows = []
    pair_count = len(df)
    unique_children = df["child_id"].nunique()

    overall = {
        "view": label,
        "metric": "trait_profile_distance",
        "pair_count": pair_count,
        "child_count": unique_children,
        "mean_parent": np.nan,
        "mean_child": np.nan,
        "mean_signed_diff": np.nan,
        "mean_abs_diff": safe_mean(df["trait_profile_distance"]),
        "median_abs_diff": safe_median(df["trait_profile_distance"]),
        "mean_norm_abs_diff": safe_mean(df["trait_profile_distance"]),
        "median_norm_abs_diff": safe_median(df["trait_profile_distance"]),
        "parent_child_correlation": np.nan,
    }
    rows.append(overall)

    for metric in BODY_TRAITS:
        child_col = f"child_{metric}"
        parent_col = f"parent_{metric}"
        signed_col = f"{metric}_signed_diff"
        abs_col = f"{metric}_abs_diff"
        norm_col = f"{metric}_norm_abs_diff"

        valid = df[[child_col, parent_col]].dropna()
        corr = valid[child_col].corr(valid[parent_col]) if len(valid) > 1 else np.nan

        rows.append(
            {
                "view": label,
                "metric": metric,
                "pair_count": pair_count,
                "child_count": unique_children,
                "mean_parent": safe_mean(df[parent_col]),
                "mean_child": safe_mean(df[child_col]),
                "mean_signed_diff": safe_mean(df[signed_col]),
                "mean_abs_diff": safe_mean(df[abs_col]),
                "median_abs_diff": safe_median(df[abs_col]),
                "mean_norm_abs_diff": safe_mean(df[norm_col]),
                "median_norm_abs_diff": safe_median(df[norm_col]),
                "parent_child_correlation": corr,
            }
        )

    return pd.DataFrame(rows)


def pick_closest_parent(df):
    order = df.sort_values(
        by=["experiment", "run", "child_id", "trait_profile_distance", "parent_slot", "parent_id"]
    )
    closest = order.groupby(["experiment", "run", "child_id"], as_index=False).first()
    return closest


def make_text_report(summary_df, closest_df):
    lines = []

    overall = summary_df[summary_df["metric"] == "trait_profile_distance"].iloc[0]
    lines.append("Parent-child body trait report")
    lines.append(
        f"- Compared {int(overall['pair_count'])} parent-child links spanning {int(overall['child_count'])} children."
    )
    lines.append(
        "- Overall trait-profile distance is the mean normalized absolute difference "
        "across all stored body traits, where 0 means identical and 1 means maximally different."
    )
    lines.append(
        f"- Mean overall trait-profile distance: {overall['mean_abs_diff']:.4f}"
    )
    lines.append(
        f"- Median overall trait-profile distance: {overall['median_abs_diff']:.4f}"
    )

    if not closest_df.empty:
        closest_mean = closest_df["trait_profile_distance"].mean()
        closest_median = closest_df["trait_profile_distance"].median()
        lines.append(
            f"- Closest-parent mean distance: {closest_mean:.4f} "
            f"(median {closest_median:.4f})."
        )

    ranked = (
        summary_df[summary_df["metric"].isin(BODY_TRAITS)]
        .sort_values("mean_norm_abs_diff", ascending=False)
        .head(5)
    )
    lines.append("- Traits with the largest normalized parent-child differences:")
    for _, row in ranked.iterrows():
        lines.append(
            f"  {row['metric']}: mean_norm_abs_diff={row['mean_norm_abs_diff']:.4f}, "
            f"correlation={row['parent_child_correlation']:.4f}"
        )

    stable = (
        summary_df[summary_df["metric"].isin(BODY_TRAITS)]
        .sort_values("mean_norm_abs_diff", ascending=True)
        .head(5)
    )
    lines.append("- Traits with the smallest normalized parent-child differences:")
    for _, row in stable.iterrows():
        lines.append(
            f"  {row['metric']}: mean_norm_abs_diff={row['mean_norm_abs_diff']:.4f}, "
            f"correlation={row['parent_child_correlation']:.4f}"
        )

    return "\n".join(lines) + "\n"


def analyze(args):
    study_root = Path(args.out_path) / args.study_name
    if not study_root.exists():
        raise FileNotFoundError(f"Study directory not found: {study_root}")

    experiments = parse_csv_arg(args.experiments)
    if not experiments:
        experiments = discover_experiments(study_root)

    requested_runs = parse_csv_arg(args.runs, cast=int)
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
            frames.append(load_robot_table(db_path, experiment, run))

    if not frames:
        raise RuntimeError("No experiment databases were found for the requested study/experiments/runs.")

    robots_df = pd.concat(frames, ignore_index=True)
    links_df = build_parent_links(robots_df)

    if args.valid_only:
        links_df = links_df[(links_df["child_valid"] == 1) & (links_df["parent_valid"] == 1)]

    links_df = add_trait_differences(links_df)
    links_df = links_df.sort_values(["experiment", "run", "child_generation", "child_id", "parent_slot"])

    summary_df = summarize_links(links_df, label="all_parent_links")
    closest_df = pick_closest_parent(links_df)
    closest_summary_df = summarize_links(closest_df, label="closest_parent_only")

    output_dir = Path(args.output_dir) if args.output_dir else study_root / "analysis" / "parent_child_traits"
    output_dir.mkdir(parents=True, exist_ok=True)

    links_path = output_dir / "parent_child_links.csv"
    summary_path = output_dir / "parent_child_trait_summary.csv"
    closest_path = output_dir / "closest_parent_links.csv"
    closest_summary_path = output_dir / "closest_parent_trait_summary.csv"
    report_path = output_dir / "research_answer.txt"

    links_df.to_csv(links_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    closest_df.to_csv(closest_path, index=False)
    closest_summary_df.to_csv(closest_summary_path, index=False)
    report_path.write_text(make_text_report(summary_df, closest_df), encoding="utf-8")

    print(f"Saved: {links_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {closest_path}")
    print(f"Saved: {closest_summary_path}")
    print(f"Saved: {report_path}")

    overall = summary_df[summary_df["metric"] == "trait_profile_distance"].iloc[0]
    print(
        "Overall mean normalized trait-profile distance:",
        round(float(overall["mean_abs_diff"]), 4),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Quantify how different stored body traits are between parents and children."
    )
    parser.add_argument("--out-path", default="experiments/results/tmp", type=str)
    parser.add_argument("--study-name", default="defaultstudy", type=str)
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
    parser.add_argument(
        "--valid-only",
        default=1,
        type=int,
        help="If 1, only analyze links where both parent and child are valid robots.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Optional directory for CSV and text outputs.",
    )
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
