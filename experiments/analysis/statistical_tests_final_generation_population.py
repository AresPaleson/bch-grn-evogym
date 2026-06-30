#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .crossover_labels import (
        display_crossover_name,
        infer_crossover_type,
    )
except ImportError:
    from crossover_labels import (
        display_crossover_name,
        infer_crossover_type,
    )


DEFAULT_EXCLUDE_COLUMNS = {
    "generation",
    "generation_zero_based",
    "robot_id",
    "born_generation",
    "run",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether final-generation population-average metrics differ "
            "between crossover setups. The independent observation is one "
            "run-level final-generation population mean per setup."
        )
    )
    parser.add_argument("--analysis-dir", required=True, type=str)
    parser.add_argument("--input-name", default="gens_robots.csv", type=str)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument(
        "--experiments",
        default="",
        type=str,
        help="Optional comma-separated experiment names to include.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        type=str,
        help=(
            "Optional comma-separated metric columns from gens_robots.csv. "
            "Empty means all numeric non-identifier metric columns."
        ),
    )
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def finite_numeric(values: Iterable) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def decision_from_p(p_value, alpha: float) -> str:
    if p_value is None or pd.isna(p_value):
        return "not_tested"
    return "significant" if float(p_value) < alpha else "not_significant"


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [
        (idx, float(p_value))
        for idx, p_value in enumerate(p_values)
        if p_value is not None and not pd.isna(p_value)
    ]
    adjusted = [np.nan] * len(p_values)
    if not indexed:
        return adjusted

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(indexed):
        adjusted_p = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, adjusted_p)
        adjusted[idx] = running_max
    return adjusted


def mean_ci95(values: pd.Series):
    values = finite_numeric(values)
    if len(values) < 2:
        return np.nan, np.nan
    margin = stats.t.ppf(0.975, df=len(values) - 1) * stats.sem(values)
    return float(values.mean() - margin), float(values.mean() + margin)


def hedges_g(a: pd.Series, b: pd.Series) -> float:
    a = finite_numeric(a)
    b = finite_numeric(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = (
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_var <= 0:
        return 0.0
    cohen_d = (a.mean() - b.mean()) / np.sqrt(pooled_var)
    correction = 1.0 - (3.0 / (4.0 * (len(a) + len(b)) - 9.0))
    return float(cohen_d * correction)


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    a_values = finite_numeric(a).to_numpy(dtype=float)
    b_values = finite_numeric(b).to_numpy(dtype=float)
    if len(a_values) == 0 or len(b_values) == 0:
        return np.nan
    greater = 0
    lower = 0
    for value_a in a_values:
        greater += int(np.sum(value_a > b_values))
        lower += int(np.sum(value_a < b_values))
    return float((greater - lower) / (len(a_values) * len(b_values)))


def eta_squared(groups: list[pd.Series]) -> float:
    all_values = np.concatenate([group.to_numpy(dtype=float) for group in groups])
    if len(all_values) == 0:
        return np.nan
    grand_mean = np.mean(all_values)
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    ss_total = sum((value - grand_mean) ** 2 for value in all_values)
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def epsilon_squared_kruskal(h_stat: float, n_total: int, group_count: int) -> float:
    denominator = n_total - group_count
    if denominator <= 0:
        return np.nan
    return float(max(0.0, (h_stat - group_count + 1.0) / denominator))


def infer_metric_columns(df: pd.DataFrame, metrics_raw: str) -> list[str]:
    requested = parse_csv_arg(metrics_raw)
    if requested:
        return [
            metric
            for metric in requested
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])
        ]

    metric_cols = []
    for column in df.columns:
        if column in DEFAULT_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            values = finite_numeric(df[column])
            if not values.empty and values.nunique() > 1:
                metric_cols.append(column)
    return metric_cols


def load_final_generation_run_means(
    analysis_dir: Path,
    input_name: str,
    metrics_raw: str,
    experiments_raw: str,
) -> tuple[pd.DataFrame, list[str]]:
    input_path = analysis_dir / input_name
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    if "experiment" not in df.columns or "run" not in df.columns or "generation" not in df.columns:
        raise ValueError("Input CSV must contain experiment, run, and generation columns.")

    experiments = parse_csv_arg(experiments_raw)
    if experiments:
        df = df[df["experiment"].isin(experiments)].copy()
    if df.empty:
        raise ValueError("No rows remain after applying the requested experiment filter.")

    metric_cols = infer_metric_columns(df, metrics_raw)
    if not metric_cols:
        raise ValueError("No numeric metric columns were available for testing.")

    for column in ["run", "generation", *metric_cols]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["experiment", "run", "generation"])
    final_generation = df.groupby(["experiment", "run"])["generation"].transform("max")
    final_df = df[df["generation"] == final_generation].copy()

    run_means = (
        final_df.groupby(["experiment", "run", "generation"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .rename(columns={metric: f"{metric}_population_mean" for metric in metric_cols})
    )
    run_means["run"] = run_means["run"].astype(int)
    run_means["final_generation"] = run_means["generation"].astype(int)
    run_means = run_means.drop(columns=["generation"])
    run_means["crossover_type"] = run_means["experiment"].map(infer_crossover_type)
    run_means["crossover_label"] = run_means["crossover_type"].map(display_crossover_name)
    return run_means, metric_cols


def descriptive_rows(run_means: pd.DataFrame, metrics: list[str]):
    rows = []
    for metric in metrics:
        value_col = f"{metric}_population_mean"
        for crossover_type, group in sorted(
            run_means.groupby("crossover_type"),
            key=lambda item: str(item[0]),
        ):
            values = finite_numeric(group[value_col])
            if values.empty:
                continue
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            ci_low, ci_high = mean_ci95(values)
            rows.append(
                {
                    "metric": metric,
                    "response_column": value_col,
                    "crossover_type": crossover_type,
                    "crossover_label": display_crossover_name(crossover_type),
                    "n_runs": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "median": values.median(),
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                    "min": values.min(),
                    "max": values.max(),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return rows


def assumption_rows(run_means: pd.DataFrame, metrics: list[str]):
    rows = []
    for metric in metrics:
        value_col = f"{metric}_population_mean"
        groups = []
        for crossover_type, group in sorted(
            run_means.groupby("crossover_type"),
            key=lambda item: str(item[0]),
        ):
            values = finite_numeric(group[value_col])
            groups.append(values)
            shapiro_stat = np.nan
            shapiro_p = np.nan
            if 3 <= len(values) <= 5000 and values.nunique() > 1:
                shapiro_stat, shapiro_p = stats.shapiro(values)
            rows.append(
                {
                    "metric": metric,
                    "response_column": value_col,
                    "test": "shapiro_wilk",
                    "crossover_type": crossover_type,
                    "n_runs": len(values),
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                }
            )

        valid_groups = [group for group in groups if len(group) >= 2 and group.nunique() > 1]
        levene_stat = np.nan
        levene_p = np.nan
        if len(valid_groups) >= 2:
            levene_stat, levene_p = stats.levene(*valid_groups, center="median")
        rows.append(
            {
                "metric": metric,
                "response_column": value_col,
                "test": "levene_median",
                "crossover_type": "all",
                "n_runs": int(sum(len(group) for group in groups)),
                "statistic": levene_stat,
                "p_value": levene_p,
            }
        )
    return rows


def assumptions_for_metric(assumptions_df: pd.DataFrame, metric: str, alpha: float):
    metric_assumptions = assumptions_df[assumptions_df["metric"] == metric]
    shapiro = metric_assumptions[metric_assumptions["test"] == "shapiro_wilk"]
    shapiro = shapiro[shapiro["p_value"].notna()]
    levene = metric_assumptions[metric_assumptions["test"] == "levene_median"]
    levene_p = levene["p_value"].iloc[0] if not levene.empty else np.nan
    normal_enough = not shapiro.empty and bool((shapiro["p_value"] >= alpha).all())
    equal_variance = pd.notna(levene_p) and float(levene_p) >= alpha
    return normal_enough, equal_variance, levene_p


def omnibus_rows(run_means: pd.DataFrame, metrics: list[str], assumptions_df: pd.DataFrame, alpha: float):
    rows = []
    for metric in metrics:
        value_col = f"{metric}_population_mean"
        groups_by_name = {
            name: finite_numeric(group[value_col])
            for name, group in sorted(
                run_means.groupby("crossover_type"),
                key=lambda item: str(item[0]),
            )
        }
        groups_by_name = {name: values for name, values in groups_by_name.items() if len(values) >= 2}
        groups = list(groups_by_name.values())
        if len(groups) < 2:
            rows.append(
                {
                    "metric": metric,
                    "response_column": value_col,
                    "selected_test": "not_tested",
                    "n_total": int(sum(len(values) for values in groups_by_name.values())),
                    "group_count": len(groups_by_name),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size_name": "",
                    "effect_size": np.nan,
                    "assumptions_supported_parametric": False,
                    "levene_p_value": np.nan,
                    "decision": "not_tested",
                    "note": "At least two setup groups with two or more independent runs are required.",
                }
            )
            continue

        normal_enough, equal_variance, levene_p = assumptions_for_metric(assumptions_df, metric, alpha)
        if len(groups) == 2:
            if normal_enough:
                statistic, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                selected_test = "welch_t_test"
                effect_name = "hedges_g"
                effect_size = hedges_g(groups[0], groups[1])
                parametric_ok = True
            else:
                statistic, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                selected_test = "mann_whitney_u"
                effect_name = "cliffs_delta"
                effect_size = cliffs_delta(groups[0], groups[1])
                parametric_ok = False
        elif normal_enough and equal_variance:
            statistic, p_value = stats.f_oneway(*groups)
            selected_test = "one_way_anova"
            effect_name = "eta_squared"
            effect_size = eta_squared(groups)
            parametric_ok = True
        else:
            statistic, p_value = stats.kruskal(*groups)
            selected_test = "kruskal_wallis"
            effect_name = "epsilon_squared"
            effect_size = epsilon_squared_kruskal(
                float(statistic),
                int(sum(len(group) for group in groups)),
                len(groups),
            )
            parametric_ok = False

        rows.append(
            {
                "metric": metric,
                "response_column": value_col,
                "selected_test": selected_test,
                "n_total": int(sum(len(group) for group in groups)),
                "group_count": len(groups),
                "statistic": statistic,
                "p_value": p_value,
                "effect_size_name": effect_name,
                "effect_size": effect_size,
                "assumptions_supported_parametric": parametric_ok,
                "levene_p_value": levene_p,
                "decision": decision_from_p(p_value, alpha),
                "note": "Run-level final-generation population means are the independent observations.",
            }
        )
    return rows


def pairwise_rows(run_means: pd.DataFrame, metrics: list[str], assumptions_df: pd.DataFrame, alpha: float):
    rows = []
    for metric in metrics:
        value_col = f"{metric}_population_mean"
        groups_by_name = {
            name: finite_numeric(group[value_col])
            for name, group in sorted(
                run_means.groupby("crossover_type"),
                key=lambda item: str(item[0]),
            )
        }
        groups_by_name = {name: values for name, values in groups_by_name.items() if len(values) >= 2}
        if len(groups_by_name) < 2:
            continue

        normal_enough, _, _ = assumptions_for_metric(assumptions_df, metric, alpha)
        metric_rows = []
        raw_p_values = []
        for group_a, group_b in combinations(groups_by_name, 2):
            values_a = groups_by_name[group_a]
            values_b = groups_by_name[group_b]
            if normal_enough:
                pair_stat, pair_p = stats.ttest_ind(values_a, values_b, equal_var=False)
                pair_test = "welch_t_test"
                effect_name = "hedges_g"
                effect_size = hedges_g(values_a, values_b)
            else:
                pair_stat, pair_p = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
                pair_test = "mann_whitney_u"
                effect_name = "cliffs_delta"
                effect_size = cliffs_delta(values_a, values_b)

            raw_p_values.append(pair_p)
            metric_rows.append(
                {
                    "metric": metric,
                    "response_column": value_col,
                    "test": pair_test,
                    "group_a": group_a,
                    "group_b": group_b,
                    "group_a_label": display_crossover_name(group_a),
                    "group_b_label": display_crossover_name(group_b),
                    "n_a": len(values_a),
                    "n_b": len(values_b),
                    "mean_a": values_a.mean(),
                    "mean_b": values_b.mean(),
                    "median_a": values_a.median(),
                    "median_b": values_b.median(),
                    "mean_difference_a_minus_b": values_a.mean() - values_b.mean(),
                    "statistic": pair_stat,
                    "p_value_raw": pair_p,
                    "p_value_holm": np.nan,
                    "effect_size_name": effect_name,
                    "effect_size": effect_size,
                    "decision_holm": "not_tested",
                }
            )

        for row, adjusted_p in zip(metric_rows, holm_adjust(raw_p_values)):
            row["p_value_holm"] = adjusted_p
            row["decision_holm"] = decision_from_p(adjusted_p, alpha)
        rows.extend(metric_rows)
    return rows


def run_final_generation_population_tests(
    *,
    analysis_dir: Path,
    input_name: str = "gens_robots.csv",
    alpha: float = 0.05,
    metrics_raw: str = "",
    experiments_raw: str = "",
) -> dict[str, Path]:
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_means, metrics = load_final_generation_run_means(
        analysis_dir,
        input_name,
        metrics_raw,
        experiments_raw,
    )

    descriptive_df = pd.DataFrame(descriptive_rows(run_means, metrics))
    assumptions_df = pd.DataFrame(assumption_rows(run_means, metrics))
    omnibus_df = pd.DataFrame(omnibus_rows(run_means, metrics, assumptions_df, alpha))
    pairwise_df = pd.DataFrame(pairwise_rows(run_means, metrics, assumptions_df, alpha))

    output_paths = {
        "run_means": analysis_dir / "final_generation_population_run_means.csv",
        "descriptive_stats": analysis_dir / "final_generation_population_descriptive_stats.csv",
        "assumption_tests": analysis_dir / "final_generation_population_assumption_tests.csv",
        "omnibus_tests": analysis_dir / "final_generation_population_omnibus_tests.csv",
        "pairwise_tests": analysis_dir / "final_generation_population_pairwise_tests.csv",
    }
    legacy_report_path = analysis_dir / "final_generation_population_statistical_report.txt"
    run_means.to_csv(output_paths["run_means"], index=False)
    descriptive_df.to_csv(output_paths["descriptive_stats"], index=False)
    assumptions_df.to_csv(output_paths["assumption_tests"], index=False)
    omnibus_df.to_csv(output_paths["omnibus_tests"], index=False)
    pairwise_df.to_csv(output_paths["pairwise_tests"], index=False)

    if legacy_report_path.exists():
        legacy_report_path.unlink()

    for path in output_paths.values():
        print(f"Saved final-generation statistical output to: {path}")
    return output_paths


def main():
    args = parse_args()
    run_final_generation_population_tests(
        analysis_dir=Path(args.analysis_dir),
        input_name=args.input_name,
        alpha=args.alpha,
        metrics_raw=args.metrics,
        experiments_raw=args.experiments,
    )


if __name__ == "__main__":
    main()
