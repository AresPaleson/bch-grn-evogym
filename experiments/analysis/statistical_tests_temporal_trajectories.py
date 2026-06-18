#!/usr/bin/env python3
import argparse
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .crossover_labels import display_crossover_name
except ImportError:
    from crossover_labels import display_crossover_name


DIVERSITY_TARGET = "mean_population_diversity"
LINEAGE_TARGET = "elite_lineage_trajectory"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run statistical tests for temporal morphology trajectory plots: "
            "mean population diversity over time and elite phenotypic lineage "
            "trajectory."
        )
    )
    parser.add_argument("--analysis-dir", required=True, type=str)
    parser.add_argument(
        "--target",
        default="both",
        choices=("both", DIVERSITY_TARGET, LINEAGE_TARGET),
        help="Which temporal trajectory dataset to test.",
    )
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument(
        "--diversity-run-name",
        default="mean_population_diversity_by_run_generation.csv",
        type=str,
    )
    parser.add_argument(
        "--lineage-name",
        default="elite_lineage_parent_child_distances.csv",
        type=str,
    )
    parser.add_argument(
        "--lineage-parent-choice",
        default="elite_parent",
        type=str,
        help="Lineage parent_choice to include, or 'all' for no filter.",
    )
    return parser.parse_args()


def finite_numeric(values: Iterable) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def decision_from_p(p_value, alpha: float) -> str:
    if p_value is None or pd.isna(p_value):
        return "not_tested"
    return "significant" if float(p_value) < alpha else "not_significant"


def format_p(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


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


def safe_levene(groups: list[pd.Series]):
    valid_groups = [finite_numeric(group) for group in groups]
    valid_groups = [group for group in valid_groups if len(group) >= 2 and group.nunique() > 1]
    if len(valid_groups) < 2:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        statistic, p_value = stats.levene(*valid_groups, center="median")
    return statistic, p_value


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


def paired_cohens_dz(first: pd.Series, last: pd.Series) -> float:
    first = finite_numeric(first)
    last = finite_numeric(last)
    if len(first) != len(last) or len(first) < 2:
        return np.nan
    diff = last.reset_index(drop=True) - first.reset_index(drop=True)
    diff_std = diff.std(ddof=1)
    if diff_std <= 0:
        return 0.0
    return float(diff.mean() / diff_std)


def wilcoxon_rank_biserial(first: pd.Series, last: pd.Series) -> float:
    first = finite_numeric(first)
    last = finite_numeric(last)
    if len(first) != len(last) or len(first) == 0:
        return np.nan
    diff = last.reset_index(drop=True) - first.reset_index(drop=True)
    diff = diff[diff != 0]
    if diff.empty:
        return 0.0
    ranks = stats.rankdata(diff.abs())
    positive_rank_sum = float(ranks[diff.to_numpy(dtype=float) > 0].sum())
    negative_rank_sum = float(ranks[diff.to_numpy(dtype=float) < 0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum
    return (
        float((positive_rank_sum - negative_rank_sum) / total_rank_sum)
        if total_rank_sum > 0
        else np.nan
    )


def one_sample_cohens_d(values: pd.Series, expected: float = 0.0) -> float:
    values = finite_numeric(values)
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if std <= 0:
        return 0.0
    return float((values.mean() - expected) / std)


def one_sample_rank_biserial(values: pd.Series, expected: float = 0.0) -> float:
    values = finite_numeric(values)
    diff = values - expected
    diff = diff[diff != 0]
    if diff.empty:
        return 0.0
    ranks = stats.rankdata(diff.abs())
    positive_rank_sum = float(ranks[diff.to_numpy(dtype=float) > 0].sum())
    negative_rank_sum = float(ranks[diff.to_numpy(dtype=float) < 0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum
    return (
        float((positive_rank_sum - negative_rank_sum) / total_rank_sum)
        if total_rank_sum > 0
        else np.nan
    )


def selected_between_group_test(
    groups_by_name: dict[str, pd.Series],
    *,
    alpha: float,
    note: str,
) -> dict:
    groups_by_name = {name: finite_numeric(values) for name, values in groups_by_name.items()}
    groups_by_name = {name: values for name, values in groups_by_name.items() if len(values) >= 2}
    groups = list(groups_by_name.values())
    if len(groups) < 2:
        return {
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
            "note": "At least two crossover groups with two or more runs are required.",
        }

    shapiro_ps = []
    for values in groups:
        if 3 <= len(values) <= 5000 and values.nunique() > 1:
            _, shapiro_p = stats.shapiro(values)
            shapiro_ps.append(shapiro_p)
    normal_enough = bool(shapiro_ps) and all(float(p_value) >= alpha for p_value in shapiro_ps)

    _, levene_p = safe_levene(groups)
    equal_variance = pd.notna(levene_p) and float(levene_p) >= alpha

    all_values = pd.concat([group.reset_index(drop=True) for group in groups], ignore_index=True)
    if all_values.nunique() <= 1:
        return {
            "selected_test": "not_tested_no_between_group_variation",
            "n_total": int(sum(len(group) for group in groups)),
            "group_count": len(groups),
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_name": "none",
            "effect_size": 0.0,
            "assumptions_supported_parametric": False,
            "levene_p_value": levene_p,
            "decision": "not_significant",
            "note": f"{note} All tested values were identical.",
        }

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

    return {
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
        "note": note,
    }


def pairwise_between_group_rows(
    groups_by_name: dict[str, pd.Series],
    *,
    alpha: float,
) -> list[dict]:
    groups_by_name = {name: finite_numeric(values) for name, values in groups_by_name.items()}
    groups_by_name = {name: values for name, values in groups_by_name.items() if len(values) >= 2}
    if len(groups_by_name) < 2:
        return []

    shapiro_ps = []
    for values in groups_by_name.values():
        if 3 <= len(values) <= 5000 and values.nunique() > 1:
            _, shapiro_p = stats.shapiro(values)
            shapiro_ps.append(shapiro_p)
    normal_enough = bool(shapiro_ps) and all(float(p_value) >= alpha for p_value in shapiro_ps)

    rows = []
    raw_p_values = []
    for group_a, group_b in combinations(groups_by_name, 2):
        values_a = groups_by_name[group_a]
        values_b = groups_by_name[group_b]
        if normal_enough:
            statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
            selected_test = "welch_t_test"
            effect_name = "hedges_g"
            effect_size = hedges_g(values_a, values_b)
        else:
            statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            selected_test = "mann_whitney_u"
            effect_name = "cliffs_delta"
            effect_size = cliffs_delta(values_a, values_b)

        raw_p_values.append(p_value)
        rows.append(
            {
                "test": selected_test,
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
                "statistic": statistic,
                "p_value_raw": p_value,
                "p_value_holm": np.nan,
                "effect_size_name": effect_name,
                "effect_size": effect_size,
                "decision_holm": "not_tested",
            }
        )

    for row, adjusted_p in zip(rows, holm_adjust(raw_p_values)):
        row["p_value_holm"] = adjusted_p
        row["decision_holm"] = decision_from_p(adjusted_p, alpha)
    return rows


def load_diversity_run_generation(analysis_dir: Path, input_name: str) -> pd.DataFrame:
    input_path = analysis_dir / input_name
    if not input_path.exists():
        raise FileNotFoundError(f"Mean population diversity CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    required = [
        "experiment",
        "run",
        "crossover_type",
        "generation_zero_based",
        "mean_pairwise_morphological_distance",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["trajectory_value"] = pd.to_numeric(
        df["mean_pairwise_morphological_distance"],
        errors="coerce",
    )
    df["trajectory_metric"] = "mean_pairwise_morphological_distance"
    return clean_run_generation_frame(df)


def load_lineage_run_generation(
    analysis_dir: Path,
    input_name: str,
    parent_choice: str,
) -> pd.DataFrame:
    input_path = analysis_dir / input_name
    if not input_path.exists():
        raise FileNotFoundError(f"Elite lineage CSV not found: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    required = [
        "experiment",
        "run",
        "crossover_type",
        "generation_zero_based",
        "distance",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {', '.join(missing)}")
    if parent_choice != "all" and "parent_choice" in df.columns:
        df = df[df["parent_choice"] == parent_choice].copy()
    if df.empty:
        raise ValueError("No elite lineage rows remain after applying the parent_choice filter.")

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    group_cols = ["experiment", "run", "crossover_type", "generation_zero_based"]
    if "parent_choice" in df.columns:
        group_cols.append("parent_choice")
    run_generation = (
        df.groupby(group_cols, as_index=False)
        .agg(
            trajectory_value=("distance", "mean"),
            lineage_edge_count=("distance", "size"),
        )
    )
    run_generation["trajectory_metric"] = "elite_lineage_parent_child_distance"
    return clean_run_generation_frame(run_generation)


def clean_run_generation_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df["generation_zero_based"] = pd.to_numeric(df["generation_zero_based"], errors="coerce")
    df["trajectory_value"] = pd.to_numeric(df["trajectory_value"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["experiment", "run", "crossover_type", "generation_zero_based", "trajectory_value"]
    )
    df["run"] = df["run"].astype(int)
    df["generation_zero_based"] = df["generation_zero_based"].astype(int)
    df["crossover_label"] = df["crossover_type"].map(display_crossover_name)
    return df.sort_values(["crossover_type", "run", "generation_zero_based"]).reset_index(drop=True)


def descriptive_generation_rows(df: pd.DataFrame, value_label: str) -> list[dict]:
    rows = []
    for (crossover_type, generation), group in df.groupby(
        ["crossover_type", "generation_zero_based"],
        sort=True,
    ):
        values = finite_numeric(group["trajectory_value"])
        if values.empty:
            continue
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        ci_low, ci_high = mean_ci95(values)
        rows.append(
            {
                "metric": value_label,
                "crossover_type": crossover_type,
                "crossover_label": display_crossover_name(crossover_type),
                "generation_zero_based": int(generation),
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


def slope_run_rows(df: pd.DataFrame, value_label: str) -> list[dict]:
    rows = []
    for (experiment, run, crossover_type), group in df.groupby(
        ["experiment", "run", "crossover_type"],
        sort=True,
    ):
        data = group[["generation_zero_based", "trajectory_value"]].dropna().sort_values(
            "generation_zero_based"
        )
        if len(data) < 2 or data["generation_zero_based"].nunique() < 2:
            continue
        regression = stats.linregress(
            data["generation_zero_based"].to_numpy(dtype=float),
            data["trajectory_value"].to_numpy(dtype=float),
        )
        first = data.iloc[0]
        last = data.iloc[-1]
        rows.append(
            {
                "metric": value_label,
                "experiment": experiment,
                "run": int(run),
                "crossover_type": crossover_type,
                "crossover_label": display_crossover_name(crossover_type),
                "n_generations": len(data),
                "first_generation": int(first["generation_zero_based"]),
                "last_generation": int(last["generation_zero_based"]),
                "first_value": float(first["trajectory_value"]),
                "last_value": float(last["trajectory_value"]),
                "last_minus_first": float(last["trajectory_value"] - first["trajectory_value"]),
                "slope_per_generation": float(regression.slope),
                "intercept": float(regression.intercept),
                "r_value": float(regression.rvalue),
                "r_squared": float(regression.rvalue**2),
                "slope_p_value": float(regression.pvalue),
                "slope_std_err": float(regression.stderr),
            }
        )
    return rows


def descriptive_run_level_rows(
    run_df: pd.DataFrame,
    value_col: str,
    value_label: str,
) -> list[dict]:
    rows = []
    for crossover_type, group in sorted(
        run_df.groupby("crossover_type"),
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
                "metric": value_label,
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


def assumption_rows(run_df: pd.DataFrame, value_col: str, value_label: str) -> list[dict]:
    rows = []
    groups = []
    for crossover_type, group in sorted(
        run_df.groupby("crossover_type"),
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
                "metric": value_label,
                "response_column": value_col,
                "test": "shapiro_wilk",
                "crossover_type": crossover_type,
                "n_runs": len(values),
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
            }
        )

    levene_stat, levene_p = safe_levene(groups)
    rows.append(
        {
            "metric": value_label,
            "response_column": value_col,
            "test": "levene_median",
            "crossover_type": "all",
            "n_runs": int(sum(len(group) for group in groups)),
            "statistic": levene_stat,
            "p_value": levene_p,
        }
    )
    return rows


def slope_omnibus_and_pairwise(
    slope_df: pd.DataFrame,
    value_label: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups_by_name = {
        name: group["slope_per_generation"]
        for name, group in sorted(
            slope_df.groupby("crossover_type"),
            key=lambda item: str(item[0]),
        )
    }
    omnibus = selected_between_group_test(
        groups_by_name,
        alpha=alpha,
        note="Independent unit is one per-run linear slope over generation_zero_based.",
    )
    omnibus["metric"] = value_label
    omnibus["response_column"] = "slope_per_generation"

    pairwise_rows = pairwise_between_group_rows(groups_by_name, alpha=alpha)
    for row in pairwise_rows:
        row["metric"] = value_label
        row["response_column"] = "slope_per_generation"
    return pd.DataFrame([omnibus]), pd.DataFrame(pairwise_rows)


def slope_against_zero_rows(
    slope_df: pd.DataFrame,
    value_label: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for crossover_type, group in sorted(
        slope_df.groupby("crossover_type"),
        key=lambda item: str(item[0]),
    ):
        values = finite_numeric(group["slope_per_generation"])
        if len(values) < 2:
            rows.append(
                {
                    "metric": value_label,
                    "crossover_type": crossover_type,
                    "n_runs": len(values),
                    "mean_slope": np.nan,
                    "median_slope": np.nan,
                    "normality_p_value": np.nan,
                    "selected_test": "not_tested",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size_name": "",
                    "effect_size": np.nan,
                    "decision": "not_tested",
                    "direction": "not_tested",
                }
            )
            continue

        shapiro_p = np.nan
        if 3 <= len(values) <= 5000 and values.nunique() > 1:
            _, shapiro_p = stats.shapiro(values)

        normal_enough = pd.notna(shapiro_p) and float(shapiro_p) >= alpha
        if normal_enough:
            statistic, p_value = stats.ttest_1samp(values, popmean=0.0)
            selected_test = "one_sample_t_test"
            effect_name = "cohens_d"
            effect_size = one_sample_cohens_d(values)
        else:
            if values.nunique() <= 1 and float(values.iloc[0]) == 0.0:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
            selected_test = "wilcoxon_signed_rank"
            effect_name = "rank_biserial"
            effect_size = one_sample_rank_biserial(values)

        mean_slope = float(values.mean())
        if mean_slope < 0:
            direction = "decreasing"
        elif mean_slope > 0:
            direction = "increasing"
        else:
            direction = "flat"

        rows.append(
            {
                "metric": value_label,
                "crossover_type": crossover_type,
                "n_runs": len(values),
                "mean_slope": mean_slope,
                "median_slope": float(values.median()),
                "normality_p_value": shapiro_p,
                "selected_test": selected_test,
                "statistic": statistic,
                "p_value": p_value,
                "effect_size_name": effect_name,
                "effect_size": effect_size,
                "decision": decision_from_p(p_value, alpha),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows)


def first_last_run_rows(df: pd.DataFrame, value_label: str) -> pd.DataFrame:
    rows = []
    for (experiment, run, crossover_type), group in df.groupby(
        ["experiment", "run", "crossover_type"],
        sort=True,
    ):
        data = group[["generation_zero_based", "trajectory_value"]].dropna().sort_values(
            "generation_zero_based"
        )
        if len(data) < 2:
            continue
        first = data.iloc[0]
        last = data.iloc[-1]
        if int(first["generation_zero_based"]) == int(last["generation_zero_based"]):
            continue
        rows.append(
            {
                "metric": value_label,
                "experiment": experiment,
                "run": int(run),
                "crossover_type": crossover_type,
                "crossover_label": display_crossover_name(crossover_type),
                "first_generation": int(first["generation_zero_based"]),
                "last_generation": int(last["generation_zero_based"]),
                "first_value": float(first["trajectory_value"]),
                "last_value": float(last["trajectory_value"]),
                "last_minus_first": float(last["trajectory_value"] - first["trajectory_value"]),
            }
        )
    return pd.DataFrame(rows)


def first_last_test_rows(
    first_last_df: pd.DataFrame,
    value_label: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    if first_last_df.empty:
        return pd.DataFrame(rows)

    for crossover_type, group in sorted(
        first_last_df.groupby("crossover_type"),
        key=lambda item: str(item[0]),
    ):
        pair_df = group[["first_value", "last_value", "last_minus_first"]].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        if len(pair_df) < 2:
            rows.append(
                {
                    "metric": value_label,
                    "crossover_type": crossover_type,
                    "n_runs": len(pair_df),
                    "first_generation_min": np.nan,
                    "last_generation_max": np.nan,
                    "first_mean": np.nan,
                    "last_mean": np.nan,
                    "mean_change_last_minus_first": np.nan,
                    "median_change_last_minus_first": np.nan,
                    "percent_change_from_first": np.nan,
                    "normality_p_value": np.nan,
                    "selected_test": "not_tested",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size_name": "",
                    "effect_size": np.nan,
                    "decision": "not_tested",
                    "direction": "not_tested",
                }
            )
            continue

        first_values = pair_df["first_value"]
        last_values = pair_df["last_value"]
        differences = pair_df["last_minus_first"]
        shapiro_p = np.nan
        if 3 <= len(differences) <= 5000 and differences.nunique() > 1:
            _, shapiro_p = stats.shapiro(differences)

        normal_enough = pd.notna(shapiro_p) and float(shapiro_p) >= alpha
        if normal_enough:
            statistic, p_value = stats.ttest_rel(last_values, first_values)
            selected_test = "paired_t_test"
            effect_name = "cohens_dz"
            effect_size = paired_cohens_dz(first_values, last_values)
        else:
            if differences.nunique() <= 1 and float(differences.iloc[0]) == 0.0:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = stats.wilcoxon(
                    last_values,
                    first_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            selected_test = "wilcoxon_signed_rank"
            effect_name = "rank_biserial"
            effect_size = wilcoxon_rank_biserial(first_values, last_values)

        mean_change = float(differences.mean())
        first_mean = float(first_values.mean())
        if mean_change < 0:
            direction = "decreased"
        elif mean_change > 0:
            direction = "increased"
        else:
            direction = "unchanged"

        rows.append(
            {
                "metric": value_label,
                "crossover_type": crossover_type,
                "n_runs": len(pair_df),
                "first_generation_min": int(group["first_generation"].min()),
                "last_generation_max": int(group["last_generation"].max()),
                "first_mean": first_mean,
                "last_mean": float(last_values.mean()),
                "mean_change_last_minus_first": mean_change,
                "median_change_last_minus_first": float(differences.median()),
                "percent_change_from_first": (
                    float((mean_change / first_mean) * 100.0)
                    if first_mean != 0
                    else np.nan
                ),
                "normality_p_value": shapiro_p,
                "selected_test": selected_test,
                "statistic": statistic,
                "p_value": p_value,
                "effect_size_name": effect_name,
                "effect_size": effect_size,
                "decision": decision_from_p(p_value, alpha),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows)


def generation_between_group_tests(
    df: pd.DataFrame,
    value_label: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    pairwise_rows = []
    for generation, group in df.groupby("generation_zero_based", sort=True):
        groups_by_name = {
            name: sub_group["trajectory_value"]
            for name, sub_group in sorted(
                group.groupby("crossover_type"),
                key=lambda item: str(item[0]),
            )
        }
        omnibus = selected_between_group_test(
            groups_by_name,
            alpha=alpha,
            note="Independent unit is one run-level trajectory value at this generation.",
        )
        omnibus["metric"] = value_label
        omnibus["generation_zero_based"] = int(generation)
        omnibus_rows.append(omnibus)

        for row in pairwise_between_group_rows(groups_by_name, alpha=alpha):
            row["metric"] = value_label
            row["generation_zero_based"] = int(generation)
            pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def format_table_value(value):
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return value


def format_results_table(df: pd.DataFrame, columns: list[str], max_rows: Optional[int] = None) -> str:
    available_columns = [column for column in columns if column in df.columns]
    if df.empty or not available_columns:
        return "No results."
    table = df[available_columns].copy()
    if max_rows is not None and len(table) > max_rows:
        table = table.head(max_rows).copy()
    for column in table.columns:
        table[column] = table[column].map(format_table_value)
    return table.to_string(index=False)


def write_results_file(
    *,
    output_path: Path,
    title: str,
    alpha: float,
    value_label: str,
    descriptive_df: pd.DataFrame,
    slope_descriptive_df: pd.DataFrame,
    slope_assumptions_df: pd.DataFrame,
    slope_omnibus_df: pd.DataFrame,
    slope_pairwise_df: pd.DataFrame,
    slope_zero_df: pd.DataFrame,
    first_last_tests_df: pd.DataFrame,
    generation_omnibus_df: pd.DataFrame,
    generation_pairwise_df: pd.DataFrame,
) -> Path:
    significant_generations = generation_omnibus_df[
        generation_omnibus_df["decision"] == "significant"
    ]
    significant_pairwise = generation_pairwise_df[
        generation_pairwise_df["decision_holm"] == "significant"
    ]
    lines = [
        title,
        "=" * len(title),
        "",
        f"Alpha: {alpha}",
        f"Metric: {value_label}",
        "Independent unit: one run-level value per generation, or one run-level slope/change.",
        "",
        "Run-Level Slope Summary",
        "-----------------------",
        format_results_table(
            slope_descriptive_df,
            [
                "crossover_type",
                "n_runs",
                "mean",
                "std",
                "median",
                "q25",
                "q75",
                "ci95_low",
                "ci95_high",
            ],
        ),
        "",
        "Slope Assumption Tests",
        "----------------------",
        format_results_table(
            slope_assumptions_df,
            ["test", "crossover_type", "n_runs", "statistic", "p_value"],
        ),
        "",
        "Between-Crossover Slope Test",
        "----------------------------",
        format_results_table(
            slope_omnibus_df,
            [
                "selected_test",
                "n_total",
                "group_count",
                "statistic",
                "p_value",
                "effect_size_name",
                "effect_size",
                "decision",
            ],
        ),
        "",
        "Between-Crossover Slope Pairwise Tests",
        "--------------------------------------",
        format_results_table(
            slope_pairwise_df,
            [
                "test",
                "group_a",
                "group_b",
                "n_a",
                "n_b",
                "mean_difference_a_minus_b",
                "p_value_raw",
                "p_value_holm",
                "effect_size_name",
                "effect_size",
                "decision_holm",
            ],
        ),
        "",
        "Slope-vs-Zero Tests",
        "-------------------",
        format_results_table(
            slope_zero_df,
            [
                "crossover_type",
                "n_runs",
                "mean_slope",
                "selected_test",
                "statistic",
                "p_value",
                "effect_size_name",
                "effect_size",
                "decision",
                "direction",
            ],
        ),
        "",
        "First-vs-Last Generation Tests",
        "------------------------------",
        format_results_table(
            first_last_tests_df,
            [
                "crossover_type",
                "n_runs",
                "first_generation_min",
                "last_generation_max",
                "first_mean",
                "last_mean",
                "mean_change_last_minus_first",
                "percent_change_from_first",
                "selected_test",
                "p_value",
                "effect_size_name",
                "effect_size",
                "decision",
                "direction",
            ],
        ),
        "",
        "Significant Generation-Level Omnibus Tests",
        "------------------------------------------",
        (
            format_results_table(
                significant_generations,
                [
                    "generation_zero_based",
                    "selected_test",
                    "n_total",
                    "statistic",
                    "p_value",
                    "effect_size_name",
                    "effect_size",
                    "decision",
                ],
                max_rows=40,
            )
            if not significant_generations.empty
            else "No significant generation-level omnibus tests."
        ),
        "",
        "Significant Generation-Level Pairwise Tests",
        "-------------------------------------------",
        (
            format_results_table(
                significant_pairwise,
                [
                    "generation_zero_based",
                    "test",
                    "group_a",
                    "group_b",
                    "n_a",
                    "n_b",
                    "mean_difference_a_minus_b",
                    "p_value_holm",
                    "effect_size_name",
                    "effect_size",
                    "decision_holm",
                ],
                max_rows=80,
            )
            if not significant_pairwise.empty
            else "No significant Holm-corrected generation-level pairwise tests."
        ),
        "",
        "Generation-Level Descriptives Preview",
        "-------------------------------------",
        format_results_table(
            descriptive_df,
            [
                "crossover_type",
                "generation_zero_based",
                "n_runs",
                "mean",
                "std",
                "median",
                "q25",
                "q75",
            ],
            max_rows=24,
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_target_tests(
    *,
    analysis_dir: Path,
    target_name: str,
    run_generation_df: pd.DataFrame,
    value_label: str,
    alpha: float,
) -> dict[str, Path]:
    prefix = f"{target_name}_statistical"
    descriptive_df = pd.DataFrame(descriptive_generation_rows(run_generation_df, value_label))
    slope_df = pd.DataFrame(slope_run_rows(run_generation_df, value_label))
    slope_descriptive_df = pd.DataFrame(
        descriptive_run_level_rows(slope_df, "slope_per_generation", value_label)
    )
    slope_assumptions_df = pd.DataFrame(
        assumption_rows(slope_df, "slope_per_generation", value_label)
    )
    slope_omnibus_df, slope_pairwise_df = slope_omnibus_and_pairwise(
        slope_df,
        value_label,
        alpha,
    )
    slope_zero_df = slope_against_zero_rows(slope_df, value_label, alpha)
    first_last_df = first_last_run_rows(run_generation_df, value_label)
    first_last_tests_df = first_last_test_rows(first_last_df, value_label, alpha)
    generation_omnibus_df, generation_pairwise_df = generation_between_group_tests(
        run_generation_df,
        value_label,
        alpha,
    )

    output_paths = {
        "descriptive_stats": analysis_dir / f"{prefix}_descriptive_stats.csv",
        "slope_run_values": analysis_dir / f"{prefix}_slope_run_values.csv",
        "slope_descriptive_stats": analysis_dir / f"{prefix}_slope_descriptive_stats.csv",
        "slope_assumption_tests": analysis_dir / f"{prefix}_slope_assumption_tests.csv",
        "slope_omnibus_tests": analysis_dir / f"{prefix}_slope_omnibus_tests.csv",
        "slope_pairwise_tests": analysis_dir / f"{prefix}_slope_pairwise_tests.csv",
        "slope_vs_zero_tests": analysis_dir / f"{prefix}_slope_vs_zero_tests.csv",
        "first_last_generation_run_values": analysis_dir / f"{prefix}_first_last_generation_run_values.csv",
        "first_last_generation_tests": analysis_dir / f"{prefix}_first_last_generation_tests.csv",
        "generation_omnibus_tests": analysis_dir / f"{prefix}_generation_omnibus_tests.csv",
        "generation_pairwise_tests": analysis_dir / f"{prefix}_generation_pairwise_tests.csv",
    }

    descriptive_df.to_csv(output_paths["descriptive_stats"], index=False)
    slope_df.to_csv(output_paths["slope_run_values"], index=False)
    slope_descriptive_df.to_csv(output_paths["slope_descriptive_stats"], index=False)
    slope_assumptions_df.to_csv(output_paths["slope_assumption_tests"], index=False)
    slope_omnibus_df.to_csv(output_paths["slope_omnibus_tests"], index=False)
    slope_pairwise_df.to_csv(output_paths["slope_pairwise_tests"], index=False)
    slope_zero_df.to_csv(output_paths["slope_vs_zero_tests"], index=False)
    first_last_df.to_csv(output_paths["first_last_generation_run_values"], index=False)
    first_last_tests_df.to_csv(output_paths["first_last_generation_tests"], index=False)
    generation_omnibus_df.to_csv(output_paths["generation_omnibus_tests"], index=False)
    generation_pairwise_df.to_csv(output_paths["generation_pairwise_tests"], index=False)

    results_path = write_results_file(
        output_path=analysis_dir / f"{prefix}_results.txt",
        title=f"{target_name.replace('_', ' ').title()} Statistical Results",
        alpha=alpha,
        value_label=value_label,
        descriptive_df=descriptive_df,
        slope_descriptive_df=slope_descriptive_df,
        slope_assumptions_df=slope_assumptions_df,
        slope_omnibus_df=slope_omnibus_df,
        slope_pairwise_df=slope_pairwise_df,
        slope_zero_df=slope_zero_df,
        first_last_tests_df=first_last_tests_df,
        generation_omnibus_df=generation_omnibus_df,
        generation_pairwise_df=generation_pairwise_df,
    )
    output_paths["statistical_results"] = results_path

    for path in output_paths.values():
        print(f"Saved temporal statistical output to: {path}")
    return output_paths


def run_temporal_trajectory_tests(
    *,
    analysis_dir: Path,
    target: str = "both",
    alpha: float = 0.05,
    diversity_run_name: str = "mean_population_diversity_by_run_generation.csv",
    lineage_name: str = "elite_lineage_parent_child_distances.csv",
    lineage_parent_choice: str = "elite_parent",
) -> dict[str, dict[str, Path]]:
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if target in ("both", DIVERSITY_TARGET):
        diversity_df = load_diversity_run_generation(analysis_dir, diversity_run_name)
        outputs[DIVERSITY_TARGET] = run_target_tests(
            analysis_dir=analysis_dir,
            target_name=DIVERSITY_TARGET,
            run_generation_df=diversity_df,
            value_label="mean_pairwise_morphological_distance",
            alpha=alpha,
        )

    if target in ("both", LINEAGE_TARGET):
        lineage_df = load_lineage_run_generation(
            analysis_dir,
            lineage_name,
            lineage_parent_choice,
        )
        outputs[LINEAGE_TARGET] = run_target_tests(
            analysis_dir=analysis_dir,
            target_name=LINEAGE_TARGET,
            run_generation_df=lineage_df,
            value_label="elite_lineage_parent_child_distance",
            alpha=alpha,
        )

    return outputs


def main():
    args = parse_args()
    run_temporal_trajectory_tests(
        analysis_dir=Path(args.analysis_dir),
        target=args.target,
        alpha=args.alpha,
        diversity_run_name=args.diversity_run_name,
        lineage_name=args.lineage_name,
        lineage_parent_choice=args.lineage_parent_choice,
    )


if __name__ == "__main__":
    main()
