#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_METRICS = (
    "morph_distance",
    "shape_distance",
    "occupied_union_material_distance",
    "material_substitution_distance",
    "trait_profile_distance",
    "parent_pair_morph_distance",
    "child_fitness",
    "fitness_delta",
)

CORRELATION_DISTANCE_METRICS = (
    "morph_distance",
    "shape_distance",
    "occupied_union_material_distance",
    "material_substitution_distance",
    "trait_profile_distance",
)

CORRELATION_OUTCOMES = (
    "child_fitness",
    "fitness_delta",
)

GENERATION_CHANGE_DISTANCE_METRICS = (
    "morph_distance",
    "shape_distance",
    "occupied_union_material_distance",
    "material_substitution_distance",
    "trait_profile_distance",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run statistical tests on parent-child morphology summaries."
        )
    )
    parser.add_argument("--analysis-dir", required=True, type=str)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--parent-slot", default="closest_parent", type=str)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        type=str,
        help="Comma-separated base metric names. Each is read as <metric>_mean from the run summary.",
    )
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def format_p(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def decision_from_p(p_value, alpha: float) -> str:
    if p_value is None or pd.isna(p_value):
        return "not_tested"
    return "significant" if float(p_value) < alpha else "not_significant"


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    indexed = [
        (idx, float(p))
        for idx, p in enumerate(p_values)
        if p is not None and not pd.isna(p)
    ]
    adjusted = [np.nan] * len(list(p_values)) if not isinstance(p_values, list) else [np.nan] * len(p_values)
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
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    sem = stats.sem(values)
    margin = stats.t.ppf(0.975, df=n - 1) * sem
    return float(values.mean() - margin), float(values.mean() + margin)


def descriptive_rows(df: pd.DataFrame, metric: str, value_col: str):
    rows = []
    for crossover_type, group in sorted(df.groupby("crossover_type"), key=lambda item: str(item[0])):
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


def assumption_rows(df: pd.DataFrame, metric: str, value_col: str):
    rows = []
    groups = []
    for crossover_type, group in sorted(df.groupby("crossover_type"), key=lambda item: str(item[0])):
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

    levene_stat = np.nan
    levene_p = np.nan
    valid_groups = [group for group in groups if len(group) >= 2 and group.nunique() > 1]
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


def assumptions_support_parametric(assumptions_df: pd.DataFrame, metric: str, alpha: float) -> tuple[bool, float]:
    metric_assumptions = assumptions_df[assumptions_df["metric"] == metric]
    shapiro = metric_assumptions[metric_assumptions["test"] == "shapiro_wilk"].copy()
    shapiro = shapiro[shapiro["p_value"].notna()]
    levene = metric_assumptions[metric_assumptions["test"] == "levene_median"]
    levene_p = levene["p_value"].iloc[0] if not levene.empty else np.nan

    normal_enough = not shapiro.empty and bool((shapiro["p_value"] >= alpha).all())
    equal_variance = pd.notna(levene_p) and float(levene_p) >= alpha
    return normal_enough and equal_variance, levene_p


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


def omnibus_and_pairwise(df: pd.DataFrame, metric: str, value_col: str, assumptions_df: pd.DataFrame, alpha: float):
    all_groups_by_name = {
        name: finite_numeric(group[value_col])
        for name, group in sorted(df.groupby("crossover_type"), key=lambda item: str(item[0]))
    }
    all_groups_by_name = {name: values for name, values in all_groups_by_name.items() if len(values) >= 1}
    groups_by_name = {name: values for name, values in all_groups_by_name.items() if len(values) >= 2}

    if len(groups_by_name) < 2:
        return (
            [
                {
                    "metric": metric,
                    "response_column": value_col,
                    "selected_test": "not_tested",
                    "n_total": int(sum(len(values) for values in all_groups_by_name.values())),
                    "group_count": len(all_groups_by_name),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size_name": "",
                    "effect_size": np.nan,
                    "assumptions_supported_parametric": False,
                    "levene_p_value": np.nan,
                    "decision": "not_tested",
                    "note": "At least two crossover groups with two or more independent runs are required.",
                }
            ],
            [],
        )

    parametric_ok, levene_p = assumptions_support_parametric(assumptions_df, metric, alpha)
    groups = list(groups_by_name.values())

    if parametric_ok:
        statistic, p_value = stats.f_oneway(*groups)
        selected_test = "one_way_anova"
        effect_name = "eta_squared"
        effect_size = eta_squared(groups)
    else:
        statistic, p_value = stats.kruskal(*groups)
        selected_test = "kruskal_wallis"
        effect_name = "epsilon_squared"
        effect_size = epsilon_squared_kruskal(
            float(statistic),
            int(sum(len(group) for group in groups)),
            len(groups),
        )

    omnibus_rows = [
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
            "note": "",
        }
    ]

    pairwise_rows = []
    raw_p_values = []
    for group_a, group_b in combinations(groups_by_name, 2):
        values_a = groups_by_name[group_a]
        values_b = groups_by_name[group_b]
        if selected_test == "one_way_anova":
            pair_stat, pair_p = stats.ttest_ind(values_a, values_b, equal_var=True)
            pair_test = "independent_t_test"
            effect_name_pair = "hedges_g"
            effect_size_pair = hedges_g(values_a, values_b)
        else:
            pair_stat, pair_p = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            pair_test = "mann_whitney_u"
            effect_name_pair = "cliffs_delta"
            effect_size_pair = cliffs_delta(values_a, values_b)

        raw_p_values.append(pair_p)
        pairwise_rows.append(
            {
                "metric": metric,
                "response_column": value_col,
                "test": pair_test,
                "group_a": group_a,
                "group_b": group_b,
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
                "effect_size_name": effect_name_pair,
                "effect_size": effect_size_pair,
                "decision_holm": "not_tested",
            }
        )

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(pairwise_rows, adjusted):
        row["p_value_holm"] = adjusted_p
        row["decision_holm"] = decision_from_p(adjusted_p, alpha)

    return omnibus_rows, pairwise_rows


def correlation_rows(links_df: pd.DataFrame, alpha: float):
    rows = []
    if links_df.empty:
        return rows

    closest_df = links_df[links_df["parent_slot"] == "closest_parent"].copy()
    if closest_df.empty:
        return rows

    groups = [("all", closest_df)]
    groups.extend(
        sorted(
            [(name, group) for name, group in closest_df.groupby("crossover_type")],
            key=lambda item: str(item[0]),
        )
    )

    for group_name, group in groups:
        for distance_metric in CORRELATION_DISTANCE_METRICS:
            for outcome in CORRELATION_OUTCOMES:
                if distance_metric not in group.columns or outcome not in group.columns:
                    continue
                data = group[[distance_metric, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) < 3 or data[distance_metric].nunique() < 2 or data[outcome].nunique() < 2:
                    rho = np.nan
                    p_value = np.nan
                else:
                    rho, p_value = stats.spearmanr(data[distance_metric], data[outcome])
                rows.append(
                    {
                        "group": group_name,
                        "distance_metric": distance_metric,
                        "outcome": outcome,
                        "n_links": len(data),
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "decision": decision_from_p(p_value, alpha),
                    }
                )
    return rows


def first_last_generation_run_means(
    links_df: pd.DataFrame,
    metrics: list[str],
    parent_slot: str,
) -> pd.DataFrame:
    if links_df.empty:
        return pd.DataFrame()

    df = links_df.copy()
    if "parent_slot" in df.columns:
        df = df[df["parent_slot"] == parent_slot].copy()
    if df.empty:
        return pd.DataFrame()

    needed = ["experiment", "crossover_type", "run", "child_generation"]
    if not all(column in df.columns for column in needed):
        return pd.DataFrame()

    available_metrics = [metric for metric in metrics if metric in df.columns]
    if not available_metrics:
        return pd.DataFrame()

    for column in ["run", "child_generation", *available_metrics]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=needed)
    if df.empty:
        return pd.DataFrame()

    generation_means = (
        df.groupby(needed, as_index=False)[available_metrics]
        .mean(numeric_only=True)
        .sort_values(["experiment", "run", "child_generation"])
    )

    rows = []
    for (experiment, crossover_type, run), group in generation_means.groupby(
        ["experiment", "crossover_type", "run"],
        sort=True,
    ):
        group = group.sort_values("child_generation")
        first = group.iloc[0]
        last = group.iloc[-1]
        if int(first["child_generation"]) == int(last["child_generation"]):
            continue
        row = {
            "experiment": experiment,
            "crossover_type": crossover_type,
            "run": int(run),
            "first_child_generation": int(first["child_generation"]),
            "last_child_generation": int(last["child_generation"]),
        }
        for metric in available_metrics:
            first_value = first[metric]
            last_value = last[metric]
            row[f"{metric}_first_mean"] = first_value
            row[f"{metric}_last_mean"] = last_value
            row[f"{metric}_last_minus_first"] = last_value - first_value
        rows.append(row)

    return pd.DataFrame(rows)


def generation_change_test_rows(
    first_last_df: pd.DataFrame,
    metrics: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows = []
    if first_last_df.empty:
        return pd.DataFrame(rows)

    for metric in metrics:
        first_col = f"{metric}_first_mean"
        last_col = f"{metric}_last_mean"
        diff_col = f"{metric}_last_minus_first"
        if first_col not in first_last_df.columns or last_col not in first_last_df.columns:
            continue

        for crossover_type, group in sorted(
            first_last_df.groupby("crossover_type"),
            key=lambda item: str(item[0]),
        ):
            pair_df = group[["run", first_col, last_col, diff_col]].replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()
            if len(pair_df) < 2:
                rows.append(
                    {
                        "metric": metric,
                        "crossover_type": crossover_type,
                        "n_runs": len(pair_df),
                        "first_generation": np.nan,
                        "last_generation": np.nan,
                        "first_mean": np.nan,
                        "last_mean": np.nan,
                        "mean_change_last_minus_first": np.nan,
                        "median_change_last_minus_first": np.nan,
                        "percent_change_from_first": np.nan,
                        "normality_test": "not_tested",
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

            first_values = pair_df[first_col]
            last_values = pair_df[last_col]
            differences = pair_df[diff_col]
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
                    "metric": metric,
                    "crossover_type": crossover_type,
                    "n_runs": len(pair_df),
                    "first_generation": int(group["first_child_generation"].min()),
                    "last_generation": int(group["last_child_generation"].max()),
                    "first_mean": first_mean,
                    "last_mean": float(last_values.mean()),
                    "mean_change_last_minus_first": mean_change,
                    "median_change_last_minus_first": float(differences.median()),
                    "percent_change_from_first": (
                        float((mean_change / first_mean) * 100.0)
                        if first_mean != 0
                        else np.nan
                    ),
                    "normality_test": "shapiro_wilk_differences",
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


def format_table_value(value):
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return value


def format_results_table(df: pd.DataFrame, columns: list[str]) -> str:
    available_columns = [column for column in columns if column in df.columns]
    if df.empty or not available_columns:
        return "No results."
    table = df[available_columns].copy()
    for column in table.columns:
        table[column] = table[column].map(format_table_value)
    return table.to_string(index=False)


def write_results_file(
    *,
    analysis_dir: Path,
    alpha: float,
    parent_slot: str,
    metrics: list[str],
    descriptive_df: pd.DataFrame,
    assumptions_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    generation_change_df: pd.DataFrame,
) -> Path:
    output_path = analysis_dir / "parent_child_morphology_statistical_results.txt"
    lines = [
        "Parent-Child Morphology Statistical Results",
        "==========================================",
        "",
        f"Alpha: {alpha}",
        f"Parent-child view: {parent_slot}",
        f"Metrics tested: {', '.join(metrics)}",
        "",
        "Descriptive Statistics",
        "----------------------",
        format_results_table(
            descriptive_df,
            [
                "metric",
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
        "Assumption Tests",
        "----------------",
        format_results_table(
            assumptions_df,
            [
                "metric",
                "test",
                "crossover_type",
                "n_runs",
                "statistic",
                "p_value",
            ],
        ),
        "",
        "Omnibus Tests",
        "-------------",
        format_results_table(
            omnibus_df,
            [
                "metric",
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
        "Pairwise Tests",
        "--------------",
        format_results_table(
            pairwise_df,
            [
                "metric",
                "test",
                "group_a",
                "group_b",
                "n_a",
                "n_b",
                "mean_difference_a_minus_b",
                "statistic",
                "p_value_raw",
                "p_value_holm",
                "effect_size_name",
                "effect_size",
                "decision_holm",
            ],
        ),
        "",
        "Fitness Correlations",
        "--------------------",
        format_results_table(
            correlations_df,
            [
                "group",
                "distance_metric",
                "outcome",
                "n_links",
                "spearman_rho",
                "p_value",
                "decision",
            ],
        ),
        "",
        "First-vs-Last Generation Tests",
        "------------------------------",
        format_results_table(
            generation_change_df,
            [
                "metric",
                "crossover_type",
                "n_runs",
                "first_generation",
                "last_generation",
                "first_mean",
                "last_mean",
                "mean_change_last_minus_first",
                "percent_change_from_first",
                "selected_test",
                "statistic",
                "p_value",
                "effect_size_name",
                "effect_size",
                "decision",
                "direction",
            ],
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_statistical_tests(
    *,
    analysis_dir: Path,
    alpha: float = 0.05,
    parent_slot: str = "closest_parent",
    metrics_raw: str = ",".join(DEFAULT_METRICS),
) -> dict[str, Path]:
    analysis_dir = Path(analysis_dir)
    summary_path = analysis_dir / "parent_child_morphology_summary_by_run.csv"
    links_path = analysis_dir / "parent_child_fitness_distance_links.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run-level morphology summary: {summary_path}")

    run_df = pd.read_csv(summary_path)
    if "parent_slot" in run_df.columns:
        run_df = run_df[run_df["parent_slot"] == parent_slot].copy()

    metrics = parse_csv_arg(metrics_raw)
    descriptive = []
    assumptions = []
    omnibus = []
    pairwise = []

    available_metrics = []
    for metric in metrics:
        value_col = f"{metric}_mean"
        if value_col not in run_df.columns:
            continue
        metric_df = run_df[["crossover_type", "run", value_col]].copy()
        metric_df[value_col] = finite_numeric(metric_df[value_col])
        metric_df = metric_df.dropna(subset=[value_col])
        if metric_df.empty:
            continue
        available_metrics.append(metric)
        descriptive.extend(descriptive_rows(metric_df, metric, value_col))
        assumptions.extend(assumption_rows(metric_df, metric, value_col))

    descriptive_df = pd.DataFrame(descriptive)
    assumptions_df = pd.DataFrame(assumptions)

    for metric in available_metrics:
        value_col = f"{metric}_mean"
        metric_df = run_df[["crossover_type", "run", value_col]].copy()
        metric_df[value_col] = finite_numeric(metric_df[value_col])
        metric_df = metric_df.dropna(subset=[value_col])
        omnibus_rows, pairwise_rows = omnibus_and_pairwise(
            metric_df,
            metric,
            value_col,
            assumptions_df,
            alpha,
        )
        omnibus.extend(omnibus_rows)
        pairwise.extend(pairwise_rows)

    omnibus_df = pd.DataFrame(omnibus)
    pairwise_df = pd.DataFrame(pairwise)

    links_df = pd.read_csv(links_path, low_memory=False) if links_path.exists() else pd.DataFrame()
    correlations_df = pd.DataFrame(correlation_rows(links_df, alpha))
    generation_change_metrics = [
        metric
        for metric in GENERATION_CHANGE_DISTANCE_METRICS
        if metric in metrics and metric in links_df.columns
    ]
    first_last_generation_df = first_last_generation_run_means(
        links_df,
        generation_change_metrics,
        parent_slot,
    )
    generation_change_df = generation_change_test_rows(
        first_last_generation_df,
        generation_change_metrics,
        alpha,
    )

    output_paths = {
        "descriptive_stats": analysis_dir / "parent_child_morphology_descriptive_stats.csv",
        "assumption_tests": analysis_dir / "parent_child_morphology_assumption_tests.csv",
        "omnibus_tests": analysis_dir / "parent_child_morphology_omnibus_tests.csv",
        "pairwise_tests": analysis_dir / "parent_child_morphology_pairwise_tests.csv",
        "fitness_correlations": analysis_dir / "parent_child_morphology_fitness_correlations.csv",
        "first_last_generation_run_means": analysis_dir / "parent_child_morphology_first_last_generation_run_means.csv",
        "first_last_generation_tests": analysis_dir / "parent_child_morphology_first_last_generation_tests.csv",
    }
    legacy_report_path = analysis_dir / "parent_child_morphology_statistical_report.txt"

    descriptive_df.to_csv(output_paths["descriptive_stats"], index=False)
    assumptions_df.to_csv(output_paths["assumption_tests"], index=False)
    omnibus_df.to_csv(output_paths["omnibus_tests"], index=False)
    pairwise_df.to_csv(output_paths["pairwise_tests"], index=False)
    correlations_df.to_csv(output_paths["fitness_correlations"], index=False)
    first_last_generation_df.to_csv(output_paths["first_last_generation_run_means"], index=False)
    generation_change_df.to_csv(output_paths["first_last_generation_tests"], index=False)

    if legacy_report_path.exists():
        legacy_report_path.unlink()

    results_path = write_results_file(
        analysis_dir=analysis_dir,
        alpha=alpha,
        parent_slot=parent_slot,
        metrics=available_metrics,
        descriptive_df=descriptive_df,
        assumptions_df=assumptions_df,
        omnibus_df=omnibus_df,
        pairwise_df=pairwise_df,
        correlations_df=correlations_df,
        generation_change_df=generation_change_df,
    )
    output_paths["statistical_results"] = results_path

    for path in output_paths.values():
        print(f"Saved statistical output to: {path}")
    return output_paths


def main():
    args = parse_args()
    run_statistical_tests(
        analysis_dir=Path(args.analysis_dir),
        alpha=args.alpha,
        parent_slot=args.parent_slot,
        metrics_raw=args.metrics,
    )


if __name__ == "__main__":
    main()
