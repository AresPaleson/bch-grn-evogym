#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
import textwrap
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .crossover_labels import (
        CROSSOVER_COLORS,
        CROSSOVER_ORDER,
        display_crossover_name,
        infer_crossover_type,
    )
except ImportError:
    from crossover_labels import (
        CROSSOVER_COLORS,
        CROSSOVER_ORDER,
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


def ordered_crossover_values(values: Iterable) -> list[str]:
    unique = [str(value) for value in pd.Series(list(values)).dropna().unique()]
    ordered = [value for value in CROSSOVER_ORDER if value in unique]
    ordered.extend(sorted(value for value in unique if value not in ordered))
    return ordered


def metric_label(metric: str) -> str:
    return str(metric).replace("_", " ")


def short_crossover_name(crossover_type: str) -> str:
    labels = {
        "promoter_aligned_cut_and_splice": "cut/splice",
        "arithmetic_recombination": "arithmetic",
        "homologous_gene_block_recombination": "homologous",
    }
    return labels.get(str(crossover_type), display_crossover_name(crossover_type))


def plot_statistical_summary(
    analysis_dir: Path,
    alpha: float,
    metrics: list[str],
    run_means: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> Path:
    output_path = analysis_dir / "final_generation_population_statistical_summary.png"
    setup_counts = (
        run_means.groupby(["crossover_type", "crossover_label"])["run"]
        .nunique()
        .reset_index(name="n_runs")
        .sort_values("crossover_type")
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    fig.suptitle(
        "Final-generation population statistical summary",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    primary_metric = "fitness" if "fitness" in metrics else metrics[0] if metrics else ""
    primary_desc = descriptive_df[descriptive_df["metric"] == primary_metric].copy()
    if primary_desc.empty:
        ax.text(0.5, 0.5, "No primary final-generation metric", ha="center", va="center")
        ax.set_axis_off()
    else:
        order = ordered_crossover_values(primary_desc["crossover_type"])
        primary_desc["order"] = primary_desc["crossover_type"].map({name: idx for idx, name in enumerate(order)})
        primary_desc = primary_desc.sort_values("order")
        labels = [display_crossover_name(name) for name in primary_desc["crossover_type"]]
        means = primary_desc["mean"].to_numpy(dtype=float)
        ci_low = primary_desc["ci95_low"].to_numpy(dtype=float)
        ci_high = primary_desc["ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([means - ci_low, ci_high - means])
        colors = [CROSSOVER_COLORS.get(name, "#6C757D") for name in primary_desc["crossover_type"]]
        ax.bar(labels, means, yerr=yerr, capsize=5, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_title(f"Primary final-generation outcome: {metric_label(primary_metric)}")
        ax.set_ylabel("Run-level final population mean")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        primary_omnibus = omnibus_df[omnibus_df["metric"] == primary_metric]
        if not primary_omnibus.empty:
            row = primary_omnibus.iloc[0]
            effect_text = (
                f"{row['effect_size_name']}={float(row['effect_size']):.4f}"
                if row["effect_size_name"] and pd.notna(row["effect_size"])
                else "effect_size=NA"
            )
            ax.text(
                0.02,
                0.96,
                f"{row['selected_test']}: p={format_p(row['p_value'])}, {effect_text}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
            )

    ax = axes[0, 1]
    if omnibus_df.empty:
        ax.text(0.5, 0.5, "No omnibus tests", ha="center", va="center")
        ax.set_axis_off()
    else:
        omni = omnibus_df.copy()
        omni["effect_size"] = pd.to_numeric(omni["effect_size"], errors="coerce")
        omni = omni.sort_values("effect_size", ascending=True)
        colors = np.where(omni["decision"] == "significant", "#4E79A7", "#B8B8B8")
        ax.barh([metric_label(metric) for metric in omni["metric"]], omni["effect_size"], color=colors)
        ax.set_title("Omnibus/direct effect size by metric")
        ax.set_xlabel("Effect size")
        ax.grid(axis="x", alpha=0.25)
        for idx, row in enumerate(omni.itertuples(index=False)):
            value = row.effect_size
            if pd.notna(value):
                ax.text(value + 0.01, idx, f"p={format_p(row.p_value)}", va="center", fontsize=7)

    ax = axes[1, 0]
    significant_pairs = pairwise_df[pairwise_df["decision_holm"] == "significant"].copy() if not pairwise_df.empty else pd.DataFrame()
    if significant_pairs.empty:
        ax.text(0.5, 0.5, "No significant Holm-corrected pairwise tests", ha="center", va="center")
        ax.set_axis_off()
    else:
        significant_pairs["abs_effect"] = pd.to_numeric(significant_pairs["effect_size"], errors="coerce").abs()
        significant_pairs = significant_pairs.sort_values("abs_effect", ascending=True).tail(12)
        labels = [
            (
                f"{metric_label(row.metric)}: "
                f"{short_crossover_name(row.group_a)} vs {short_crossover_name(row.group_b)}"
            )
            for row in significant_pairs.itertuples(index=False)
        ]
        effects = significant_pairs["effect_size"].to_numpy(dtype=float)
        ax.barh(labels, effects, color="#59A14F")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title("Largest significant pairwise effects")
        ax.set_xlabel("Effect size")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
        for idx, row in enumerate(significant_pairs.itertuples(index=False)):
            ax.text(row.effect_size, idx, f" p={format_p(row.p_value_holm)}", va="center", fontsize=7)

    ax = axes[1, 1]
    ax.set_axis_off()
    summary_lines = [
        f"Alpha: {alpha}",
        "Independent unit: one final-generation population mean per run.",
        "Methods: Shapiro-Wilk checks choose Welch/ANOVA or Mann-Whitney/Kruskal-Wallis; pairwise tests use Holm correction.",
    ]
    if not setup_counts.empty:
        setup_text = "; ".join(
            f"{row.crossover_label}: n={int(row.n_runs)}"
            for row in setup_counts.itertuples(index=False)
        )
        summary_lines.append("Setups: " + setup_text + ".")
    if not omnibus_df.empty:
        significant_omni = omnibus_df[omnibus_df["decision"] == "significant"]
        if significant_omni.empty:
            summary_lines.append("No omnibus/direct tests were significant.")
        else:
            result_text = "; ".join(
                f"{metric_label(row.metric)} p={format_p(row.p_value)}, {row.effect_size_name}={row.effect_size:.2f}"
                for row in significant_omni.itertuples(index=False)
                if pd.notna(row.effect_size)
            )
            summary_lines.append("Significant omnibus/direct tests: " + result_text + ".")
    if not significant_pairs.empty:
        pair_text = "; ".join(
            (
                f"{metric_label(row.metric)} {short_crossover_name(row.group_a)} vs {short_crossover_name(row.group_b)} "
                f"p={format_p(row.p_value_holm)}, {row.effect_size_name}={row.effect_size:.2f}"
            )
            for row in significant_pairs.tail(6).itertuples(index=False)
        )
        summary_lines.append("Selected significant pairwise tests: " + pair_text + ".")

    wrapped_lines = []
    for line in summary_lines:
        wrapped_lines.extend(textwrap.wrap(line, width=68) or [""])
        wrapped_lines.append("")
    ax.text(
        0.0,
        1.0,
        "\n".join(wrapped_lines).strip(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        linespacing=1.35,
    )
    ax.set_title("Statistical result table", loc="left")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


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

    chart_path = plot_statistical_summary(
        analysis_dir,
        alpha,
        metrics,
        run_means,
        descriptive_df,
        omnibus_df,
        pairwise_df,
    )
    output_paths["statistical_summary_chart"] = chart_path

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
