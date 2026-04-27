#!/usr/bin/env python3
"""
Statistical analysis for prompting strategies (S1, S2, S3).

Expected data (CSV/TSV/XLSX) should contain:
  - prompt/group column (default: "struktur")
  - metric columns, e.g. faa, clarity, context, qwork, optional bertscore

Output:
  - Markdown report (default: stats_report_from_csv.md)
  - Console summary

Default behavior (no arguments):
  - Input: "Metriks RSBP - analytics (2).csv" in the same folder as this script
  - Prompt column: "struktur"
  - Metrics: FAA, clarity, context, q_work, bert_mean
  - Groups: S1, S2, S3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = str(BASE_DIR / "Metriks RSBP - analytics (2).csv")
DEFAULT_OUTPUT = str(BASE_DIR / "stats_report_from_csv.md")
DEFAULT_PROMPT_COL = "struktur"
DEFAULT_METRICS = "FAA,clarity,context,q_work,bert_mean"
DEFAULT_GROUPS = "S1,S2,S3"
DEFAULT_ALPHA = 0.05


@dataclass
class TestResult:
    metric: str
    n_total: int
    group_ns: Dict[str, int]
    means: Dict[str, float]
    stds: Dict[str, float]
    anova_f: float
    anova_p: float
    eta_sq: float
    levene_p: float
    shapiro_ps: Dict[str, Optional[float]]
    faa_chi2_p: Optional[float]
    faa_chi2_stat: Optional[float]
    faa_chi2_dof: Optional[int]
    posthoc_text: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ANOVA and post-hoc tests for S1/S2/S3 metrics."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to input data file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name (if input is xlsx/xls).",
    )
    parser.add_argument(
        "--prompt-col",
        default=DEFAULT_PROMPT_COL,
        help=f"Column name that stores prompting strategy labels (default: {DEFAULT_PROMPT_COL}).",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=f"Comma-separated metric columns to analyze (default: {DEFAULT_METRICS}).",
    )
    parser.add_argument(
        "--groups",
        default=DEFAULT_GROUPS,
        help=f"Comma-separated group labels to compare (default: {DEFAULT_GROUPS}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Significance threshold (default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output Markdown report path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def read_input(path: str, sheet: Optional[str]) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported file extension: {ext}")


def normalize_prompt_label(raw: str) -> str:
    x = str(raw).strip().upper()
    mapping = {
        "STRUKTUR1": "S1",
        "STRUKTUR2": "S2",
        "STRUKTUR3": "S3",
        "S1": "S1",
        "SP": "S1",
        "STANDARD PROMPT (SP)": "S1",
        "STANDARD PROMPTING (SP)": "S1",
        "S2": "S2",
        "SP+COT": "S2",
        "SP + CHAIN-OF-THOUGHT (SP+COT)": "S2",
        "SP WITH HIDDEN CHAIN-OF-THOUGHT (SP+COT)": "S2",
        "S3": "S3",
        "SP+QC": "S3",
        "SP + QUALITY CONSTRAINTS (SP+QC)": "S3",
        "SP WITH QUALITY CONSTRAINTS (SP+QC)": "S3",
    }
    return mapping.get(x, str(raw).strip())


def parse_locale_number(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return np.nan

    # Remove non-breaking spaces / regular spaces
    s = s.replace("\u00a0", "").replace(" ", "")

    # Heuristics for locale-formatted numbers:
    # - "2.178,64" -> 2178.64
    # - "0,78" -> 0.78
    # - "1234.56" -> 1234.56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    # else keep as is

    try:
        return float(s)
    except Exception:
        return np.nan


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / math.sqrt(pooled)


def holm_correction(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    indexed = list(enumerate(pvals))
    indexed.sort(key=lambda z: z[1])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def run_metric_tests(
    df: pd.DataFrame,
    metric: str,
    prompt_col: str,
    groups: Sequence[str],
    alpha: float,
) -> TestResult:
    sub = df[[prompt_col, metric]].copy()
    sub[metric] = sub[metric].apply(parse_locale_number)
    sub = sub.dropna(subset=[metric, prompt_col])

    grouped: Dict[str, np.ndarray] = {}
    for g in groups:
        vals = sub.loc[sub[prompt_col] == g, metric].to_numpy(dtype=float)
        grouped[g] = vals

    if any(len(grouped[g]) == 0 for g in groups):
        missing = [g for g in groups if len(grouped[g]) == 0]
        raise ValueError(
            f"Metric '{metric}' has no data for groups: {', '.join(missing)}"
        )

    all_vals = [grouped[g] for g in groups]
    group_ns = {g: len(grouped[g]) for g in groups}
    means = {g: float(np.mean(grouped[g])) for g in groups}
    stds = {g: float(np.std(grouped[g], ddof=1)) if len(grouped[g]) > 1 else 0.0 for g in groups}

    f_stat, p_val = stats.f_oneway(*all_vals)

    overall = sub[metric].to_numpy(dtype=float)
    grand_mean = np.mean(overall)
    ss_between = sum(len(grouped[g]) * (means[g] - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for x in overall)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    try:
        _, levene_p = stats.levene(*all_vals, center="median")
    except Exception:
        levene_p = float("nan")

    shapiro_ps: Dict[str, Optional[float]] = {}
    for g in groups:
        arr = grouped[g]
        if len(arr) < 3:
            shapiro_ps[g] = None
        else:
            try:
                _, p_sh = stats.shapiro(arr)
                shapiro_ps[g] = float(p_sh)
            except Exception:
                shapiro_ps[g] = None

    faa_chi2_p = None
    faa_chi2_stat = None
    faa_chi2_dof = None

    is_binary = np.array_equal(np.unique(overall), np.array([0.0, 1.0])) or np.array_equal(
        np.unique(overall), np.array([0.0])
    ) or np.array_equal(np.unique(overall), np.array([1.0]))
    if metric.lower() == "faa" and is_binary:
        contingency = []
        for g in groups:
            arr = grouped[g]
            n0 = int(np.sum(arr == 0))
            n1 = int(np.sum(arr == 1))
            contingency.append([n0, n1])
        chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)
        faa_chi2_stat = float(chi2_stat)
        faa_chi2_p = float(chi2_p)
        faa_chi2_dof = int(chi2_dof)

    posthoc_text: List[str] = []
    if HAS_STATSMODELS:
        try:
            tukey = pairwise_tukeyhsd(
                endog=sub[metric].to_numpy(dtype=float),
                groups=sub[prompt_col].astype(str).to_numpy(),
                alpha=alpha,
            )
            posthoc_text.append("Tukey HSD:")
            for row in tukey.summary().data[1:]:
                g1, g2, meandiff, p_adj, low, high, reject = row
                posthoc_text.append(
                    f"- {g1} vs {g2}: mean_diff={float(meandiff):.4f}, "
                    f"p_adj={float(p_adj):.6f}, CI=[{float(low):.4f}, {float(high):.4f}], "
                    f"reject={bool(reject)}"
                )
        except Exception as ex:
            posthoc_text.append(f"Tukey HSD failed: {ex}")
    else:
        # Fallback: Welch t-test pairwise + Holm correction
        pairs: List[Tuple[str, str, float, float]] = []
        raw_ps: List[float] = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                a, b = grouped[g1], grouped[g2]
                t_stat, p_pair = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                d = cohen_d(a, b)
                pairs.append((g1, g2, float(t_stat), float(d)))
                raw_ps.append(float(p_pair))
        adj_ps = holm_correction(raw_ps)
        posthoc_text.append("Pairwise Welch t-test with Holm correction:")
        for (g1, g2, t_stat, d), p_raw, p_adj in zip(pairs, raw_ps, adj_ps):
            posthoc_text.append(
                f"- {g1} vs {g2}: t={t_stat:.4f}, p_raw={p_raw:.6f}, "
                f"p_holm={p_adj:.6f}, cohen_d={d:.4f}"
            )
        posthoc_text.append(
            "Note: Install statsmodels for Tukey HSD: pip install statsmodels"
        )

    return TestResult(
        metric=metric,
        n_total=len(sub),
        group_ns=group_ns,
        means=means,
        stds=stds,
        anova_f=float(f_stat),
        anova_p=float(p_val),
        eta_sq=eta_sq,
        levene_p=float(levene_p),
        shapiro_ps=shapiro_ps,
        faa_chi2_p=faa_chi2_p,
        faa_chi2_stat=faa_chi2_stat,
        faa_chi2_dof=faa_chi2_dof,
        posthoc_text=posthoc_text,
    )


def build_report(
    input_path: str,
    prompt_col: str,
    groups: Sequence[str],
    alpha: float,
    results: Sequence[TestResult],
) -> str:
    lines: List[str] = []
    lines.append("# Statistical Report: Prompting Strategies")
    lines.append("")
    lines.append(f"- Input file: `{input_path}`")
    lines.append(f"- Prompt column: `{prompt_col}`")
    lines.append(f"- Groups analyzed: `{', '.join(groups)}`")
    lines.append(f"- Alpha: `{alpha}`")
    lines.append("")

    for r in results:
        lines.append(f"## Metric: `{r.metric}`")
        lines.append("")
        lines.append(f"- N total: `{r.n_total}`")
        lines.append("- Group N/mean/std:")
        for g in groups:
            lines.append(
                f"  - `{g}`: n={r.group_ns[g]}, mean={r.means[g]:.4f}, std={r.stds[g]:.4f}"
            )
        lines.append(
            f"- One-way ANOVA: F={r.anova_f:.6f}, p={r.anova_p:.6g}, "
            f"eta_sq={r.eta_sq:.4f}"
        )
        lines.append(f"- Levene (homogeneity): p={r.levene_p:.6g}")
        shap = []
        for g in groups:
            val = r.shapiro_ps[g]
            if val is None:
                shap.append(f"{g}=NA(<3)")
            else:
                shap.append(f"{g}={val:.6g}")
        lines.append(f"- Shapiro-Wilk per group: {', '.join(shap)}")

        if r.faa_chi2_p is not None:
            lines.append(
                f"- FAA binary check (chi-square): chi2={r.faa_chi2_stat:.6f}, "
                f"dof={r.faa_chi2_dof}, p={r.faa_chi2_p:.6g}"
            )

        lines.append("- Post-hoc:")
        for t in r.posthoc_text:
            lines.append(f"  {t}")

        if r.anova_p < alpha:
            lines.append(
                f"- Conclusion: Significant difference detected (p < {alpha})."
            )
        else:
            lines.append(
                f"- Conclusion: No significant overall difference (p >= {alpha})."
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    if len(groups) < 2:
        print("Error: at least 2 groups are required.", file=sys.stderr)
        return 2

    df = read_input(args.input, args.sheet)
    if args.prompt_col not in df.columns:
        print(
            f"Error: prompt column '{args.prompt_col}' not found. "
            f"Columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    df = df.copy()
    # Some spreadsheets export merged-like prompt labels with blanks.
    df[args.prompt_col] = df[args.prompt_col].ffill()
    df[args.prompt_col] = df[args.prompt_col].apply(normalize_prompt_label)

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        print(
            f"Error: metric columns not found: {missing_metrics}. "
            f"Columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    results: List[TestResult] = []
    for metric in metrics:
        try:
            res = run_metric_tests(
                df=df,
                metric=metric,
                prompt_col=args.prompt_col,
                groups=groups,
                alpha=args.alpha,
            )
            results.append(res)
        except Exception as ex:
            print(f"[WARN] Skip metric '{metric}': {ex}", file=sys.stderr)

    if not results:
        print("No metric could be analyzed.", file=sys.stderr)
        return 1

    report = build_report(
        input_path=args.input,
        prompt_col=args.prompt_col,
        groups=groups,
        alpha=args.alpha,
        results=results,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[OK] Analysis complete. Report written to: {args.output}")
    print("")
    for r in results:
        sig = "significant" if r.anova_p < args.alpha else "not_significant"
        print(
            f"- {r.metric}: p={r.anova_p:.6g} ({sig}), "
            f"means={{{', '.join([f'{g}:{r.means[g]:.3f}' for g in groups])}}}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
