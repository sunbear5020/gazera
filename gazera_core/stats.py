"""Statistics for condition / group comparisons.

Design choices, all reported alongside the numbers so a reader can check them:

* normality is tested per group (Shapiro-Wilk, n >= 3); small groups are
  treated as non-normal because the test has almost no power below n = 8;
* the parametric test is used only when every group looks normal, otherwise the
  rank-based alternative (Mann-Whitney U / Wilcoxon signed-rank);
* effect sizes are reported for both families (Cohen's d / rank-biserial r);
* p-values can be Holm-corrected across dependent variables, which is what most
  multi-metric eye-tracking comparisons need.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

__all__ = [
    "analyze_comparisons",
    "compare_many_groups",
    "compare_paired",
    "compare_two_groups",
    "describe_groups",
    "holm_correction",
    "normality_table",
]

_MIN_NORMAL_N = 8


def _clean(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def describe_groups(df: pd.DataFrame, dv: str, group: str = "condition") -> pd.DataFrame:
    """n, mean, sd, se, median, IQR and 95% CI per group."""
    rows = []
    if df is None or df.empty or dv not in df:
        return pd.DataFrame()
    grouped = df.groupby(group, dropna=False, sort=True) if group in df else [("all", df)]
    for name, sub in grouped:
        values = _clean(sub[dv])
        n = len(values)
        sd = float(values.std(ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 0 else math.nan
        ci = (math.nan, math.nan)
        if n > 1:
            half = float(sps.t.ppf(0.975, n - 1)) * se
            ci = (float(values.mean() - half), float(values.mean() + half))
        rows.append(
            {
                "group": str(name),
                "n": n,
                "mean": float(values.mean()) if n else math.nan,
                "sd": sd,
                "se": se,
                "median": float(np.median(values)) if n else math.nan,
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)) if n else math.nan,
                "min": float(values.min()) if n else math.nan,
                "max": float(values.max()) if n else math.nan,
                "ci95_low": ci[0],
                "ci95_high": ci[1],
            }
        )
    return pd.DataFrame(rows)


def normality_table(df: pd.DataFrame, dv: str, group: str = "condition") -> pd.DataFrame:
    """Shapiro-Wilk per group plus the decision used by the pipeline."""
    rows = []
    grouped = df.groupby(group, dropna=False, sort=True) if group in df else [("all", df)]
    for name, sub in grouped:
        values = _clean(sub[dv])
        n = len(values)
        w = p = math.nan
        normal = False
        if n >= 3:
            try:
                w, p = sps.shapiro(values)
                normal = bool(p >= 0.05) and n >= _MIN_NORMAL_N
            except Exception:  # pragma: no cover - constant input
                w = p = math.nan
                normal = False
        rows.append({"group": str(name), "n": n, "W": float(w), "p": float(p), "normal": normal})
    return pd.DataFrame(rows)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return math.nan
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    u = sps.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray([1.0 if v != v else float(v) for v in pvalues])
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return [float(v) for v in adjusted]


def compare_two_groups(
    df: pd.DataFrame,
    dv: str,
    group: str = "condition",
    *,
    alpha: float = 0.05,
    force: str | None = None,
) -> dict[str, Any]:
    """Compare a dependent variable between exactly two groups.

    ``force`` may be ``"parametric"`` or ``"nonparametric"`` to override the
    automatic normality-based choice.
    """
    groups = [g for g, sub in df.groupby(group, dropna=False, sort=True) if len(_clean(sub[dv])) >= 2]
    if len(groups) != 2:
        return {
            "metric": dv,
            "test": "not run",
            "note": f"needs exactly 2 groups with n>=2, found {len(groups)}",
            "p": math.nan,
            "effect_size": math.nan,
        }
    a = _clean(df.loc[df[group] == groups[0], dv])
    b = _clean(df.loc[df[group] == groups[1], dv])
    norm = normality_table(df[df[group].isin(groups)], dv, group)
    if force == "parametric":
        parametric = True
    elif force == "nonparametric":
        parametric = False
    else:
        parametric = bool(norm["normal"].all()) if len(norm) else False

    result: dict[str, Any] = {
        "metric": dv,
        "group_a": str(groups[0]),
        "group_b": str(groups[1]),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "normality": {row.group: {"W": row.W, "p": row.p, "normal": bool(row.normal)} for row in norm.itertuples()},
    }
    if parametric:
        t, p = sps.ttest_ind(a, b, equal_var=False)
        d = _cohens_d(a, b)
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        dfree = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
            (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
        )
        half = float(sps.t.ppf(0.975, dfree)) * se
        result.update(
            {
                "test": "Welch t-test",
                "statistic": float(t),
                "df": float(dfree),
                "p": float(p),
                "effect_size": float(d),
                "effect_size_name": "Cohen's d",
                "ci95_diff": [float(a.mean() - b.mean() - half), float(a.mean() - b.mean() + half)],
            }
        )
    else:
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        r = _rank_biserial(a, b)
        result.update(
            {
                "test": "Mann-Whitney U",
                "statistic": float(u),
                "p": float(p),
                "effect_size": float(r),
                "effect_size_name": "rank-biserial r",
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
            }
        )
    result["alpha"] = alpha
    result["significant"] = bool(result["p"] < alpha) if result["p"] == result["p"] else False
    return result


def compare_paired(
    df: pd.DataFrame,
    dv: str,
    within: str = "condition",
    subject: str = "subject",
    *,
    force: str | None = None,
) -> dict[str, Any]:
    """Paired comparison of two repeated conditions (same participants)."""
    if within not in df or subject not in df:
        return {"metric": dv, "test": "not run", "note": f"needs '{within}' and '{subject}' columns"}
    wide = df.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean")
    if wide.shape[1] != 2:
        return {"metric": dv, "test": "not run", "note": f"needs exactly 2 levels in '{within}'"}
    wide = wide.dropna()
    if len(wide) < 3:
        return {"metric": dv, "test": "not run", "note": "needs at least 3 complete pairs"}
    a = wide.iloc[:, 0].to_numpy(float)
    b = wide.iloc[:, 1].to_numpy(float)
    diff = a - b
    try:
        w_stat, p_norm = sps.shapiro(diff)
        normal = bool(p_norm >= 0.05) and len(diff) >= _MIN_NORMAL_N
    except Exception:  # pragma: no cover
        normal = False
    parametric = (force != "nonparametric") and (normal or force == "parametric")
    if parametric:
        t, p = sps.ttest_rel(a, b)
        statistic, test = (float(t), "Paired t-test")
        d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
        effect, effect_name = d, "Cohen's dz"
    else:
        stat, p = sps.wilcoxon(a, b, zero_method="wilcox") if np.any(diff != 0) else (0.0, 1.0)
        statistic, test = (float(stat), "Wilcoxon signed-rank")
        nz = diff[diff != 0]
        ranks = sps.rankdata(np.abs(nz))
        effect = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
        effect_name = "matched rank-biserial r"
    return {
        "metric": dv,
        "test": test,
        "condition_a": str(wide.columns[0]),
        "condition_b": str(wide.columns[1]),
        "n_pairs": int(len(wide)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "statistic": statistic,
        "p": float(p),
        "effect_size": effect,
        "effect_size_name": effect_name,
        "significant": bool(p < 0.05),
    }


def compare_many_groups(df: pd.DataFrame, dv: str, group: str = "condition") -> dict[str, Any]:
    """One-way ANOVA with a Kruskal-Wallis fallback for k > 2 groups."""
    groups = [(name, _clean(sub[dv])) for name, sub in df.groupby(group, dropna=False, sort=True)]
    groups = [(n, v) for n, v in groups if len(v) >= 2]
    if len(groups) < 2:
        return {"metric": dv, "test": "not run", "note": "needs at least 2 groups with n>=2"}
    norm = normality_table(df, dv, group)
    parametric = bool(norm["normal"].all()) if len(norm) else False
    if parametric:
        stat, p = sps.f_oneway(*[v for _, v in groups])
        test = "One-way ANOVA"
    else:
        stat, p = sps.kruskal(*[v for _, v in groups])
        test = "Kruskal-Wallis"
    return {
        "metric": dv,
        "test": test,
        "groups": [n for n, _ in groups],
        "statistic": float(stat),
        "p": float(p),
        "significant": bool(p < 0.05),
    }


def analyze_comparisons(
    subject_df: pd.DataFrame,
    dvs: Sequence[str] | None = None,
    *,
    group: str = "condition",
    paired: str | None = None,
    subject: str = "subject",
    correction: str | None = "holm",
) -> pd.DataFrame:
    """Run the right test for every DV and return one tidy results table.

    ``paired`` names a within-subject factor (usually ``condition``); when the
    same participants appear in every level the paired test is used.
    """
    if subject_df is None or subject_df.empty:
        return pd.DataFrame()
    dvs = list(dvs or [c for c in subject_df.columns if subject_df[c].dtype.kind in "fi" and not c.startswith("aoi_x")])
    dvs = [d for d in dvs if d in subject_df.columns and "condition" not in d]
    rows = []
    for dv in dvs:
        try:
            if group not in subject_df or subject_df[group].nunique(dropna=True) < 2:
                continue
            levels = subject_df[group].nunique(dropna=True)
            if levels == 2:
                use_paired = False
                if paired and paired in subject_df and subject in subject_df:
                    use_paired = subject_df[subject].nunique(dropna=True) >= 2
                result = (
                    compare_paired(subject_df, dv, within=paired, subject=subject)
                    if use_paired
                    else compare_two_groups(subject_df, dv, group)
                )
            else:
                result = compare_many_groups(subject_df, dv, group)
            result["metric"] = dv
            rows.append(result)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({"metric": dv, "test": "error", "note": str(exc), "p": math.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if correction == "holm" and "p" in out:
        out["p_adjusted"] = holm_correction(out["p"].tolist())
        out["significant_adjusted"] = [
            bool(p < 0.05) if p == p else False for p in out["p_adjusted"]
        ]
    return out
