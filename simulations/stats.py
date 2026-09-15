"""
stats.py -- The statistical machinery the article's Section 8.2 must describe.

THE SAMPLE-SIZE RULE (Reviewer 1, comment 6)
--------------------------------------------
The unit of independent replication is the SEED.  Every experiment in this package
runs `n_mc` Monte Carlo iterations inside a seed and *averages* them into one number
per seed.  Averaging reduces the variance of that seed's estimate; it does not create
independent observations.  Therefore

        n = number of seeds,      df = n - 1,

and never n_seeds x n_mc.  Every function below takes per-seed vectors, and
`describe()` records `n_monte_carlo_per_seed` separately so the distinction is visible
in the raw output files.

All arm-to-arm comparisons are PAIRED: the same seed list drives every arm (common
random numbers), so the difference vector is formed seed by seed.  Paired tests are
both more powerful here and the only correct treatment of the design.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats as sps

from config import CAMPAIGN


# ----------------------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------------------

@dataclass
class Summary:
    name: str
    n_seeds: int
    mean: float
    sd: float
    sem: float
    ci_low: float
    ci_high: float
    ci_halfwidth: float
    t_crit: float
    alpha: float
    median: float
    minimum: float
    maximum: float
    n_monte_carlo_per_seed: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self, unit: str = "", digits: int = 3) -> str:
        u = f" {unit}" if unit else ""
        return (f"{self.mean:.{digits}f}{u} "
                f"(95 % CI [{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}], "
                f"SD {self.sd:.{digits}f}, n = {self.n_seeds})")


def describe(values: Sequence[float], name: str = "", alpha: float | None = None,
             n_monte_carlo_per_seed: int | None = None) -> Summary:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    mean = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(sps.t.ppf(1.0 - alpha / 2.0, n - 1)) if n > 1 else float("nan")
    hw = tcrit * sem if n > 1 else 0.0
    return Summary(name=name, n_seeds=int(n), mean=mean, sd=sd, sem=float(sem),
                   ci_low=mean - hw, ci_high=mean + hw, ci_halfwidth=float(hw),
                   t_crit=tcrit, alpha=alpha,
                   median=float(np.median(x)) if n else float("nan"),
                   minimum=float(x.min()) if n else float("nan"),
                   maximum=float(x.max()) if n else float("nan"),
                   n_monte_carlo_per_seed=n_monte_carlo_per_seed)


# ----------------------------------------------------------------------------------
# Paired comparison
# ----------------------------------------------------------------------------------

def interpret_d(d: float) -> str:
    """Cohen's (1988) conventional benchmarks."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    if a < 1.2:
        return "large"
    return "very large"


@dataclass
class Comparison:
    metric: str
    arm_a: str
    arm_b: str
    n_seeds: int
    mean_a: float
    mean_b: float
    mean_diff: float           # a - b
    diff_sd: float
    diff_ci_low: float
    diff_ci_high: float
    relative_diff_pct: float
    t_stat: float
    df: int
    p_raw: float
    cohens_d: float
    d_interpretation: str
    wilcoxon_stat: float
    wilcoxon_p: float
    higher_is_better: bool

    def as_dict(self) -> dict:
        return asdict(self)


def paired_compare(a: Sequence[float], b: Sequence[float], metric: str,
                   arm_a: str, arm_b: str, alpha: float | None = None,
                   higher_is_better: bool = True) -> Comparison:
    """Paired t-test + Wilcoxon signed-rank + paired Cohen's d on the SAME seeds."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired comparison requires equal-length, seed-aligned vectors")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    d = x - y
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0

    if sd == 0.0:
        # Every seed produced exactly the same difference.  If that difference is
        # zero the arms are identical; if it is non-zero the separation is perfect
        # and the t statistic is unbounded.  Reporting p = 1 in the second case
        # would understate the evidence, so the two cases are distinguished.
        t_stat, p_raw = ((0.0, 1.0) if abs(md) < 1e-15 else (float("inf"), 0.0))
    else:
        t_stat, p_raw = sps.ttest_rel(x, y)
        t_stat, p_raw = float(t_stat), float(p_raw)

    tcrit = float(sps.t.ppf(1 - alpha / 2, n - 1)) if n > 1 else float("nan")
    # Cohen's d for paired samples: mean difference / SD of the differences
    dz = md / sd if sd > 0 else 0.0

    try:
        if np.allclose(d, 0.0):
            w_stat, w_p = 0.0, 1.0
        else:
            w = sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    base = abs(float(y.mean()))
    return Comparison(
        metric=metric, arm_a=arm_a, arm_b=arm_b, n_seeds=int(n),
        mean_a=float(x.mean()), mean_b=float(y.mean()), mean_diff=md,
        diff_sd=sd, diff_ci_low=md - tcrit * sem, diff_ci_high=md + tcrit * sem,
        relative_diff_pct=100.0 * md / base if base > 0 else float("nan"),
        t_stat=t_stat, df=int(n - 1), p_raw=p_raw,
        cohens_d=float(dz), d_interpretation=interpret_d(dz),
        wilcoxon_stat=w_stat, wilcoxon_p=w_p,
        higher_is_better=higher_is_better)


# ----------------------------------------------------------------------------------
# Multiple-comparison correction
# ----------------------------------------------------------------------------------

def bonferroni(p_values: Sequence[float], alpha: float | None = None) -> dict:
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    thresh = alpha / m if m else alpha
    return {
        "method": "Bonferroni",
        "alpha_family": alpha,
        "n_comparisons": int(m),
        "adjusted_threshold": float(thresh),
        "adjusted_p": [float(min(1.0, v * m)) for v in p],
        "significant": [bool(v <= thresh) for v in p],
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float | None = None) -> dict:
    """Step-down Holm procedure: uniformly more powerful than Bonferroni and it
    controls the same family-wise error rate."""
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.zeros(m)
    sig = np.zeros(m, dtype=bool)
    running = 0.0
    still = True
    thresholds = np.zeros(m)
    for rank, idx in enumerate(order):
        thresholds[idx] = alpha / (m - rank)
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
        if still and p[idx] <= alpha / (m - rank):
            sig[idx] = True
        else:
            still = False
    return {
        "method": "Holm-Bonferroni",
        "alpha_family": alpha,
        "n_comparisons": int(m),
        "per_comparison_threshold": [float(v) for v in thresholds],
        "adjusted_p": [float(v) for v in adj],
        "significant": [bool(v) for v in sig],
    }


# ----------------------------------------------------------------------------------
# Power analysis
# ----------------------------------------------------------------------------------

def paired_t_power(effect_size_d: float, n: int, alpha: float | None = None) -> float:
    """Achieved power of a two-sided paired t-test with effect size d (Cohen's d_z).

    Uses the exact non-central t distribution with non-centrality d*sqrt(n).
    """
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    if n < 2:
        return float("nan")
    df = n - 1
    nc = abs(effect_size_d) * np.sqrt(n)
    crit = sps.t.ppf(1 - alpha / 2, df)
    with np.errstate(all="ignore"):
        upper = 1.0 - sps.nct.cdf(crit, df, nc)
        lower = sps.nct.cdf(-crit, df, nc)
    power = upper + lower
    if not np.isfinite(power):
        # The non-central t CDF underflows for very large non-centrality parameters.
        # In that regime the power is indistinguishable from 1 (or from alpha at
        # nc = 0), so fall back to the normal approximation rather than emitting NaN.
        z = sps.norm.ppf(1 - alpha / 2)
        power = sps.norm.sf(z - nc) + sps.norm.cdf(-z - nc)
    return float(np.clip(power, 0.0, 1.0))


def minimum_detectable_effect(n: int, power: float = 0.80,
                              alpha: float | None = None) -> float:
    """Smallest Cohen's d_z detectable at the requested power (bisection on d)."""
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if paired_t_power(mid, n, alpha) < power:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def required_n(effect_size_d: float, power: float = 0.80,
               alpha: float | None = None, n_max: int = 5000) -> int:
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    for n in range(3, n_max):
        if paired_t_power(effect_size_d, n, alpha) >= power:
            return n
    return n_max


# ----------------------------------------------------------------------------------
# Family assembly
# ----------------------------------------------------------------------------------

def build_family(comparisons: Sequence[Comparison], alpha: float | None = None,
                 family_name: str = "") -> dict:
    """Apply both corrections to a family of comparisons and annotate each row."""
    alpha = CAMPAIGN.alpha if alpha is None else alpha
    ps = [c.p_raw for c in comparisons]
    bon = bonferroni(ps, alpha)
    holm = holm_bonferroni(ps, alpha)
    rows = []
    for i, c in enumerate(comparisons):
        r = c.as_dict()
        r["bonferroni_threshold"] = bon["adjusted_threshold"]
        r["bonferroni_adjusted_p"] = bon["adjusted_p"][i]
        r["significant_bonferroni"] = bon["significant"][i]
        r["holm_threshold"] = holm["per_comparison_threshold"][i]
        r["holm_adjusted_p"] = holm["adjusted_p"][i]
        r["significant_holm"] = holm["significant"][i]
        r["achieved_power"] = paired_t_power(c.cohens_d, c.n_seeds, alpha)
        r["mde_at_n_power80"] = minimum_detectable_effect(c.n_seeds, 0.80, alpha)
        r["verdict"] = _verdict(r)
        rows.append(r)
    return {
        "family": family_name,
        "alpha_family": alpha,
        "n_comparisons": len(comparisons),
        "bonferroni_threshold": bon["adjusted_threshold"],
        "rows": rows,
    }


def _verdict(row: dict) -> str:
    if row["significant_bonferroni"]:
        return (f"significant after Bonferroni correction "
                f"(p = {row['p_raw']:.2e} <= {row['bonferroni_threshold']:.2e}); "
                f"effect size d = {row['cohens_d']:.2f} ({row['d_interpretation']})")
    if row["significant_holm"]:
        return ("significant under Holm-Bonferroni but NOT under Bonferroni; "
                "report as borderline")
    if row["p_raw"] <= row["alpha_family"] if "alpha_family" in row else False:
        return "nominally significant but NOT after correction: report as not significant"
    return "NOT significant"


def summarise_arms(per_seed: Dict[str, Sequence[float]], metric: str,
                   n_mc: int | None = None) -> Dict[str, dict]:
    return {arm: describe(v, name=f"{metric}/{arm}",
                          n_monte_carlo_per_seed=n_mc).as_dict()
            for arm, v in per_seed.items()}


def all_pairwise(per_seed: Dict[str, Sequence[float]], metric: str,
                 reference: str | None = None,
                 higher_is_better: bool = True) -> List[Comparison]:
    """If `reference` is given, compare every other arm against it; otherwise compare
    every unordered pair."""
    arms = list(per_seed.keys())
    out: List[Comparison] = []
    if reference is not None:
        for arm in arms:
            if arm == reference:
                continue
            out.append(paired_compare(per_seed[reference], per_seed[arm], metric,
                                      reference, arm,
                                      higher_is_better=higher_is_better))
    else:
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                out.append(paired_compare(per_seed[arms[i]], per_seed[arms[j]],
                                          metric, arms[i], arms[j],
                                          higher_is_better=higher_is_better))
    return out


def normality_check(values: Sequence[float]) -> dict:
    """Shapiro-Wilk on the paired differences: reported so the reader can judge how
    much weight to put on the t-test versus the Wilcoxon result."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.allclose(x, x[0]):
        return {"test": "Shapiro-Wilk", "W": float("nan"), "p": float("nan"),
                "normal_at_005": None}
    w, p = sps.shapiro(x)
    return {"test": "Shapiro-Wilk", "W": float(w), "p": float(p),
            "normal_at_005": bool(p > 0.05)}
