import numpy as np
import pytest
from scipy import stats as sps

import stats as S


def test_describe_matches_the_textbook_t_interval():
    x = [10.0, 11.0, 9.5, 10.5, 12.0, 9.0, 10.2, 11.3]
    d = S.describe(x)
    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    hw = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    assert d.n_seeds == n
    assert d.mean == pytest.approx(mean)
    assert d.sd == pytest.approx(sd)
    assert d.ci_low == pytest.approx(mean - hw)
    assert d.ci_high == pytest.approx(mean + hw)


def test_degrees_of_freedom_are_seeds_minus_one_not_iterations():
    """Reviewer 1 comment 6: recording n_mc must NOT change the inference."""
    x = np.random.default_rng(0).normal(5, 1, 30)
    a = S.describe(x, n_monte_carlo_per_seed=1)
    b = S.describe(x, n_monte_carlo_per_seed=1000)
    assert a.n_seeds == b.n_seeds == 30
    assert a.t_crit == b.t_crit == pytest.approx(sps.t.ppf(0.975, 29))
    assert a.ci_halfwidth == pytest.approx(b.ci_halfwidth)


def test_paired_compare_matches_scipy():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 30)
    b = a - rng.normal(0.8, 1.0, 30)
    c = S.paired_compare(a, b, "m", "A", "B")
    t, p = sps.ttest_rel(a, b)
    assert c.t_stat == pytest.approx(t)
    assert c.p_raw == pytest.approx(p)
    assert c.df == 29
    d = a - b
    assert c.cohens_d == pytest.approx(d.mean() / d.std(ddof=1))


def test_paired_compare_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        S.paired_compare([1, 2, 3], [1, 2], "m", "A", "B")


def test_identical_arms_are_not_significant():
    x = np.random.default_rng(2).normal(0, 1, 30)
    c = S.paired_compare(x, x.copy(), "m", "A", "B")
    assert c.p_raw == pytest.approx(1.0)
    assert c.cohens_d == 0.0


def test_constant_non_zero_difference_is_perfect_separation():
    x = np.full(30, 1.0)
    c = S.paired_compare(x + 2.0, x, "m", "A", "B")
    assert c.p_raw == 0.0
    assert not np.isfinite(c.t_stat)


def test_bonferroni_threshold_and_flags():
    p = [0.001, 0.02, 0.04, 0.5]
    b = S.bonferroni(p, 0.05)
    assert b["adjusted_threshold"] == pytest.approx(0.0125)
    assert b["significant"] == [True, False, False, False]
    assert b["adjusted_p"][0] == pytest.approx(0.004)


def test_holm_is_never_more_conservative_than_bonferroni():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = list(rng.random(6) ** 3)
        bon = S.bonferroni(p, 0.05)
        holm = S.holm_bonferroni(p, 0.05)
        for i in range(len(p)):
            assert holm["adjusted_p"][i] <= bon["adjusted_p"][i] + 1e-12
            if bon["significant"][i]:
                assert holm["significant"][i]


def test_holm_adjusted_p_is_monotone_in_the_sorted_order():
    p = [0.001, 0.008, 0.03, 0.04, 0.9]
    holm = S.holm_bonferroni(p, 0.05)
    adj = [holm["adjusted_p"][i] for i in np.argsort(p)]
    assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1))


def test_power_is_increasing_in_effect_size_and_in_n():
    assert S.paired_t_power(0.2, 30) < S.paired_t_power(0.8, 30)
    assert S.paired_t_power(0.5, 10) < S.paired_t_power(0.5, 60)
    assert 0.0 <= S.paired_t_power(0.0, 30) <= 0.06


def test_minimum_detectable_effect_round_trips():
    for n in (10, 30, 60):
        mde = S.minimum_detectable_effect(n, 0.80)
        assert S.paired_t_power(mde, n) == pytest.approx(0.80, abs=5e-3)


def test_required_n_round_trips():
    n = S.required_n(0.5, 0.80)
    assert S.paired_t_power(0.5, n) >= 0.80
    assert S.paired_t_power(0.5, n - 1) < 0.80


def test_effect_size_interpretation_bands():
    assert S.interpret_d(0.1) == "negligible"
    assert S.interpret_d(0.3) == "small"
    assert S.interpret_d(0.6) == "medium"
    assert S.interpret_d(1.0) == "large"
    assert S.interpret_d(2.0) == "very large"
    assert S.interpret_d(-1.0) == "large"


def test_build_family_annotates_every_row():
    rng = np.random.default_rng(4)
    base = rng.normal(10, 1, 30)
    arms = {"ref": base,
            "a": base - 1.5 + rng.normal(0, 0.4, 30),
            "b": base - 0.02 + rng.normal(0, 0.9, 30),
            "c": base - 3.0 + rng.normal(0, 0.5, 30)}
    comps = S.all_pairwise(arms, "metric", reference="ref")
    fam = S.build_family(comps, family_name="test")
    assert fam["n_comparisons"] == 3
    assert fam["bonferroni_threshold"] == pytest.approx(0.05 / 3)
    for row in fam["rows"]:
        for key in ("bonferroni_threshold", "holm_adjusted_p", "achieved_power",
                    "mde_at_n_power80", "verdict", "cohens_d", "wilcoxon_p"):
            assert key in row
    by = {r["arm_b"]: r for r in fam["rows"]}
    assert by["c"]["significant_bonferroni"] is True
    assert by["b"]["significant_bonferroni"] is False


def test_all_pairwise_without_reference_covers_every_pair():
    arms = {"x": [1, 2, 3, 4], "y": [2, 3, 4, 5], "z": [0, 1, 2, 3]}
    comps = S.all_pairwise(arms, "m")
    assert len(comps) == 3


def test_normality_check_reports_a_verdict():
    rng = np.random.default_rng(5)
    n = S.normality_check(rng.normal(0, 1, 40))
    assert n["normal_at_005"] in (True, False)
    assert 0.0 <= n["p"] <= 1.0
