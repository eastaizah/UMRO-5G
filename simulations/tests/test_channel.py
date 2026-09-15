import numpy as np
import pytest

from channel import (SNR_THRESHOLDS_DB, CQI_SE, se_from_snr_db, cqi_from_snr_db,
                     shannon_se, path_loss_db, noise_power_dbm, CellChannel,
                     rayleigh_se_matrix)
from config import CHANNEL, NUM_MU1, CQI_TABLE_256QAM


def test_cqi_table_is_the_3gpp_256qam_table():
    assert len(CQI_TABLE_256QAM) == 15
    ses = [r[3] for r in CQI_TABLE_256QAM]
    assert ses == sorted(ses)
    assert ses[0] == pytest.approx(0.1523)
    assert ses[-1] == pytest.approx(7.4063)
    # modulation orders must be one of QPSK/16QAM/64QAM/256QAM
    assert set(r[1] for r in CQI_TABLE_256QAM) <= {2, 4, 6, 8}
    # spectral efficiency must equal modulation order x code rate
    for _, q, cr, se in CQI_TABLE_256QAM:
        assert q * cr / 1024.0 == pytest.approx(se, rel=2e-3)


def test_snr_thresholds_are_monotone():
    assert np.all(np.diff(SNR_THRESHOLDS_DB) > 0)


def test_se_selection_is_monotone_and_bounded():
    snr = np.linspace(-30, 40, 400)
    se = se_from_snr_db(snr)
    assert np.all(np.diff(se) >= -1e-12)
    assert se.max() == pytest.approx(CQI_SE[-1])
    assert se_from_snr_db(-40.0) == 0.0


def test_selected_se_never_exceeds_the_shannon_reference():
    snr = np.linspace(-20, 35, 300)
    assert np.all(se_from_snr_db(snr) <= shannon_se(snr) + 1e-9)


def test_cqi_index_matches_selected_se():
    snr = np.linspace(-15, 35, 200)
    cqi = cqi_from_snr_db(snr)
    se = se_from_snr_db(snr)
    for c, s in zip(cqi, se):
        if c == 0:
            assert s == 0.0
        else:
            assert s == pytest.approx(CQI_SE[int(c) - 1])


def test_path_loss_matches_3gpp_formula():
    d = 500.0
    expected = CHANNEL.pl_intercept_db + CHANNEL.pl_slope_db * np.log10(d / 1000.0)
    assert path_loss_db(d) == pytest.approx(expected)
    # monotone increasing with distance
    assert path_loss_db(1000.0) > path_loss_db(100.0)
    # clamped below the minimum distance
    assert path_loss_db(1.0) == pytest.approx(path_loss_db(CHANNEL.min_distance_m))


def test_noise_power_matches_ktb_plus_nf():
    bw = 360e3
    expected = -174.0 + 10 * np.log10(bw) + CHANNEL.noise_figure_db
    assert noise_power_dbm(bw) == pytest.approx(expected)


def test_block_fading_is_constant_inside_the_coherence_time():
    ch = CellChannel(np.random.default_rng(1), 20, NUM_MU1)
    n = ch.coherence_tti
    a = ch.step(0).copy()
    b = ch.step(n - 1).copy()
    assert np.array_equal(a, b)
    c = ch.step(n).copy()
    assert not np.array_equal(a, c)


def test_channel_is_reproducible_from_the_seed():
    def trace(seed):
        ch = CellChannel(np.random.default_rng(seed), 15, NUM_MU1)
        return np.array([ch.step(t).sum() for t in range(200)])
    assert np.allclose(trace(7), trace(7))
    assert not np.allclose(trace(7), trace(8))


def test_rayleigh_matrix_shape_and_block_structure():
    rng = np.random.default_rng(3)
    mean = np.full(6, 12.0)
    m = rayleigh_se_matrix(rng, mean, 47, 10)
    assert m.shape == (47, 6)
    assert np.array_equal(m[0], m[9])
    assert not np.array_equal(m[0], m[10])


def test_mean_snr_decreases_with_distance():
    rng = np.random.default_rng(11)
    ch = CellChannel(rng, 400, NUM_MU1)
    near = ch.mean_snr_db[ch.distance_m < np.median(ch.distance_m)].mean()
    far = ch.mean_snr_db[ch.distance_m >= np.median(ch.distance_m)].mean()
    assert near > far
