import numpy as np

from imgui_ophys.lessons.quality import model as M


def test_nu_of_shot_noise_depends_on_photons_not_sample_rate():
    rng = np.random.default_rng(0)
    phi0 = 1e6
    for fs in (30.0, 1000.0):
        per_sample = phi0 / fs
        dff = 100 * (rng.poisson(per_sample, 200_000) / per_sample - 1)
        assert abs(M.nu(dff, fs) / M.nu_shot(phi0) - 1) < 0.02


def test_matched_filter_puts_the_spike_sample_at_i():
    h = np.array([0.0, 1.0, 0.5, 0.25])  # one sample before the spike
    x = np.zeros(20)
    x[9:13] = h
    y = M.matched_filter(x, h, before=1)
    assert np.argmax(y) == 10 and np.isclose(y[10], h @ h)


def test_dprime_matches_white_noise_prediction():
    clean = M.make_clean("GCaMP6f", 30.0, 0.5)
    for phi0 in (1e3, 1e4):
        ds = [M.record(clean, phi0, seed=s).dprime for s in range(8)]
        predicted = M.dprime_white(M.record(clean, phi0).template, clean.fs, phi0)
        assert abs(np.mean(ds) / predicted - 1) < 0.15


def test_same_nu_much_lower_dprime_for_voltage():
    calcium = M.record(M.make_clean("GCaMP6f", 30.0, 0.5), 1e4)
    voltage = M.record(M.make_clean("ASAP3", 1000.0, 6.0), 1e4)
    assert abs(calcium.nu / voltage.nu - 1) < 0.15
    assert calcium.dprime > 5 * voltage.dprime


def test_spectrum_of_white_noise_is_flat_at_2_var_over_fs():
    x = np.random.default_rng(1).normal(0, 2.0, 400_000)
    f, p = M.spectrum(x, 1000.0)
    assert abs(p.mean() / (2 * 4.0 / 1000.0) - 1) < 0.02


def test_motion_shows_in_the_slow_band_not_in_nu():
    clean = M.make_clean("ASAP3", 1000.0, 6.0)
    still, moving = M.record(clean, 1e6), M.record(clean, 1e6, artifact=5.0)
    assert abs(moving.nu / still.nu - 1) < 0.05
    assert M.slow_excess(moving.dff - moving.truth, clean.fs, 1e6) > 4 * M.slow_excess(still.dff - still.truth, clean.fs, 1e6)


def test_lowpass_lowers_nu_but_not_dprime():
    clean = M.make_clean("GCaMP6f", 30.0, 0.5)
    raw, filtered = M.record(clean, 1e4), M.record(clean, 1e4, cutoff=5.0)
    assert filtered.nu < 0.5 * raw.nu
    assert abs(filtered.dprime / raw.dprime - 1) < 0.2


def test_survival_starts_at_the_largest_value_and_ends_at_all():
    v = np.arange(1.0, 101.0)
    x, frac = M.survival(v)
    assert x[0] == 100.0 and frac[0] == 0.01
    assert x[-1] == 1.0 and frac[-1] == 1.0


def test_lag1_of_white_noise_is_zero_and_of_ar1_its_coefficient():
    rng = np.random.default_rng(2)
    T = 20_000
    white = rng.standard_normal((T, 50))
    assert np.all(np.abs(M.lag1(white)) < 4 / np.sqrt(T))
    ar = np.zeros((T, 3))
    for t in range(1, T):
        ar[t] = [0.2, 0.6, 0.95] * ar[t - 1] + rng.standard_normal(3)
    assert np.allclose(M.lag1(ar), [0.2, 0.6, 0.95], atol=0.02)


def test_lowpass_runs_along_the_last_axis():
    x = np.random.default_rng(3).standard_normal((4, 500))
    assert np.allclose(M.lowpass(x, 30.0, 5.0), [M.lowpass(row, 30.0, 5.0) for row in x])


def test_r1_is_the_light_s_r1_times_its_share_of_the_variance():
    field = M.make_field("GCaMP6s", 30.0, 0.5)
    recorded, expected = M.count_field(field, 1e6)
    share = M.share_not_shot(recorded, expected)
    lit = share > 0.3
    assert lit.sum() > 20
    error = M.lag1(recorded)[lit] - (M.lag1(expected) * share)[lit]
    assert abs(error.mean()) < 0.01 and np.abs(error).mean() < 0.03


def test_silent_cell_shows_in_the_mean_but_not_in_r1_until_it_moves():
    field = M.make_field("GCaMP6f", 30.0, 0.5)
    x, y = M.CELLS[2][:2]
    center = y * M.SIDE + x
    null = 4 / np.sqrt(field.t.size)
    still, _ = M.count_field(field, 1e6)
    assert np.argmax(still.mean(0)) == center
    assert abs(M.lag1(still)[center]) < null
    moving, _ = M.count_field(field, 1e6, artifact=5.0)
    assert M.lag1(moving)[center] > 0.3


def test_lowpass_makes_background_noise_look_like_signal():
    field = M.make_field("GCaMP6f", 30.0, 0.5)
    raw, _ = M.count_field(field, 1e4)
    filtered, _ = M.count_field(field, 1e4, cutoff=5.0)
    assert abs(M.lag1(raw)[0]) < 4 / np.sqrt(field.t.size)
    assert M.lag1(filtered)[0] > 0.5
