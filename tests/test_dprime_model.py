import math

import numpy as np

from imgui_ophys import sim
from imgui_ophys.lessons.dprime import model as M


def test_dprime_of_two_gaussians_is_their_separation_in_sds():
    rng = np.random.default_rng(0)
    bottom = rng.normal(0.0, 1.0, 20000)
    top = rng.normal(3.0, 1.0, 20000)
    assert abs(M.dprime(top, bottom) - 3.0) < 0.05
    assert math.isnan(M.dprime(np.zeros(0), bottom))
    assert math.isnan(M.dprime(np.ones(5), np.ones(5)))


def test_accuracy_is_phi_of_half_dprime():
    assert abs(M.accuracy(3.0) - 0.933) < 0.001
    assert abs(M.accuracy(0.0) - 0.5) < 1e-9
    assert math.isnan(M.accuracy(math.nan))


def test_windows_sit_before_the_spike_and_at_the_peak():
    fs, frames, peak = 10.0, 3, 2
    f = np.arange(100, dtype=float)
    s = M.score(f, np.array([5.0]), fs, peak, frames, before=1.0)
    assert s.n == 1 and s.centers[0] == 50
    assert np.array_equal(s.bottom[0], [47, 48, 49])
    assert np.array_equal(s.top[0], [52, 53, 54])


def test_isolation_drops_spikes_with_a_neighbour_before_them():
    spikes = np.array([1.0, 1.3, 5.0, 9.0, 9.1])
    assert np.array_equal(M.isolated(spikes, before=1.0, after=0.2), [1.0, 5.0])


def test_brighter_cell_scores_higher():
    dim = M.score_params(sim.TraceParams(brightness=2.0, duration=120, seed=3), frames=6)
    bright = M.score_params(sim.TraceParams(brightness=100.0, duration=120, seed=3), frames=6)
    assert dim.n > 5 and bright.n > 5
    assert bright.d > 2 * dim.d


def test_sweep_and_indicators_return_one_score_per_setting():
    p = sim.TraceParams(duration=60, seed=1)
    d = M.sweep(p, "brightness", np.array([1.0, 100.0]), frames=6)
    assert d.shape == (2,) and d[1] > d[0]
    assert [n for n, _ in M.by_indicator(p, frames=6)] == [i.name for i in M.CALCIUM]


def test_detector_finds_peaks_above_threshold_and_keeps_them_apart():
    f = np.zeros(100)
    f[[20, 22, 60]] = [5.0, 4.0, 5.0]
    f += np.sin(np.arange(100)) * 0.1
    assert np.array_equal(M.detect(f, threshold=3.0, min_gap=5), [20, 60])
    assert np.array_equal(M.detect(f, threshold=3.0, min_gap=1), [20, 22, 60])


def test_detector_on_a_silent_cell_still_produces_events():
    p = sim.TraceParams(rate=0.0, duration=60, seed=2)
    tr = sim.make_trace(p)
    dff = sim.to_dff(tr.recorded, float(np.percentile(tr.recorded, 20)))
    picked = M.detected_spike_times(dff, p.fs, peak=1, threshold=3.0, min_gap=7)
    assert picked.size > 0
