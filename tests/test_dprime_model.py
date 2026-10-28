import math

import numpy as np
import pytest

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


def test_noise_rows_land_on_their_targets_and_end_with_the_clean_trace():
    p = sim.TraceParams(brightness=5000.0, neuropil=0.0, read_noise=0.0, bleach=0.0, duration=40, seed=3)
    tr = sim.make_trace(p)
    dff = sim.to_dff(tr.recorded, sim.baseline(tr.recorded, "running window", int(5 * p.fs)))
    peak, before = M.peak_frames(sim.BY_NAME[p.indicator], p.fs), M.quiet_before(sim.BY_NAME[p.indicator])
    rows = M.noise_rows(dff, tr.spikes, p.fs, peak, 6, before, (1.0, 3.0))
    assert [round(d, 1) for d, _ in rows[:2]] == [1.0, 3.0]
    assert rows[-1][0] > 3.0 and np.array_equal(rows[-1][1], dff)
    with pytest.raises(ValueError):
        M.noise_rows(dff, tr.spikes, p.fs, peak, 6, before, (50.0,))
