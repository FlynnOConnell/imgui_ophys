import numpy as np

from imgui_ophys import sim


def test_shot_noise_is_sqrt_of_photons():
    p = sim.TraceParams(rate=0.0, read=False, background=False, bleaching=False, duration=200, seed=2)
    tr = sim.make_trace(p)
    photons = p.brightness * p.pixels
    assert abs(tr.recorded.mean() - photons) < 0.02 * photons
    assert abs(tr.recorded.std() - np.sqrt(photons)) < 0.1 * np.sqrt(photons)


def test_parts_add_up_and_switch_off():
    tr = sim.make_trace(sim.TraceParams())
    assert np.allclose(tr.recorded, sum(tr.parts.values()))
    for name in tr.on:
        tr.on[name] = False
    assert np.allclose(tr.recorded, tr.parts["signal"])


def test_calcium_spike_reaches_its_amplitude():
    p = sim.TraceParams(rate=0.0, seed=0)
    ind = sim.BY_NAME[p.indicator]
    k = sim.calcium_kernel(ind.rise, ind.decay, 1e-3)
    assert k.max() == 1.0 and k[0] == 0.0


def test_voltage_indicator_dims_on_a_spike_and_frames_blur_it():
    fast = sim.make_trace(sim.TraceParams(indicator="ASAP3", fs=2000, duration=2, rate=5, seed=3))
    slow = sim.make_trace(sim.TraceParams(indicator="ASAP3", fs=200, duration=2, rate=5, seed=3))
    assert fast.dff.min() < 0
    assert fast.dff.min() < slow.dff.min()  # a 5 ms exposure averages most of a 1 ms spike away


def test_smoothing_lowers_noise_and_peaks():
    x = np.zeros(200)
    x[100] = 1.0
    for method in ("moving average", "gaussian", "exponential"):
        y = sim.denoise(x, method, 4)
        assert y.size == x.size and y.max() < 0.5
    assert np.array_equal(sim.denoise(x, "none", 4), x)


def test_neuropil_is_steady_from_the_first_frame():
    for ind in ("GCaMP6f", "ASAP3"):
        i = sim.BY_NAME[ind]
        tr = sim.make_trace(sim.TraceParams(indicator=ind, fs=i.fs, duration=i.duration, rate=5))
        n = tr.parts["neuropil"]
        q = n.size // 4
        assert abs(n[:q].mean() - n[-q:].mean()) < 0.1 * n.mean()


def test_running_baseline_follows_bleaching():
    tr = sim.make_trace(sim.TraceParams(bleach=0.5, rate=0.2, seed=4))
    f = tr.recorded
    whole = sim.baseline(f, "whole recording", 0)
    running = sim.baseline(f, "running window", int(5 * 30))
    assert np.ptp(whole) == 0 and running[:50].mean() > running[-50:].mean()
    drift = lambda f0: abs(np.median(sim.to_dff(f, f0)[:150]) - np.median(sim.to_dff(f, f0)[-150:]))
    assert drift(running) < drift(whole) / 3
