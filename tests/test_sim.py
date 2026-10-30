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
    fast = sim.make_trace(sim.TraceParams(indicator="JEDI3sub", fs=2000, duration=2, rate=5, seed=3))
    slow = sim.make_trace(sim.TraceParams(indicator="JEDI3sub", fs=200, duration=2, rate=5, seed=3))
    assert fast.dff.min() < 0
    assert fast.dff.min() < slow.dff.min()  # a 5 ms exposure averages most of a 1 ms spike away
    bright = sim.make_trace(sim.TraceParams(indicator="ASAP6.3", fs=1000, duration=2, rate=5, seed=3))
    assert bright.dff.max() > 0 and np.allclose(sim.upright(bright.dff, sim.BY_NAME["ASAP6.3"]), bright.dff)
    assert np.allclose(sim.upright(fast.dff, sim.BY_NAME["JEDI3sub"]), -fast.dff)


def test_ar1_matches_the_recurrence():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000)
    a = np.exp(-1 / 200)
    y, acc = np.empty_like(x), 0.5
    for i, v in enumerate(x):
        acc = a * acc + v
        y[i] = acc
    assert np.allclose(sim.ar1(x, a, start=0.5), y, atol=1e-9)


def test_snr_slider_round_trips_through_brightness():
    for name in ("GCaMP6f", "JEDI3sub", "ASAP6.3"):
        ind = sim.BY_NAME[name]
        peak = sim.spike_peak(ind, ind.fs)
        for snr in (0.5, 3.0, 20.0):
            brightness = sim.brightness_for_snr(peak, snr, 50, 1.5, 0.3)
            assert abs(sim.snr_of(peak, brightness, 50, 1.5, 0.3) - snr) < 1e-9
    rec = sim.Recording(indicator="GCaMP6f")
    assert abs(rec.params().brightness - sim.BRIGHTNESS["calcium"]) < 1e-9
    rec.set_kind("voltage")
    assert rec.kind == "voltage" and abs(rec.params().brightness - sim.BRIGHTNESS["voltage"]) < 1e-9


def test_snr_formula_matches_the_simulated_noise():
    p = sim.Recording(indicator="GCaMP6f").params(duration=120.0, rate=0.0, bleach=0.0)
    tr = sim.make_trace(p)
    measured = np.std((tr.parts["shot"] + tr.parts["read"]) / tr.baseline)
    assert abs(measured - sim.noise_sd(p.brightness, p.pixels, p.read_noise, p.neuropil)) < 0.05 * measured


def test_smoothing_lowers_noise_and_peaks():
    x = np.zeros(200)
    x[100] = 1.0
    for method in ("moving average", "gaussian", "exponential"):
        y = sim.denoise(x, method, 4)
        assert y.size == x.size and y.max() < 0.5
    assert np.array_equal(sim.denoise(x, "none", 4), x)


def test_neuropil_is_steady_from_the_first_frame():
    for ind in ("GCaMP6f", "JEDI3sub"):
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
