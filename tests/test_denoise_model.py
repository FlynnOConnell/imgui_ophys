"""The Denoising page's model gives what vnoiser's own calls give on the same z, and counts hits as stated."""

import numpy as np
import pytest

from imgui_ophys import sim
from imgui_ophys.lessons.denoise import model as M

vnoiser = pytest.importorskip("vnoiser")


@pytest.fixture(scope="module")
def cell():
    """The simulated JEDI3sub cell: 10 s at 1 kHz, spikes known."""
    p = sim.Recording(indicator="JEDI3sub").params()
    trace = sim.make_trace(p)
    return trace.recorded, p.fs, trace.spikes


@pytest.fixture(scope="module")
def run(cell):
    f, fs, spikes = cell
    return M.run(f=f, fs=fs, spikes=spikes, negative=True)


def test_zscore_reproduces_domain_zscore_on_a_one_roi_domain(cell):
    f, fs, _ = cell
    _, dfof, _, z = M.zscore(f, fs, negative=True)
    expected = vnoiser.domain_zscore({0: f}, {"soma": [0]}, cfg=M.dfof_config(fs, negative=True))
    np.testing.assert_allclose(z, expected.z[0], atol=1e-12)
    np.testing.assert_allclose(dfof, expected.dfof_raw[0], atol=1e-12)


def test_denoised_matches_the_package_at_the_same_scales(run, cell):
    f, fs, _ = cell
    denoiser = vnoiser.Denoiser.upstream(fs, freq_scales=M.scales_at(fs))
    denoised, _ = denoiser.run(run.z)
    np.testing.assert_allclose(run.denoised, denoised.real, atol=1e-10)
    np.testing.assert_allclose(run.baseline, denoiser.lp_dfof_, atol=1e-12)
    np.testing.assert_allclose(run.rescaled, denoiser.rescaled_signal_.real, atol=1e-10)


def test_bands_are_the_package_rows_with_their_masks_blurred(run, cell):
    f, fs, _ = cell
    denoiser = vnoiser.Denoiser.upstream(fs, freq_scales=M.scales_at(fs))
    denoiser.run(run.z)
    reduced, thres = denoiser.reduced_, denoiser.threshold_result_
    assert len(run.bands) == reduced["reduced_coeff"].shape[0] > 0
    assert [b.lo_hz for b in run.bands] == sorted(b.lo_hz for b in run.bands)
    for band in run.bands:
        row = int(np.flatnonzero(reduced["reduced_freqs"][:, 0] == band.lo_hz)[0])
        scaled = band.trace / np.sqrt(reduced["reduced_scale"][row])
        np.testing.assert_allclose(scaled * band.mask, thres["all_scaled"][row].real, atol=1e-10)
        np.testing.assert_allclose(band.masked, thres["all_scaled"][row].real, atol=1e-12)
        assert np.all((band.mask > 0) & (band.mask <= 1.0 + 1e-9))
    total = sum(band.masked for band in run.bands)
    np.testing.assert_allclose(total, run.rescaled, atol=1e-10)


def test_peaks_are_detect_peaks_on_final_trace(run, cell):
    f, fs, _ = cell
    final = vnoiser.final_trace(run.rescaled, run.baseline)
    np.testing.assert_allclose(run.final, final)
    np.testing.assert_array_equal(run.peaks, vnoiser.detect_peaks(final, float(np.floor(fs)), M.spike_config(fs)))
    assert M.spike_config(fs).bp[1] == pytest.approx(400.0)
    assert M.spike_config(600.0).bp[1] == pytest.approx(0.95 * 300.0)
    assert M.spike_config(2 * M.REFERENCE_FS).distance_samples == 6


def test_scales_keep_their_length_in_seconds():
    base = M.scales_at(M.REFERENCE_FS)
    np.testing.assert_allclose(base[[0, -1]], [1.0, 1000.0])
    np.testing.assert_allclose(M.scales_at(2 * M.REFERENCE_FS), 2 * base)
    assert M.scales_at(30.0)[0] == 1.0


def test_hits_count_spikes_with_a_peak_within_tolerance():
    spikes = np.array([1.0, 2.0, 3.0, 4.0])
    peaks = np.array([1.004, 2.02, 3.0, 5.0, 5.1])
    assert M.peak_hits(spikes, peaks, tolerance_s=0.01) == (2, 4, 3)
    assert M.peak_hits(spikes, np.array([]), tolerance_s=0.01) == (0, 4, 0)
    assert M.peak_hits(np.array([]), peaks, tolerance_s=0.01) == (0, 0, 5)


def test_run_counts_hits_on_the_simulated_cell(run, cell):
    _, fs, spikes = cell
    assert run.hits == M.peak_hits(spikes, run.peaks / fs)
    assert run.amp.shape == (M.N_SCALES, run.z.size) and run.amp.dtype == np.float32
    assert run.image.shape[0] == M.N_SCALES and run.image.shape[1] <= M.IMAGE_COLUMNS


def test_read_trace_reads_npy_columns_and_npz_rois(tmp_path):
    rng = np.random.default_rng(0)
    columns = rng.normal(size=(500, 3))
    np.save(tmp_path / "a.npy", columns)
    f, fs = M.read_trace(str(tmp_path / "a.npy"))
    assert fs is None and np.array_equal(f, columns[:, 0])
    (tmp_path / "a.csv").write_text("t,f\n" + "\n".join(f"{i},{v}" for i, v in enumerate(columns[:, 1])))
    f, fs = M.read_trace(str(tmp_path / "a.csv"))
    assert fs is None and np.allclose(f, np.arange(500))
    rois = {"roi_0": rng.normal(size=(500, 10)), "roi_1": rng.normal(size=(500, 20))}
    np.savez(tmp_path / "scan.npz", fs=1587.3, unit="MUnit_30", **rois)
    f, fs = M.read_trace(str(tmp_path / "scan.npz"))
    assert fs == pytest.approx(1587.3)
    np.testing.assert_allclose(f, np.concatenate([rois["roi_0"], rois["roi_1"]], axis=1).mean(axis=1))
