"""The Denoising page's two inputs: the gevisim cell with its noise knob, and an exported MUnit in either layout."""

import numpy as np
import pytest

pytest.importorskip("vnoiser")
pytest.importorskip("gevisim")

from imgui_ophys.lessons.denoise import signals as S  # noqa: E402


def test_the_cell_is_a_jedi_soma_with_spikes():
    sig = S.make_signal(S.SignalParams(duration_s=4.0))
    assert sig.fs == S.REFERENCE_FS and sig.n == round(4.0 * S.REFERENCE_FS)
    assert np.all(np.isfinite(sig.x)) and sig.truth.shape == sig.x.shape
    px = sig.pixels
    assert px.pixels.shape[0] > 50 and px.edges == [(0, px.pixels.shape[0])] and px.names == ["soma"]
    assert np.all(sig.events_s["AP"] < 4.0) and sig.events_s["AP"].size > 5
    assert np.all(px.pixels >= 0) and px.clean_pixels.mean() == pytest.approx(px.pixels.mean(), rel=0.05)


def test_noise_scales_the_shot_noise_and_zero_is_the_clean_cell():
    quiet = S.make_signal(S.SignalParams(noise=0.0, duration_s=4.0))
    np.testing.assert_allclose(quiet.x, quiet.truth)
    one = S.make_signal(S.SignalParams(noise=1.0, duration_s=4.0))
    three = S.make_signal(S.SignalParams(noise=3.0, duration_s=4.0))
    sd = lambda sig: float((sig.pixels.pixels - sig.pixels.clean_pixels).std())
    assert sd(three) == pytest.approx(3 * sd(one), rel=0.1)
    assert np.corrcoef(one.x[1000:], one.truth[1000:])[0, 1] > np.corrcoef(three.x[1000:], three.truth[1000:])[0, 1]


def test_a_new_seed_is_another_draw_of_the_same_cell():
    a = S.make_signal(S.SignalParams(seed=1, duration_s=2.0))
    b = S.make_signal(S.SignalParams(seed=2, duration_s=2.0))
    np.testing.assert_allclose(a.pixels.clean_f, b.pixels.clean_f)
    np.testing.assert_allclose(a.truth / a.truth.std(), b.truth / b.truth.std())  # truth is in the noisy trace's SD units
    assert not np.allclose(a.x, b.x)


def test_the_trace_is_what_vnoiser_preprocess_makes():
    from vnoiser.preprocess import domain_zscore

    sig = S.make_signal(S.SignalParams(duration_s=6.0))
    px = sig.pixels
    z = domain_zscore({0: px.pixels.astype(float).mean(axis=0)}, {"soma": [0]}).z[0]
    np.testing.assert_allclose(sig.x, z, atol=1e-5)
    assert px.n_startup == 1000 and np.all(px.dfof[:1000] == px.dfof[1000:].mean())


def test_munit_pixel_export_reads_the_chosen_lines_and_window(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "MSession_0_MUnit_1.npz"
    rois = {f"roi_{i}": rng.integers(1150, 1250, size=(3000, w), dtype=np.uint16) for i, w in enumerate((19, 20, 10))}
    np.savez(path, fs=1587.3, unit="MSession_0/MUnit_1", channel=0, offset=-1053.0, source="x.mesc", **rois)
    sig = S.make_signal(S.SignalParams(source="munit", scan_path=str(path), rois=(0, 2), start_s=0.5, duration_s=1.0))
    i0 = round(0.5 * 1587.3)
    expected = np.concatenate([rois["roi_0"][i0:i0 + 1587].T, rois["roi_2"][i0:i0 + 1587].T]).astype(float)
    np.testing.assert_allclose(sig.pixels.pixels, expected)
    assert sig.fs == pytest.approx(1587.3) and sig.pixels.edges == [(0, 19), (19, 29)] and sig.truth is None
    assert sig.pixels.names == ["line 0", "line 2"]


def test_munit_per_roi_export_is_read_as_one_pixel_rois(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "MSession_0_MUnit_7.npz"
    F = rng.normal(900, 30, size=(2, 3, 4000)).astype(np.float32)
    np.savez(path, t=np.arange(4000) / 813.0, fs=813.0, F=F, dff=F, channels=np.array(["green", "red"]))
    info = S.scan_info(str(path))
    assert info["widths"] == {0: 1, 1: 1, 2: 1} and info["unit"] == "MSession_0_MUnit_7"
    sig = S.make_signal(S.SignalParams(source="munit", scan_path=str(path), rois=(1,), duration_s=2.0))
    np.testing.assert_allclose(sig.pixels.f, F[0, 1][: round(2.0 * 813.0)])


def test_find_scans_sees_both_layouts(tmp_path):
    (tmp_path / "a" / "linescans").mkdir(parents=True)
    (tmp_path / "b" / "c").mkdir(parents=True)
    for f in ("a/linescans/MSession_0_MUnit_3.npz", "b/c/MSession_0_MUnit_1.npz", "b/other.npz"):
        (tmp_path / f).write_bytes(b"")
    found = S.find_scans(str(tmp_path))
    assert [p.split(str(tmp_path) + "/")[1] for p in found] == ["a/linescans/MSession_0_MUnit_3.npz", "b/c/MSession_0_MUnit_1.npz"]


def test_a_missing_file_is_a_message():
    with pytest.raises(ValueError, match="No file"):
        S.make_signal(S.SignalParams(source="munit", scan_path="/nope/unit.npz"))
