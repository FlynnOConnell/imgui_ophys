"""
Draw every lesson for a few frames on hello_imgui's null backend: no window or GPU. Skips when the backend cannot
start here.
"""

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui, immapp

from imgui_ophys.app import App, runner_params
from imgui_ophys.lessons.nmf.view import METHODS, VIEWS, NmfLesson

FRAMES = 6


def enable_textures() -> None:
    # the null renderer uploads no font atlas; imgui 1.92 asserts in NewFrame unless the backend claims textures
    imgui.get_io().backend_flags |= imgui.BackendFlags_.renderer_has_textures


def run_frames(app: App, steps) -> None:
    """Call app.gui each frame; steps[i](app) runs before frame i * FRAMES. Exits after the last."""
    state = {"frame": 0}

    def frame() -> None:
        i, f = divmod(state["frame"], FRAMES)
        if i >= len(steps):
            hello_imgui.get_runner_params().app_shall_exit = True
            return
        if f == 0:
            steps[i](app)
        app.gui()
        state["frame"] += 1

    params = runner_params(app)
    params.platform_backend_type = hello_imgui.PlatformBackendType.null
    params.renderer_backend_type = hello_imgui.RendererBackendType.null
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.callbacks.post_init = enable_textures
    params.callbacks.show_gui = frame
    try:
        immapp.run(params, immapp.AddOnsParams(with_implot=True))
    except Exception as e:
        if "backend" in str(e).lower():
            pytest.skip(f"null backend unavailable: {e}")
        raise


def test_every_stage_and_subsection_renders():
    app = App()
    steps = []
    for i, lesson in enumerate(app.lessons):
        steps.append(lambda a, i=i: a.select(i))
        for j in range(len(lesson.outline())):
            steps.append(lambda a, j=j: setattr(a.lesson, "jump", j))
    run_frames(app, steps)
    assert [x.key for x in app.lessons] == ["motion", "nmf", "traces", "dprime", "denoise", "svd"]


def test_nmf_every_method_and_view_renders():
    app = App(lesson="nmf")
    lesson = app.lesson
    assert isinstance(lesson, NmfLesson)
    steps = []
    for mode in ("none", *(m[0] for m in METHODS)):
        for view in range(len(VIEWS)):
            steps.append(lambda a, m=mode, v=view: (a.lesson.run(m), setattr(a.lesson, "view", v)))
    steps.append(lambda a: setattr(a.lesson, "k", 6) or a.lesson.run("pca"))
    steps.append(lambda a: [a.lesson.guide.walk.advance() for _ in range(9)])
    run_frames(app, steps)
    assert lesson.mode == "pca" and lesson.A.shape[1] == 6
    assert lesson.guide.walk.sweep == 1


def test_traces_every_indicator_and_part_renders():
    from imgui_ophys import sim

    app = App(lesson="traces")
    steps = []
    for ind in sim.INDICATORS:
        steps.append(lambda a, i=ind: a.lesson.set_recording(sim.Recording(indicator=i.name)))
    for name in ("bleaching", "neuropil", "shot", "read"):
        steps.append(lambda a, n=name: a.lesson.toggle(n))
    steps.append(lambda a: a.lesson.set_recording(sim.Recording(indicator="GCaMP6f", snr=1.0)))
    run_frames(app, steps)
    assert not any(app.lesson.trace.on.values())
    assert app.lesson.params.brightness < sim.BRIGHTNESS["calcium"]


def test_dprime_every_kind_and_preset_renders():
    from imgui_ophys import sim
    from imgui_ophys.lessons.dprime.view import PRESETS

    app = App(lesson="dprime")
    steps = [lambda a: a.lesson.set_recording(sim.Recording(indicator="JEDI3sub"))]
    for j in range(len(app.lesson.outline())):
        steps.append(lambda a, j=j: setattr(a.lesson, "jump", j))
    steps.append(lambda a: a.lesson.set_recording(sim.Recording(indicator="GCaMP6f")))
    run_frames(app, steps)
    assert app.lesson.indicator.kind == "calcium"
    assert app.lesson.sections[1][1] == "Spike +/- 0.8 s"
    assert len(PRESETS) == 4


def test_denoise_every_section_on_both_sources(tmp_path):
    pytest.importorskip("vnoiser")
    pytest.importorskip("gevisim")
    rng = np.random.default_rng(0)
    scan = tmp_path / "MSession_0_MUnit_1.npz"
    np.savez(scan, fs=1587.3, unit="MSession_0/MUnit_1", roi_0=rng.integers(1150, 1250, size=(4000, 10), dtype=np.uint16))
    app = App(lesson="denoise")
    lesson = app.lesson
    lesson.settings.signal.duration_s = 2.0
    steps = [lambda a: a.lesson.wait()]
    for j in range(len(lesson.outline())):
        steps.append(lambda a, j=j: setattr(a.lesson, "jump", j))
    steps.append(lambda a: setattr(a.lesson, "heat_mode", 3))
    steps.append(lambda a: (setattr(a.lesson.t0, "value", 0.5), setattr(a.lesson.t1, "value", 0.7)))

    def munit(a):
        sp = a.lesson.settings.signal
        sp.source, sp.scan_path, sp.rois, sp.duration_s = "munit", str(scan), (0,), 2.0
        a.lesson.wait()

    steps.append(munit)
    for j in range(len(lesson.outline())):
        steps.append(lambda a, j=j: setattr(a.lesson, "jump", j))
    run_frames(app, steps)
    assert lesson.run.error == "" and lesson.run.signal.pixels.names == ["line 0"]
    assert lesson.run.peaks is not None
