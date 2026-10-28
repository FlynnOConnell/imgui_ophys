"""
Draw every lesson for a few frames on hello_imgui's null backend: no window or GPU. Skips when the backend cannot
start here.
"""

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
    assert [x.key for x in app.lessons] == ["motion", "nmf", "traces", "dprime", "svd"]


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


def test_traces_every_indicator_part_and_denoiser_renders():
    from imgui_ophys import sim

    app = App(lesson="traces")
    steps = []
    for ind in sim.INDICATORS:
        steps.append(lambda a, i=ind: (setattr(a.lesson.params, "indicator", i.name), setattr(a.lesson.params, "fs", i.fs),
                                       setattr(a.lesson.params, "duration", i.duration), a.lesson.make()))
    for name in ("bleaching", "neuropil", "shot", "read"):
        steps.append(lambda a, n=name: a.lesson.toggle(n))
    for base in sim.BASELINES:
        steps.append(lambda a, b=base: (setattr(a.lesson, "baseline", b), a.lesson.refilter()))
    for method in sim.DENOISERS:
        steps.append(lambda a, m=method: (setattr(a.lesson, "method", m), a.lesson.refilter()))
    run_frames(app, steps)
    assert not any(app.lesson.trace.on.values())
