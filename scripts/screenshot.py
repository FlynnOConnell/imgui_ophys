"""
One PNG per page for the README, drawn in a hidden window: the rail with its controls and the top of the page.

    .venv/bin/python scripts/screenshot.py [--out docs] [--scale 1.3] [--size 1800x1100] [--pages motion nmf ...]
                                           [--indicator JEDI3sub] [--snr 3] [--suffix -voltage]

The pages that follow the rail's Recording are drawn for the indicator and SNR given, and the files take the suffix.

Frames come off the GL front buffer after each swap (hello_imgui's own screenshot gives only the last frame). Needs
PyOpenGL and pillow in the environment.
"""

import argparse
from pathlib import Path

import numpy as np
from imgui_bundle import hello_imgui, imgui, immapp
from OpenGL import GL
from PIL import Image

from imgui_ophys import sim
from imgui_ophys.app import App, runner_params

SETTLE = 6  # frames drawn after a page opens before its capture, so layout and autofit have landed
SECTION = {"nmf": 2, "traces": 1}  # the subsection a page is scrolled to; the first elsewhere


class Shooter:
    def __init__(self, app: App, out: Path, pages: list, suffix: str = ""):
        self.app, self.out = app, out
        keys = [x.key for x in app.lessons]
        self.plan = []
        for key in pages:
            index = keys.index(key)
            self.plan.append((lambda i=index: app.select(i), None))
            self.plan.append((lambda k=key: setattr(app.lesson, "jump", SECTION.get(k, 0)), None))
            self.plan += [(None, None)] * SETTLE
            self.plan.append((None, f"{key}{suffix}.png"))
        self.frame = 0
        self.pending = None

    def gui(self) -> None:
        if self.frame >= len(self.plan):
            hello_imgui.get_runner_params().app_shall_exit = True
            return
        action, capture = self.plan[self.frame]
        if action:
            action()
        self.app.gui()
        self.pending = capture
        self.frame += 1

    def grab(self) -> None:
        if self.pending is None:
            return
        w, h = imgui.get_io().display_size
        fb = hello_imgui.final_app_window_screenshot_framebuffer_scale()
        W, H = int(w * fb), int(h * fb)
        GL.glReadBuffer(GL.GL_FRONT)
        data = GL.glReadPixels(0, 0, W, H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        Image.fromarray(np.frombuffer(data, np.uint8).reshape(H, W, 3)[::-1]).save(self.out / self.pending)
        print("wrote", self.out / self.pending)
        self.pending = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs"))
    ap.add_argument("--scale", type=float, default=1.3)
    ap.add_argument("--size", default="1800x1100")
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--indicator", default="GCaMP6f")
    ap.add_argument("--snr", type=float, default=0.0, help="0: the indicator's opening SNR")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    app = App(scale=args.scale, recording=sim.Recording(indicator=args.indicator, snr=args.snr))
    shooter = Shooter(app, out, args.pages or [x.key for x in app.lessons], args.suffix)
    params = runner_params(app)
    params.callbacks.show_gui = shooter.gui
    params.callbacks.after_swap = shooter.grab
    params.app_window_params.hidden = True
    params.app_window_params.window_geometry.size = tuple(int(v) for v in args.size.split("x"))
    params.app_window_params.restore_previous_geometry = False
    params.ini_folder_type = hello_imgui.IniFolderType.temp_folder
    params.fps_idling.enable_idling = False
    immapp.run(params, immapp.AddOnsParams(with_implot=True))


if __name__ == "__main__":
    main()
