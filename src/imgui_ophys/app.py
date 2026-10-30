"""
The window `imgui-ophys` opens: the pipeline's stages in a rail on the left, the current stage's subsections under it
and its parameters below, and the stage's page on the right. With imgui_debugger installed (the debug extra), a Debug
menu opens a live inspector on the current stage and a style editor.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from imgui_bundle import hello_imgui, icons_fontawesome_6 as fa, imgui, immapp, implot
from imgui_bundle import portable_file_dialogs as pfd

from imgui_ophys import __version__, config, sim
from imgui_ophys import plotting as P
from imgui_ophys.lesson import Lesson
from imgui_ophys.lessons import LESSONS
from imgui_ophys.widgets import help_mark, section

try:
    from imgui_debugger import DebugTools
except ImportError:  # optional: pip install imgui_ophys[debug]
    DebugTools = None

TITLE_WINDOW = "ophys lab"
SIZE_WINDOW = (1500, 1000)
WIDTH_RAIL_EM = 24
IMPLOT_PADDINGS = (
    "plot_padding", "label_padding", "legend_padding", "legend_inner_padding", "legend_spacing",
    "mouse_pos_padding", "annotation_padding", "major_tick_len", "minor_tick_len",
)


class App:
    """One instance of every lesson, the one on screen, the UI scale and the debug panels."""

    def __init__(self, lesson: str = "", scale: float = 1.0, recording: Optional[sim.Recording] = None):
        self.lessons: list[Lesson] = [cls() for cls in LESSONS]
        keys = [x.key for x in self.lessons]
        self.index = keys.index(lesson) if lesson in keys else 0
        self.recording = recording or sim.Recording()
        self.recording_version = 0
        self.file_dialog: Optional[pfd.open_file] = None
        for x in self.lessons:
            x.goto = self.goto
            x.touch = self.touch_recording
        self.scale = config.clamp(scale)
        self.scale_applied = 1.0
        self.scale_edit: Optional[float] = None
        self.tools = DebugTools.default(target=self.lesson, title="Lesson state") if DebugTools else None

    @property
    def lesson(self) -> Lesson:
        return self.lessons[self.index]

    def select(self, index: int) -> None:
        self.index = index
        debugger = self.tools.debugger() if self.tools else None
        if debugger is not None:
            debugger.set_target(self.lesson)

    def goto(self, key: str) -> None:
        """Open the stage with this key."""
        keys = [x.key for x in self.lessons]
        if key in keys:
            self.select(keys.index(key))

    def touch_recording(self) -> None:
        """The shared Recording changed: every page that uses it rebuilds when it is next drawn."""
        self.recording_version += 1

    def sync_recording(self) -> None:
        lesson = self.lesson
        if lesson.uses_recording and lesson.recording_version != self.recording_version:
            lesson.recording_version = self.recording_version
            lesson.set_recording(self.recording)

    # ------------------------------------------------------------------
    # scale
    # ------------------------------------------------------------------

    def apply_scale(self) -> None:
        """Font, imgui and ImPlot sizes to self.scale. Called every frame; does work only when the scale moved."""
        if self.scale == self.scale_applied:
            return
        ratio = self.scale / self.scale_applied
        style = imgui.get_style()
        style.font_scale_main = self.scale
        style.scale_all_sizes(ratio)
        ps = implot.get_style()
        for name in IMPLOT_PADDINGS:
            v = getattr(ps, name)
            setattr(ps, name, imgui.ImVec2(v.x * ratio, v.y * ratio))
        ps.plot_min_size = imgui.ImVec2(ps.plot_min_size.x * ratio, ps.plot_min_size.y * ratio)
        P.SCALE = self.scale
        self.scale_applied = self.scale

    def set_scale(self, scale: float) -> None:
        self.scale = config.clamp(scale)
        config.save({"ui_scale": self.scale})

    def scale_keys(self) -> None:
        """Ctrl + / Ctrl - / Ctrl 0, as in a browser."""
        if not imgui.get_io().key_ctrl:
            return
        if imgui.is_key_pressed(imgui.Key.equal) or imgui.is_key_pressed(imgui.Key.keypad_add):
            self.set_scale(self.scale * 1.1)
        elif imgui.is_key_pressed(imgui.Key.minus) or imgui.is_key_pressed(imgui.Key.keypad_subtract):
            self.set_scale(self.scale / 1.1)
        elif imgui.is_key_pressed(imgui.Key._0) or imgui.is_key_pressed(imgui.Key.keypad0):
            self.set_scale(config.detect_scale())

    # ------------------------------------------------------------------
    # frame
    # ------------------------------------------------------------------

    def gui(self) -> None:
        self.scale_keys()
        self.apply_scale()
        self.poll_file_dialog()
        self.sync_recording()
        lesson = self.lesson
        lesson.update(imgui.get_io().delta_time)
        hello_imgui.get_runner_params().fps_idling.enable_idling = not lesson.animating()
        em = imgui.get_font_size()
        rail = min(max(0.3 * imgui.get_content_region_avail().x, 17 * em), WIDTH_RAIL_EM * em)
        if imgui.begin_child("rail", imgui.ImVec2(rail, 0), imgui.ChildFlags_.borders):
            self.draw_rail()
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("page", imgui.ImVec2(0, 0)):
            self.draw_page()
        imgui.end_child()
        if self.tools:
            self.tools.render()

    def draw_menus(self) -> None:
        if imgui.begin_menu("Stages"):
            for i, lesson in enumerate(self.lessons):
                if imgui.menu_item(lesson.title, "", i == self.index)[0]:
                    self.select(i)
            imgui.end_menu()
        if self.tools:
            self.tools.draw_menu()

    def draw_rail(self) -> None:
        imgui.text_colored(P.ACCENT, TITLE_WINDOW)
        imgui.same_line()
        imgui.text_disabled(f"v{__version__}")
        imgui.spacing()
        em = imgui.get_font_size()
        group = ""
        for i, lesson in enumerate(self.lessons):
            if lesson.group != group:
                group = lesson.group
                imgui.spacing()
                imgui.separator_text(group)
            label = lesson.title if group else f"{i + 1}  {lesson.title}"
            if imgui.selectable(f"{label}##stage{i}", i == self.index)[0]:
                self.select(i)
            if i == self.index:
                imgui.indent(1.4 * em)
                for j, (icon, title) in enumerate(lesson.outline()):
                    imgui.push_style_color(imgui.Col_.text, imgui.get_style_color_vec4(imgui.Col_.text_disabled))
                    if imgui.selectable(f"{icon}  {title}##{lesson.key}{j}", False)[0]:
                        lesson.jump = j
                    imgui.pop_style_color()
                imgui.unindent(1.4 * em)
        imgui.spacing()
        imgui.separator()
        if imgui.begin_child("controls", imgui.ImVec2(0, -imgui.get_frame_height_with_spacing() * 1.5)):
            em = imgui.get_font_size()
            # leave about 9 em for each widget's label
            imgui.push_item_width(max(6 * em, imgui.get_content_region_avail().x - 9 * em))
            if self.lesson.uses_recording:
                self.draw_recording()
            self.lesson.controls()
            imgui.pop_item_width()
        imgui.end_child()
        imgui.separator()
        imgui.set_next_item_width(8 * imgui.get_font_size())
        # applied on release: rescaling mid-drag would move the slider out from under the mouse
        shown = self.scale_edit if self.scale_edit is not None else self.scale
        changed, value = imgui.slider_float("text size", shown, config.SCALE_MIN, config.SCALE_MAX, "%.2fx")
        if changed:
            self.scale_edit = value
        if self.scale_edit is not None and not imgui.is_item_active():
            self.set_scale(self.scale_edit)
            self.scale_edit = None
        help_mark("Also Ctrl + / Ctrl -. Ctrl 0 resets to the detected scale. Saved in ~/.imgui_ophys/settings.json.")

    def draw_recording(self) -> None:
        """Calcium or voltage, the indicator, the SNR, and a recorded file: the one block every trace page shares."""
        rec = self.recording
        section(f"{fa.ICON_FA_DNA}  Recording")
        for i, kind in enumerate(sim.KINDS):
            if i:
                imgui.same_line()
            if imgui.radio_button(kind, rec.kind == kind) and rec.kind != kind:
                rec.set_kind(kind)
                self.touch_recording()
        names = [x.name for x in sim.INDICATORS if x.kind == rec.kind]
        changed, k = imgui.combo("indicator", names.index(rec.indicator), names)
        if changed:
            rec.indicator = names[k]
            self.touch_recording()
        help_mark(sim.BY_NAME[rec.indicator].note)
        shown = rec.snr if rec.snr > 0 else rec.default_snr()
        changed, value = imgui.slider_float("SNR", shown, 0.3, 30.0, "%.1f", imgui.SliderFlags_.logarithmic)
        if changed:
            rec.snr = value
        if imgui.is_item_deactivated_after_edit():
            self.touch_recording()
        help_mark("One spike's peak dF/F over the noise SD of dF/F per frame, at rest. It sets the cell's brightness.")
        if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}  Open a trace"):
            self.file_dialog = pfd.open_file("A recorded trace", str(Path.home()), ["Traces", "*.npy *.npz *.csv *.txt *.tsv", "All files", "*"])
        help_mark("A .npy, .npz (roi_<i> arrays and fs, as the line scan export writes), .csv or .txt, one trace per column. The Denoising page runs on it; the other pages keep the simulated cell.")
        if rec.path:
            imgui.same_line()
            if imgui.small_button(fa.ICON_FA_XMARK):
                rec.path = ""
                self.touch_recording()
            imgui.text_disabled(Path(rec.path).name)

    def poll_file_dialog(self) -> None:
        if self.file_dialog is not None and self.file_dialog.ready():
            chosen = self.file_dialog.result()
            self.file_dialog = None
            if chosen:
                self.recording.path = chosen[0]
                self.touch_recording()

    def draw_page(self) -> None:
        lesson = self.lesson
        imgui.text(lesson.title)
        imgui.same_line()
        imgui.text_disabled(f"  {lesson.summary}")
        imgui.separator()
        if imgui.begin_child("##page_body"):
            lesson.page()
        imgui.end_child()


def runner_params(app: App) -> hello_imgui.RunnerParams:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = TITLE_WINDOW
    params.app_window_params.window_geometry.size = tuple(int(v * min(app.scale, 1.5)) for v in SIZE_WINDOW)
    params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    params.imgui_window_params.show_menu_bar = True
    params.imgui_window_params.show_menu_app = False
    params.imgui_window_params.show_menu_view = False
    params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
    params.fps_idling.fps_idle = 20
    params.ini_filename = "imgui_ophys.ini"
    params.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
    params.callbacks.show_gui = app.gui
    params.callbacks.show_menus = app.draw_menus
    params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
    return params


def run(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="imgui-ophys", description="Interactive lessons on optical physiology.")
    parser.add_argument("--app", default="", choices=["", *(cls.key for cls in LESSONS)], help="stage to open on")
    parser.add_argument("--scale", type=float, default=0.0, help="UI scale, e.g. 2 on a 2x HiDPI panel (default: saved, then detected)")
    parser.add_argument("--file", default="", help="a recorded trace (.npy, .npz, .csv) for the Denoising page")
    args = parser.parse_args(argv)
    app = App(lesson=args.app, scale=config.ui_scale(args.scale), recording=sim.Recording(path=args.file))
    immapp.run(runner_params(app), immapp.AddOnsParams(with_implot=True))


if __name__ == "__main__":
    run()
