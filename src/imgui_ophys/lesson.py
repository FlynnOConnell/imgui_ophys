"""
What every stage gives the window: a name for the rail, its subsections (listed under it in the rail and drawn in
order on the page), the parameters drawn under the rail, and per-frame updates.
"""

from typing import Callable, ClassVar, Optional

from imgui_bundle import icons_fontawesome_6 as fa
from imgui_bundle import imgui

from imgui_ophys import sim

REFERENCES = (fa.ICON_FA_BOOK, "References")


class Lesson:
    """
    One stage of the pipeline. Subclasses set `key`, `title`, `summary` and `sections`, and draw subsection i in
    `section_<i>()`; `controls` draws parameters under the rail, `update` runs once per frame before drawing.
    """

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    summary: ClassVar[str] = ""  # one line under the title on the page
    sections: ClassVar[tuple[tuple[str, str], ...]] = ()  # (icon, title) per subsection
    group: ClassVar[str] = ""  # "" for a pipeline stage; else a header the rail lists it under, after the stages
    references: ClassVar[tuple[tuple[str, str], ...]] = ()  # (citation, url), listed at the bottom of the page
    uses_recording: ClassVar[bool] = False  # True: the page shows the rail's Recording and follows its changes
    goto: Optional[Callable[[str], None]] = None  # set by the window: open another stage by its key
    touch: Optional[Callable[[], None]] = None  # set by the window: the page changed the shared Recording itself
    recording: sim.Recording = sim.Recording()  # the window's, shared; replaced by set_recording
    recording_version: int = -1  # the window's counter when set_recording last ran
    jump: Optional[int] = None  # a subsection to scroll to on the next frame

    def update(self, dt: float) -> None:
        """Advance by dt seconds of wall time."""

    def controls(self) -> None:
        """Parameters, under the rail, below the Recording when the page uses it."""

    def set_recording(self, recording: sim.Recording) -> None:
        """The rail's Recording changed (or the page is opening on it): rebuild what depends on it."""
        self.recording = recording

    def outline(self) -> tuple[tuple[str, str], ...]:
        """The subsections as the rail lists them: `sections`, then References when there are any."""
        return self.sections + ((REFERENCES,) if self.references else ())

    def page(self) -> None:
        """Every subsection in order, then the references, scrolling to `jump` when set."""
        for i in range(len(self.outline())):
            if self.jump == i:
                imgui.set_scroll_here_y(0.0)
                self.jump = None
            if i < len(self.sections):
                getattr(self, f"section_{i}")()
            else:
                self.draw_references()
        imgui.dummy(imgui.ImVec2(0, imgui.get_font_size() * 3))

    def draw_references(self) -> None:
        em = imgui.get_font_size()
        imgui.dummy(imgui.ImVec2(0, 1.6 * em))
        imgui.text_colored(imgui.get_style_color_vec4(imgui.Col_.text_disabled), f"{REFERENCES[0]}  {REFERENCES[1]}")
        imgui.separator()
        width = min(imgui.get_content_region_avail().x, 60 * em)
        for i, (citation, url) in enumerate(self.references):
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + width)
            imgui.text(f"{i + 1}.  {citation}")
            imgui.pop_text_wrap_pos()
            if url:
                imgui.indent(1.6 * em)
                imgui.text_link_open_url(f"{url}##ref{i}", url)
                imgui.unindent(1.6 * em)
            imgui.spacing()

    def animating(self) -> bool:
        """True while something moves on its own, so the window keeps drawing at full rate."""
        return False
