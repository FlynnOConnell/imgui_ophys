"""
ImPlot helpers the lessons share: colors, line specs, colormaps, an image plot that reports clicks, and a trace
plot with a time cursor.
"""

from typing import Callable, Optional, Sequence

import numpy as np
from imgui_bundle import imgui, implot

ImVec4 = imgui.ImVec4

ACCENT = ImVec4(0.40, 0.68, 1.00, 1.0)
DIM = ImVec4(0.62, 0.62, 0.65, 1.0)
RAW = ImVec4(0.60, 0.60, 0.58, 1.0)
WARN = ImVec4(1.0, 0.75, 0.3, 1.0)
ERROR = ImVec4(1.0, 0.4, 0.4, 1.0)
WHITE = ImVec4(1, 1, 1, 0.85)
HIGHLIGHT = ImVec4(0.922, 0.408, 0.204, 1.0)  # orange: the selected pixel and the time cursor
NEGATIVE = ImVec4(0.910, 0.482, 0.643, 1.0)  # pink: negative weights

# one color per component, in component order
COMPONENT_COLORS = tuple(
    ImVec4(r, g, b, 1.0)
    for r, g, b in (
        (0.165, 0.471, 0.839),  # blue
        (0.929, 0.631, 0.000),  # yellow
        (0.106, 0.686, 0.478),  # aqua
        (0.533, 0.529, 0.502),  # gray
        (0.580, 0.404, 0.741),  # purple
        (0.839, 0.153, 0.157),  # red
        (0.549, 0.337, 0.294),  # brown
        (0.890, 0.467, 0.761),  # magenta
    )
)

SCALE = 1.0  # the UI scale; line weights and marker sizes are in pixels, so they grow with it

IMAGE_FLAGS = (
    implot.Flags_.no_legend | implot.Flags_.no_menus | implot.Flags_.no_box_select
    | implot.Flags_.no_mouse_text | implot.Flags_.equal | implot.Flags_.no_title
)
TRACE_FLAGS = implot.Flags_.no_menus | implot.Flags_.no_box_select | implot.Flags_.no_title | implot.Flags_.no_mouse_text
HIDDEN_AXIS = implot.AxisFlags_.no_decorations | implot.AxisFlags_.lock


def component_color(index: int, alpha: float = 1.0) -> ImVec4:
    c = COMPONENT_COLORS[index % len(COMPONENT_COLORS)]
    return ImVec4(c.x, c.y, c.z, alpha)


def spec(color: Optional[ImVec4] = None, weight: float = 1.0, fill_alpha: float = 1.0, flags: int = 0) -> implot.Spec:
    return implot.Spec(line_color=color, fill_color=color, line_weight=weight * SCALE, fill_alpha=fill_alpha, flags=flags)


def begin_plot(title: str, size: imgui.ImVec2, flags: int = 0) -> bool:
    """implot.begin_plot with the legend in a row above the plot, so big text never covers the data."""
    if not implot.begin_plot(title, size, flags):
        return False
    if not flags & implot.Flags_.no_legend:
        implot.setup_legend(implot.Location_.north_west, implot.LegendFlags_.outside | implot.LegendFlags_.horizontal)
    return True


class Colormaps:
    """
    Colormaps registered with ImPlot, made on first use because they need an ImPlot context: a black-to-color map
    per component, a pink-black-color diverging map per component, and gray.
    """

    def __init__(self, n: int = 64):
        ramp = np.linspace(0, 1, n)[:, None]
        alpha = np.ones((n, 1))
        neg = np.array([NEGATIVE.x, NEGATIVE.y, NEGATIVE.z])
        self.sequential, self.diverging = [], []
        for k, c in enumerate(COMPONENT_COLORS):
            rgb = np.array([c.x, c.y, c.z])
            self.sequential.append(add_colormap(f"ophys_seq{k}", np.hstack([ramp * rgb, alpha])))
            div = np.vstack([(1 - ramp) * neg, ramp * rgb])
            self.diverging.append(add_colormap(f"ophys_div{k}", np.hstack([div, np.ones((2 * n, 1))])))
        self.gray = add_colormap("ophys_gray", np.hstack([np.repeat(ramp, 3, axis=1), alpha]))
        div = np.vstack([(1 - ramp) * neg, ramp * np.array([0.92, 0.92, 0.92])])
        self.residual = add_colormap("ophys_residual", np.hstack([div, np.ones((2 * n, 1))]))


def add_colormap(name: str, rgba: np.ndarray) -> int:
    """The colormap's index, reusing one already registered under name (ImPlot asserts on duplicates)."""
    existing = implot.get_colormap_index(name)
    if existing != -1:
        return existing
    return implot.add_colormap(name, np.ascontiguousarray(rgba, dtype=np.float32), False)


def image_plot(
    label: str,
    img: np.ndarray,
    size: float,
    vmin: float,
    vmax: float,
    cmap: int,
    on_click: Optional[Callable[[int, int], None]] = None,
    mark: Optional[tuple[int, int]] = None,
) -> None:
    """
    An (H, W) image as a heatmap, row 0 at the top, size pixels wide. on_click gets the (x, y) pixel clicked; mark
    outlines one (x, y) pixel.
    """
    h, w = img.shape
    implot.push_colormap(cmap)
    if implot.begin_plot(label, imgui.ImVec2(size, size * h / w), IMAGE_FLAGS):
        implot.setup_axes("", "", HIDDEN_AXIS, HIDDEN_AXIS)
        implot.setup_axes_limits(0, w, 0, h, imgui.Cond_.always)
        implot.plot_heatmap(
            "##img", np.ascontiguousarray(img, dtype=np.float64), vmin, vmax, "", implot.Point(0, 0), implot.Point(w, h)
        )
        if mark is not None:
            x, y = mark
            xs = np.array([x, x + 1, x + 1, x, x], dtype=np.float64)
            ys = h - np.array([y, y, y + 1, y + 1, y], dtype=np.float64)
            implot.plot_line("##mark", xs, ys, spec(HIGHLIGHT, 2.5))
        if on_click is not None and implot.is_plot_hovered() and imgui.is_mouse_down(0):
            p = implot.get_plot_mouse_pos()
            on_click(int(np.clip(np.floor(p.x), 0, w - 1)), int(np.clip(h - 1 - np.floor(p.y), 0, h - 1)))
        implot.end_plot()
    implot.pop_colormap()


def trace_plot(
    label: str,
    t: np.ndarray,
    series: Sequence[tuple],
    height: float,
    cursor: Optional[float] = None,
    legend: bool = True,
) -> None:
    """
    Lines against t, the x range locked to t. series holds (name, values, color, weight); a name starting with ##
    stays out of the legend. cursor draws a vertical line at that x.
    """
    flags = TRACE_FLAGS | (0 if legend else implot.Flags_.no_legend)
    if begin_plot(label, imgui.ImVec2(-1, height), flags):
        implot.setup_axes("", "", implot.AxisFlags_.no_label, implot.AxisFlags_.auto_fit)
        implot.setup_axis_limits(implot.ImAxis_.x1, float(t[0]), float(t[-1]), imgui.Cond_.always)
        for name, values, color, weight in series:
            implot.plot_line(name, t, np.ascontiguousarray(values, dtype=np.float64), spec(color, weight))
        if cursor is not None:
            implot.plot_inf_lines("##now", np.array([float(cursor)]), spec(HIGHLIGHT, 1.0))
        implot.end_plot()
