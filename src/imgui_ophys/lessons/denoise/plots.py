"""
Plot helpers of the Denoising page: its trace colors and band palette, lines decimated to the shared time window,
levels and shaded event windows, Hz ticks on a log axis, scale bars, its three colormaps and the rail's form rows.
"""

import math
from typing import Optional

import numpy as np
from imgui_bundle import imgui, implot

from imgui_ophys import diagram as D
from imgui_ophys import plotting as P
from imgui_ophys.widgets import CARD, KEY, u32

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4

INPUT = ImVec4(0.70, 0.70, 0.74, 0.75)
TRUTH = ImVec4(0.35, 0.85, 0.45, 1.0)  # the noiseless trace and the simulated action potentials
DENOISED = KEY
REMOVED = ImVec4(0.78, 0.58, 1.00, 0.85)
BASELINE = P.ACCENT
LEVEL_UP = P.WARN
LEVEL_DOWN = P.DIM
BAR = ImVec4(0.52, 0.52, 0.55, 1.0)  # scale bars sit back a little from the traces
TEXT = D.TEXT

# band colors, in band order from slowest to fastest
BAND_COLORS = tuple(
    ImVec4(r / 255, g / 255, b / 255, 1.0)
    for r, g, b in (
        (78, 121, 167), (242, 142, 43), (225, 87, 89), (118, 183, 178), (89, 161, 79), (237, 201, 72),
        (176, 122, 161), (255, 157, 167), (156, 117, 95), (186, 176, 172), (31, 119, 180), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (188, 189, 34),
    )
)

MAX_POINTS = 4000  # per series on screen; longer views are drawn as per-pixel-column min/max
NICE_HZ = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
TIME_AXIS = implot.AxisFlags_.no_decorations  # pans and zooms, draws nothing: a time bar stands in
VALUE_AXIS = implot.AxisFlags_.no_decorations | implot.AxisFlags_.auto_fit | implot.AxisFlags_.range_fit
WIDTH_CAPTION_EM = 7.0

# a scan's darkest pixels are the plot's background; a signed map's 0 is a card
RAMPS = {
    "scan": ((D.PLOT.x, D.PLOT.y, D.PLOT.z), (0.55, 0.56, 0.60), (0.97, 0.97, 0.98)),
    "sequential": ((D.PLOT.x, D.PLOT.y, D.PLOT.z), (0.20, 0.38, 0.66), (P.ACCENT.x, P.ACCENT.y, P.ACCENT.z), (0.86, 0.94, 1.00)),
    "diverging": ((0.30, 0.58, 1.00), (CARD.x, CARD.y, CARD.z), (1.00, 0.66, 0.22)),
}


def band_color(index: int, alpha: float = 1.0) -> ImVec4:
    c = BAND_COLORS[index % len(BAND_COLORS)]
    return ImVec4(c.x, c.y, c.z, alpha)


def faded(c: ImVec4, alpha: float) -> ImVec4:
    return ImVec4(c.x, c.y, c.z, alpha)


def marker(color: ImVec4, size: float = 4.0, edge: Optional[ImVec4] = None) -> implot.Spec:
    """Filled circles, size in unscaled pixels."""
    return implot.Spec(marker=implot.Marker_.circle, marker_size=size * P.SCALE, marker_fill_color=color,
                       marker_line_color=edge or color)


def window_indices(n: int, fs: float, t0: float, t1: float) -> tuple[int, int]:
    """The sample range [i0, i1) covering t0..t1 seconds, one sample of margin each side, clipped to the trace."""
    i0 = max(0, int(math.floor(t0 * fs)) - 1)
    i1 = min(n, int(math.ceil(t1 * fs)) + 2)
    return i0, max(i1, min(i0 + 2, n))


def decimate(y: np.ndarray, fs: float, t0: float, t1: float, max_points: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """
    (t, y) of the part of y between t0 and t1 seconds, as float64. Past max_points samples each bin of samples is
    drawn as its min then its max, so a spike one sample wide still shows at any zoom.
    """
    i0, i1 = window_indices(y.size, fs, t0, t1)
    seg = np.ascontiguousarray(y[i0:i1], dtype=np.float64)
    if seg.size <= max_points:
        return np.arange(i0, i0 + seg.size, dtype=np.float64) / fs, seg
    width = int(math.ceil(seg.size / (max_points // 2)))
    whole = seg.size // width * width
    bins = seg[:whole].reshape(-1, width)
    out = np.empty(bins.shape[0] * 2)
    out[0::2], out[1::2] = bins.min(axis=1), bins.max(axis=1)
    t = (i0 + np.repeat(np.arange(bins.shape[0]) * width, 2) + np.tile([0, width - 1], bins.shape[0])) / fs
    return t.astype(np.float64), out


def line(label: str, y: np.ndarray, fs: float, t0: float, t1: float, color: Optional[ImVec4] = None,
         weight: float = 1.0, flags: int = 0) -> None:
    """y against time, decimated to the view."""
    t, v = decimate(y, fs, t0, t1)
    if v.size == 0:  # y ends before the view; ImPlot rejects an empty array
        return
    implot.plot_line(label, t, v, P.spec(color, weight, flags=flags))


def hline(label: str, value: float, color: ImVec4, weight: float = 1.0) -> None:
    implot.plot_inf_lines(label, np.array([value], dtype=np.float64),
                          P.spec(color, weight, flags=implot.InfLinesFlags_.horizontal))


def vlines(label: str, values, color: ImVec4, weight: float = 1.0) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size:
        implot.plot_inf_lines(label, values, P.spec(color, weight))


def shade_intervals(intervals: np.ndarray, fs: float, t0: float, t1: float, color: ImVec4) -> None:
    """Fill each (start, end) sample interval across the plot's full height."""
    if intervals is None or len(intervals) == 0:
        return
    starts, ends = intervals[:, 0] / fs, intervals[:, 1] / fs
    visible = (ends >= t0) & (starts <= t1)
    if not np.any(visible):
        return
    draw = implot.get_plot_draw_list()
    implot.push_plot_clip_rect()
    top = implot.get_plot_pos().y
    bottom = top + implot.get_plot_size().y
    col = imgui.color_convert_float4_to_u32(color)
    for a, b in zip(starts[visible], ends[visible]):
        x0 = implot.plot_to_pixels(float(a), 0.0).x
        x1 = implot.plot_to_pixels(float(b), 0.0).x
        draw.add_rect_filled(ImVec2(x0, top), ImVec2(max(x1, x0 + 1), bottom), col)
    implot.pop_plot_clip_rect()


def hz_ticks(lo_hz: float, hi_hz: float, axis=implot.ImAxis_.y1) -> None:
    """Ticks in Hz on an axis plotted as log10(Hz)."""
    values = [f for f in NICE_HZ if lo_hz <= f <= hi_hz]
    if values:
        implot.setup_axis_ticks(axis, [math.log10(f) for f in values], [f"{f:g}" for f in values], False)


def fit_scale(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> tuple[float, float]:
    """Color limits from percentiles, never equal."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def pool_columns(m: np.ndarray, columns: int, how: str = "mean") -> tuple[np.ndarray, int]:
    """m with its columns averaged (or maxed) in equal blocks to about `columns`; also the samples covered."""
    width = max(1, int(math.ceil(m.shape[1] / columns)))
    whole = max(m.shape[1] // width * width, 1)
    if width == 1:
        return m[:, :whole], whole
    blocks = m[:, :whole].reshape(m.shape[0], -1, width)
    return (blocks.max(axis=2) if how == "max" else blocks.mean(axis=2)), whole


def pool_rows(m: np.ndarray, rows: int) -> np.ndarray:
    """m with its rows averaged in equal blocks to at most `rows`."""
    height = int(math.ceil(m.shape[0] / rows))
    if height == 1:
        return m
    whole = m.shape[0] // height * height
    return m[:whole].reshape(-1, height, m.shape[1]).mean(axis=1)


# ----------------------------------------------------------------------
# scale bars
# ----------------------------------------------------------------------


def nice(x: float) -> float:
    """x rounded down to 1, 2 or 5 times a power of ten."""
    e = 10 ** math.floor(math.log10(x))
    return e * max(m for m in (1.0, 2.0, 5.0) if m * e <= x)


def seconds_label(seconds: float) -> str:
    return f"{seconds * 1000:g} ms" if seconds < 1 else f"{seconds:g} s"


def time_bar(dl, fraction: float = 0.2) -> None:
    """A bar for a round fraction of the visible time, at the bottom right of the current plot, its length under it."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    limits = implot.get_plot_limits().x
    span = limits.max - limits.min
    if span <= 0:
        return
    seconds = nice(fraction * span)
    right = pos.x + size.x - D.em(0.6)
    left = right - seconds / span * size.x
    y = pos.y + size.y - D.em(1.2)
    dl.add_line(ImVec2(left, y), ImVec2(right, y), u32(BAR), 3.0 * P.SCALE)
    D.text(dl, 0.5 * (left + right), y + D.em(0.1), seconds_label(seconds), BAR, 0.85, center=True)


def value_bar(dl, height: float, label: str) -> None:
    """A vertical bar of `height` plot units at the right of the current plot, from its top, labelled to its left."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    limits = implot.get_plot_limits()
    span = limits.y.max - limits.y.min
    if span <= 0:
        return
    x = pos.x + size.x - D.em(0.6)
    top = pos.y + D.em(0.4)
    bottom = top + height / span * size.y
    dl.add_line(ImVec2(x, top), ImVec2(x, bottom), u32(BAR), 3.0 * P.SCALE)
    D.text(dl, x - D.text_width(label, 0.85) - D.em(0.3), 0.5 * (top + bottom) - D.em(0.5), label, BAR, 0.85)


def spike_ticks(dl, spikes: np.ndarray, color: ImVec4 = TRUTH) -> None:
    """A tick per spike along the bottom of the current plot."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    limits = implot.get_plot_limits().x
    y1 = pos.y + size.y - D.em(0.15)
    for s in spikes:
        if limits.min <= s <= limits.max:
            x = implot.plot_to_pixels(float(s), 0.0).x
            dl.add_line(ImVec2(x, y1 - D.em(0.5)), ImVec2(x, y1), u32(color), 1.5 * P.SCALE)


def ramp(stops: tuple, n: int = 64) -> np.ndarray:
    """(n, 4) float32 RGBA, linear between evenly spaced stops."""
    x = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    rgb = np.stack([np.interp(t, x, [s[i] for s in stops]) for i in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((n, 1))], axis=1).astype(np.float32)


def colormaps() -> dict:
    """The page's colormaps, registered with ImPlot once: name -> id."""
    return {name: P.add_colormap(f"denoise_{name}", ramp(stops)) for name, stops in RAMPS.items()}


# ----------------------------------------------------------------------
# the rail
# ----------------------------------------------------------------------


def tooltip(text: str) -> None:
    """text, wrapped, while the last item is hovered."""
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30)
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def begin_form(name: str) -> bool:
    """A two column table: dim captions in a fixed column, widgets filling the rest; end it with imgui.end_table."""
    if not imgui.begin_table(f"##form_{name}", 2, imgui.TableFlags_.sizing_stretch_prop):
        return False
    imgui.table_setup_column("##caption", imgui.TableColumnFlags_.width_fixed, WIDTH_CAPTION_EM * imgui.get_font_size())
    imgui.table_setup_column("##widget", imgui.TableColumnFlags_.width_stretch)
    return True


def row(caption: str, hint: str = "") -> None:
    """Start a form row: the caption, then the cursor in the widget column, the next widget as wide as it."""
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    imgui.text_disabled(caption)
    if hint:
        tooltip(hint)
    imgui.table_next_column()
    imgui.set_next_item_width(-1)


def swatch(key: str, color: ImVec4) -> None:
    em = imgui.get_font_size()
    imgui.color_button(f"##{key}", color, imgui.ColorEditFlags_.no_tooltip, ImVec2(0.8 * em, 0.8 * em))
    imgui.same_line()
