"""
Draw-list figures for the guide pages: matrices drawn cell by cell with their values, stacks of frames, sparklines,
boxes and arrows, and section headings.

Everything takes screen pixels; `canvas` reserves a w x h area at the cursor and returns its top left.
"""

import math
from typing import Callable, Optional, Sequence

import numpy as np
from imgui_bundle import imgui

from imgui_ophys.plotting import ACCENT, DIM, HIGHLIGHT, NEGATIVE, component_color
from imgui_ophys.widgets import CARD, EDGE, KEY, u32

ImVec2, ImVec4 = imgui.ImVec2, imgui.ImVec4
TEXT = ImVec4(0.92, 0.92, 0.94, 1.0)
PLOT = ImVec4(0.06, 0.06, 0.08, 1.0)
GUIDE_WIDTH_EM = 60


def em(x: float = 1.0) -> float:
    return imgui.get_font_size() * x


def guide_width() -> float:
    return min(imgui.get_content_region_avail().x, em(GUIDE_WIDTH_EM))


def canvas(w: float, h: float) -> tuple[float, float]:
    """Reserve w x h at the cursor; returns its top left in screen pixels."""
    p = imgui.get_cursor_screen_pos()
    imgui.dummy(ImVec2(w, h))
    return p.x, p.y


def mouse_in(x: float, y: float, w: float, h: float) -> bool:
    m = imgui.get_mouse_pos()
    return imgui.is_window_hovered() and x <= m.x < x + w and y <= m.y < y + h


def heading(icon: str, text: str, caption: str = "") -> None:
    """An accent section heading with a dim one-line caption under it."""
    imgui.dummy(ImVec2(0, em(1.6)))
    imgui.text_colored(ACCENT, f"{icon}  {text}")
    if caption:
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + guide_width())
        imgui.text_colored(DIM, caption)
        imgui.pop_text_wrap_pos()
    imgui.dummy(ImVec2(0, em(0.4)))


def text(dl, x: float, y: float, s: str, color: ImVec4 = TEXT, size: float = 1.0, center: bool = False) -> float:
    """s at (x, y), size times the font; centered on x when center. Returns its width."""
    w = text_width(s, size)
    dl.add_text(imgui.get_font(), em(size), ImVec2(x - w / 2 if center else x, y), u32(color, color.w), s)
    return w


def text_width(s: str, size: float = 1.0) -> float:
    """s's width at size times the font, measured at that size (glyphs do not scale exactly)."""
    if size == 1.0:
        return imgui.calc_text_size(s).x
    imgui.push_font(None, em(size) / imgui.get_style().font_scale_main)
    w = imgui.calc_text_size(s).x
    imgui.pop_font()
    return w


# ----------------------------------------------------------------------
# colors
# ----------------------------------------------------------------------


def gray(v: float) -> ImVec4:
    v = float(np.clip(v, 0, 1))
    return ImVec4(v, v, v, 1.0)


def tinted(color: ImVec4) -> Callable[[float], ImVec4]:
    """0 black to 1 color."""

    def f(v: float) -> ImVec4:
        v = float(np.clip(v, 0, 1))
        return ImVec4(color.x * v, color.y * v, color.z * v, 1.0)

    return f


def diverging(color: ImVec4) -> Callable[[float], ImVec4]:
    """-1 pink, 0 black, +1 color."""

    def f(v: float) -> ImVec4:
        v = float(np.clip(v, -1, 1))
        c = color if v >= 0 else NEGATIVE
        a = abs(v)
        return ImVec4(c.x * a, c.y * a, c.z * a, 1.0)

    return f


def ink(bg: ImVec4) -> ImVec4:
    """Black or white, whichever reads on bg."""
    lum = 0.299 * bg.x + 0.587 * bg.y + 0.114 * bg.z
    return ImVec4(0.05, 0.05, 0.05, 1.0) if lum > 0.55 else ImVec4(0.95, 0.95, 0.95, 1.0)


# ----------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------


def box(dl, x: float, y: float, w: float, h: float, label: str, color: ImVec4, fill_alpha: float = 0.0, lit: bool = False) -> None:
    """A rounded card with its label centered; lit draws the border in the accent."""
    a, b = ImVec2(x, y), ImVec2(x + w, y + h)
    dl.add_rect_filled(a, b, u32(CARD), 4.0)
    if fill_alpha:
        dl.add_rect_filled(a, b, u32(color, fill_alpha), 4.0)
    dl.add_rect(a, b, u32(ACCENT if lit else color, 1.0 if lit else 0.6), 4.0, 2.0 if lit else 1.0)
    if label:
        size = imgui.calc_text_size(label)
        dl.add_text(ImVec2(x + (w - size.x) / 2, y + (h - size.y) / 2), u32(color), label)


def arrow(dl, x0: float, y0: float, x1: float, y1: float, color: ImVec4 = DIM, thickness: float = 1.5) -> None:
    """A straight arrow from (x0, y0) to its tip at (x1, y1)."""
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    head = em(0.45)
    col = u32(color, color.w)
    dl.add_line(ImVec2(x0, y0), ImVec2(x1 - ux * head * 0.8, y1 - uy * head * 0.8), col, thickness)
    dl.add_triangle_filled(
        ImVec2(x1, y1),
        ImVec2(x1 - ux * head - uy * head * 0.5, y1 - uy * head + ux * head * 0.5),
        ImVec2(x1 - ux * head + uy * head * 0.5, y1 - uy * head - ux * head * 0.5),
        col,
    )


def matrix(
    dl,
    x: float,
    y: float,
    values: np.ndarray,
    cw: float,
    ch: Optional[float] = None,
    color: Callable[[float], ImVec4] = gray,
    lo: float = 0.0,
    hi: float = 1.0,
    fmt: Optional[str] = None,
    rows: Sequence[int] = (),
    cols: Sequence[int] = (),
    cells: Sequence[tuple[int, int]] = (),
    outline: ImVec4 = HIGHLIGHT,
) -> Optional[tuple[int, int]]:
    """
    A 2-D array drawn cell by cell, cw x ch pixels each, colored by color((v - lo) / (hi - lo)) (diverging colors get
    v / hi). fmt writes each value in its cell. rows, cols and cells are outlined. Returns the hovered (row, col).
    """
    ch = cw if ch is None else ch
    nr, nc = values.shape
    gap = 1.0 if min(cw, ch) > 4 else 0.0
    span = (hi - lo) or 1.0
    signed = lo < 0
    for r in range(nr):
        for c in range(nc):
            v = float(values[r, c])
            bg = color(v / max(abs(lo), abs(hi)) if signed else (v - lo) / span)
            a = ImVec2(x + c * cw, y + r * ch)
            dl.add_rect_filled(a, ImVec2(a.x + cw - gap, a.y + ch - gap), u32(bg))
            if fmt:
                s = fmt.format(v)
                # as big as fits: at most 0.8 em, and 90% of the cell's width
                px = min(ch * 0.55, em(0.8))
                w = text_width(s, px / em())
                if w > 0.9 * (cw - gap):
                    px *= 0.9 * (cw - gap) / w
                    w = 0.9 * (cw - gap)
                dl.add_text(imgui.get_font(), px, ImVec2(a.x + (cw - gap - w) / 2, a.y + (ch - gap - px) / 2), u32(ink(bg)), s)
    col = u32(outline)
    for r in rows:
        dl.add_rect(ImVec2(x - 1, y + r * ch - 1), ImVec2(x + nc * cw, y + (r + 1) * ch), col, 0, 2.0)
    for c in cols:
        dl.add_rect(ImVec2(x + c * cw - 1, y - 1), ImVec2(x + (c + 1) * cw, y + nr * ch), col, 0, 2.0)
    for r, c in cells:
        dl.add_rect(ImVec2(x + c * cw - 1, y + r * ch - 1), ImVec2(x + (c + 1) * cw, y + (r + 1) * ch), col, 0, 2.5)
    if mouse_in(x, y, nc * cw, nr * ch):
        m = imgui.get_mouse_pos()
        return int((m.y - y) // ch), int((m.x - x) // cw)
    return None


def sparkline(
    dl,
    x: float,
    y: float,
    w: float,
    h: float,
    values: np.ndarray,
    color: ImVec4,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    cursor: Optional[int] = None,
    thickness: float = 1.5,
    frame: bool = True,
    fill: bool = False,
) -> None:
    """values across w, scaled into h (to lo..hi, else their own range), in a dark plot box."""
    if frame:
        dl.add_rect_filled(ImVec2(x, y), ImVec2(x + w, y + h), u32(PLOT), 3.0)
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return
    lo = float(v.min()) if lo is None else lo
    hi = float(v.max()) if hi is None else hi
    span = (hi - lo) or 1.0
    pad = em(0.25)
    xs = x + pad + np.arange(v.size) * (w - 2 * pad) / (v.size - 1)
    ys = y + h - pad - (np.clip(v, lo, hi) - lo) / span * (h - 2 * pad)
    pts = [ImVec2(float(a), float(b)) for a, b in zip(xs, ys)]
    if fill:
        base = y + h - pad - (np.clip(0.0, lo, hi) - lo) / span * (h - 2 * pad)
        for a, b in zip(pts[:-1], pts[1:]):
            dl.add_quad_filled(a, b, ImVec2(b.x, base), ImVec2(a.x, base), u32(color, 0.25))
    dl.add_polyline(pts, u32(color, color.w), thickness, 0)
    if cursor is not None and 0 <= cursor < v.size:
        cx = float(xs[cursor])
        dl.add_line(ImVec2(cx, y + pad), ImVec2(cx, y + h - pad), u32(HIGHLIGHT, 0.8), 1.5)


def ticks(dl, x: float, y: float, w: float, h: float, times: np.ndarray, duration: float, color: ImVec4 = TEXT) -> None:
    """A spike raster: a vertical tick per time, across w for 0..duration."""
    dl.add_rect_filled(ImVec2(x, y), ImVec2(x + w, y + h), u32(PLOT), 3.0)
    pad = em(0.25)
    for t in times:
        tx = x + pad + t / duration * (w - 2 * pad)
        dl.add_line(ImVec2(tx, y + pad), ImVec2(tx, y + h - pad), u32(color), 1.5)


def stack(dl, x: float, y: float, frames: Sequence[np.ndarray], cell: float, hi: float, step: float) -> tuple[float, float]:
    """
    Frames drawn as a deck: frames[0] in front with its outline lit, each later one up and right by step behind it.
    Returns the front frame's top left.
    """
    n = len(frames)
    front_y = y + (n - 1) * step
    for i in range(n - 1, -1, -1):
        fx, fy = x + i * step, front_y - i * step
        matrix(dl, fx, fy, frames[i] * (1.0 if i == 0 else 0.55), cell, color=gray, lo=0, hi=hi)
        side = frames[i].shape[0] * cell
        dl.add_rect(ImVec2(fx - 1, fy - 1), ImVec2(fx + side, fy + side), u32(HIGHLIGHT if i == 0 else EDGE), 0, 2.0 if i == 0 else 1.0)
    return x, front_y


def legend_chip(dl, x: float, y: float, label: str, color: ImVec4) -> float:
    """A small colored square and its label; returns the width used."""
    s = em(0.7)
    dl.add_rect_filled(ImVec2(x, y + em(0.15)), ImVec2(x + s, y + em(0.15) + s), u32(color))
    return s + em(0.3) + text(dl, x + s + em(0.3), y, label, color) + em(0.9)


def part_color(k: int) -> ImVec4:
    return component_color(k)


def segments(dl, x: float, y: float, parts: Sequence[tuple[str, ImVec4]], size: float = 1.0) -> float:
    """Text pieces in their own colors, one after another; returns the total width."""
    # spaces at a piece's edges are measured here: text widths can leave them out
    space = text_width("a a", size) - text_width("aa", size)
    w = 0.0
    for s, color in parts:
        core = s.strip(" ")
        lead = len(s) - len(s.lstrip(" "))
        trail = len(s) - len(s.rstrip(" ")) if core else 0
        w += lead * space
        if core:
            w += text(dl, x + w, y, core, color, size)
        w += trail * space
    return w


def chip_row(dl, x: float, y: float, w: float, labels: Sequence[str], lit: int, done: int = -1) -> float:
    """
    One row of equal chips with arrows between, like a recipe: chip lit is outlined in the accent, chips up to
    done are filled. Returns the row height.
    """
    n = len(labels)
    gap = em(0.9)
    cw = (w - (n - 1) * gap) / n
    h = em(1.8)
    for i, label in enumerate(labels):
        cx = x + i * (cw + gap)
        box(dl, cx, y, cw, h, "", KEY if i == lit else DIM, 0.18 if i <= done else 0.0, i == lit)
        size = min(0.85, (cw - em(0.4)) / max(text_width(label), 1))
        text(dl, cx + cw / 2, y + (h - em(size)) / 2, label, KEY if i == lit else TEXT, size, center=True)
        if i:
            arrow(dl, cx - gap + em(0.1), y + h / 2, cx - em(0.1), y + h / 2)
    return h
