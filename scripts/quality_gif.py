"""
docs/quality.gif: spike detectability d' on an AOD voltage recording, staged the way manim explainers stage things: a
question first, a few words on screen per beat, the objects named by their colors, live numbers where a number is the
point, the rest dimmed while one thing is shown, and a pause after each beat. Beats: the trace and its patch; the spikes marked; one
spike; moving 2 ms window; no spike, jitter (1 SD); spike, a jump (d'); pick a threshold (live hit and false-alarm rates
as it rises); is 6 good.

The recording is an SCE013 ASAP7 chessboard patch (its mean image, ROI and summed trace saved by the voltage-snr-metric
workspace in gif_unit.npz). Spikes are added because the recording's own are too few and too small to teach from; the
canvas says so. The drawing uses imgui_masknmf_manim's canvas (manim's Cairo camera, no Scene), so run this in that
repo's venv:

    ../imgui_masknmf_manim/.venv/bin/python scripts/quality_gif.py [--data gif_unit.npz] [--out docs/quality.gif]
"""

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import manim as mn
import numpy as np
from imgui_masknmf_manim import canvas as cv
from imgui_masknmf_manim import style

REPO = Path(__file__).resolve().parent.parent
DATA = Path.home() / "agent_assets/pml_utilities/20261026_voltage-snr-metric/artifacts/data/gif_unit.npz"
FPS = 20  # a GIF's frame delays are in 1/100 s, so 20 fps is exact
SECONDS_HOLD = 4.0  # after the story ends, with the shape still sliding
SECONDS_RISE = 5.0  # the threshold climbs from 1 to THRESHOLD SD in this long
SECONDS_FADE = 0.6  # to the background at the end, so the loop restarts from an empty canvas without a jump
SECONDS_SLIDE = 6.0  # the shape crosses the window in this long once the story has drawn the match
PAD = 14  # px kept around everything the animation draws

A_SPIKE = 0.20  # added spikes: 20% dF/F
RATE = 2.0  # added spikes / s over the whole recording (the window shows the stretch with three)
RISE, DECAY = 0.0005, 0.0015  # s: the reference spike's shape
HIGHPASS = 0.020  # s: running mean removed before matching, so slow wobbles don't count
WINDOW = 1.0  # s shown
THRESHOLD = 4.0  # SD
GRADES = ((0.0, 3.5, style.ERROR, "spikes lost in noise"), (3.5, 6.0, style.WARN, "marginal"), (6.0, 10.0, style.OK, "reliable"))

TRACE = style.TEXT
SPIKE = (1.00, 0.55, 0.35)
MATCH = style.ACCENT
SHAPE = style.BACKGROUND
KEY = style.KEY
X_LEFT, W_LEFT = -2.2, 9.2
X_RIGHT, W_RIGHT = 4.9, 3.4
Y_TOP, Y_BOTTOM, H = 1.75, -0.95, 1.8
Y_TITLE = 3.5
Y_COUNTERS = -2.2
Y_SCALE = -3.25


# ----------------------------------------------------------------------------------------------------------------------
# numbers
# ----------------------------------------------------------------------------------------------------------------------


def q(x: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def spike_shape(fs: float) -> np.ndarray:
    """The reference spike: a dip with RISE and DECAY, peak -1, sampled at fs."""
    t = np.arange(0, max(0.008, 6 * DECAY), 1 / fs)
    h = (1 - np.exp(-t / RISE)) * np.exp(-t / DECAY)
    return -h / h.max()


def running_mean(x: np.ndarray, width: int) -> np.ndarray:
    padded = np.pad(x, width // 2, mode="edge")
    c = np.cumsum(np.insert(padded, 0, 0.0))
    return (c[width:] - c[:-width])[: x.size] / width


def inject(F: np.ndarray, fs: float, rng) -> tuple:
    """F with A_SPIKE spikes multiplied in at RATE / s, at least 50 ms apart; returns (F, onset samples)."""
    h = spike_shape(fs)
    n = F.size
    candidates = np.arange(h.size, n - h.size)
    times = np.sort(rng.choice(candidates, int(RATE * n / fs), replace=False))
    times = times[np.insert(np.diff(times) > 0.05 * fs, 0, True)]
    gain = np.zeros(n)
    for t0 in times:
        gain[t0 : t0 + h.size] += h
    return F * (1 + A_SPIKE * gain), times


def match(dff: np.ndarray, fs: float) -> tuple:
    """(high-passed dff, the high-passed shape, the matched filter's output z-scored on the samples away from spikes)."""
    width = max(3, int(round(HIGHPASS * fs)))
    x = dff - running_mean(dff, width)
    h = spike_shape(fs)
    hp = np.pad(h, width)
    hp = hp - running_mean(hp, width)
    hp = hp[width // 2 : width // 2 + h.size + width]  # keeps the dip and the undershoot that follows
    y = np.correlate(x, hp, "full")[hp.size - 1 :]  # y[t] = sum x[t + k] hp[k]: high where hp's dip starts at t
    offset = width - width // 2  # hp's dip sits this far past h's, so y peaks that much before the spike's onset
    y = np.concatenate([np.full(offset, y[0]), y[: y.size - offset]])
    return x, hp, y


def analyse(data) -> dict:
    fs = float(data["fs"])
    rng = np.random.default_rng(0)
    F, onsets = inject(data["F"].astype(float), fs, rng)
    dff_plain = 100 * (data["F"] / np.median(data["F"]) - 1)
    dff = 100 * (F / np.median(F) - 1)
    x, hp, y = match(dff, fs)
    near = np.zeros(F.size, bool)
    reach = int(0.012 * fs)
    for t0 in onsets:
        near[max(0, t0 - reach) : t0 + reach] = True
    free = ~near
    free[: int(fs)] = False  # the first second: the shutter transient
    center = np.median(y[free])
    spread = 1.4826 * np.median(np.abs(y[free] - center))
    z = (y - center) / spread
    half = max(1, int(round(0.002 * fs)))
    peaks = np.array([z[t0 - half : t0 + half + 1].max() for t0 in onsets])
    peak_at = np.array([t0 - half + int(np.argmax(z[t0 - half : t0 + half + 1])) for t0 in onsets])
    # the window: 1 s with three added spikes, the first at least 0.15 s in, none in the first second
    n = int(WINDOW * fs)
    candidates = []
    for t0 in onsets:
        if t0 < fs:
            continue
        i0 = t0 - int(0.15 * fs)
        inside = (onsets >= i0) & (onsets < i0 + n - int(0.1 * fs))
        if inside.sum() == 3 and i0 + n <= F.size:
            candidates.append(int(i0))
    if not candidates:
        raise RuntimeError("no 1 s window with three added spikes")
    # the steadiest of them: where the slow brightness (50 ms mean) moves least, so no drift or motion steals the picture
    drift = [np.ptp(running_mean(dff_plain[i0 : i0 + n], max(3, int(0.05 * fs)))) for i0 in candidates]
    start = candidates[int(np.argmin(drift))]
    sl = slice(start, start + n)
    return dict(fs=fs, start=start, n=n, dff_plain=dff_plain[sl], dff=dff[sl], z=z[sl], onsets=onsets[(onsets >= start) & (onsets < start + n)] - start,
                peak_at=peak_at[(peak_at >= start) & (peak_at < start + n)] - start, shape=spike_shape(fs), h=hp, noise=z[free],
                peaks=peaks, dprime=float(peaks.mean()), D=float(data["D"]), photons_per_ms=float(data["photons_per_ms"]),
                mean=data["mean"], mask=data["mask"].reshape(data["mean"].shape))


# ----------------------------------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------------------------------


def put(mobject: mn.Mobject, x: float, y: float) -> mn.Mobject:
    return mobject.move_to(np.array([x, y, 0.0]))


def axes(x_min: float, x_max: float, y_min: float, y_max: float, width: float, height: float) -> mn.Axes:
    """Axes without ticks or tips, like canvas.plain_axes but with x from x_min."""
    return mn.Axes(
        x_range=[x_min, x_max, x_max - x_min], y_range=[y_min, y_max, y_max - y_min], x_length=width, y_length=height,
        tips=False, axis_config={"include_ticks": False, "stroke_width": 1.5, "color": style.manim_color(style.EDGE)},
    )


def path(ax: mn.Axes, x: np.ndarray, y: np.ndarray, color: tuple, stroke_width: float) -> mn.VMobject:
    return mn.VMobject(stroke_color=style.manim_color(color), stroke_width=stroke_width).set_points_as_corners(cv.axes_points(ax, x, y))


def vline(ax: mn.Axes, x: float, y_min: float, y_max: float, color: tuple, stroke_width: float = 1.5, opacity: float = 0.9) -> mn.Line:
    bottom, top = cv.axes_points(ax, np.full(2, x), np.array([y_min, y_max]))
    return mn.Line(bottom, top, color=style.manim_color(color), stroke_width=stroke_width, stroke_opacity=opacity)


def hline(ax: mn.Axes, y: float, x_min: float, x_max: float, color: tuple, stroke_width: float = 1.5, opacity: float = 0.9) -> mn.Line:
    left, right = cv.axes_points(ax, np.array([x_min, x_max]), np.full(2, y))
    return mn.Line(left, right, color=style.manim_color(color), stroke_width=stroke_width, stroke_opacity=opacity)


def band(ax: mn.Axes, y_min: float, y_max: float, x_min: float, x_max: float, color: tuple, opacity: float) -> mn.Rectangle:
    corners = cv.axes_points(ax, np.array([x_min, x_max]), np.array([y_min, y_max]))
    rectangle = mn.Rectangle(width=corners[1][0] - corners[0][0], height=corners[1][1] - corners[0][1],
                             fill_color=style.manim_color(color), fill_opacity=opacity, stroke_width=0)
    return rectangle.move_to(corners[0], aligned_edge=mn.DL)


def bars(ax: mn.Axes, edges: np.ndarray, heights: np.ndarray, color: tuple, opacity: float = 0.6) -> mn.VGroup:
    """A histogram as filled rectangles from 0 to each height."""
    group = mn.VGroup()
    for left, right, height in zip(edges[:-1], edges[1:], heights):
        if height > 0:
            group.add(band(ax, 0.0, height, left, right, color, opacity))
    return group


def dot(point: np.ndarray, color: tuple, radius: float = 0.07) -> mn.Dot:
    return mn.Dot(point, radius=radius, color=style.manim_color(color))


def title(text: str) -> mn.Text:
    return put(cv.label(text, style.TEXT, 32), 0, Y_TITLE)


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def per_minute(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") if x >= 10 else f"{x:.1f}"


def roi_outline(image: mn.ImageMobject, mask: np.ndarray) -> mn.VGroup:
    """The edges of the mask's pixels that border pixels outside it."""
    rows, cols = mask.shape
    size = image.height / rows
    corner = image.get_corner(mn.UL)
    group = mn.VGroup()

    def edge(r0, c0, r1, c1):
        a = corner + np.array([c0 * size, -r0 * size, 0.0])
        b = corner + np.array([c1 * size, -r1 * size, 0.0])
        group.add(mn.Line(a, b, color=style.manim_color(style.PICK), stroke_width=2.0))

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            if r == 0 or not mask[r - 1, c]:
                edge(r, c, r, c + 1)
            if r == rows - 1 or not mask[r + 1, c]:
                edge(r + 1, c, r + 1, c + 1)
            if c == 0 or not mask[r, c - 1]:
                edge(r, c, r + 1, c)
            if c == cols - 1 or not mask[r, c + 1]:
                edge(r, c + 1, r + 1, c + 1)
    return group


def build(a: dict) -> tuple:
    """(story, beat start times, the Create of the match, the sliding shape, its points at sample 0, axes and ranges the live elements need)."""
    fs, n = a["fs"], a["n"]
    t = np.arange(n) / fs
    seconds, chapters, titles = [], [], []

    def beat(duration: float, text: str | None, *animations, lag_ratio: float = 0.1):
        """One titled step; text None keeps the previous title."""
        swaps = []
        if text is not None:
            new = title(text)
            swaps = [mn.AnimationGroup(mn.FadeOut(titles[-1], run_time=0.3), mn.FadeIn(new, run_time=0.5), lag_ratio=1.0) if titles else mn.FadeIn(new, run_time=0.5)]
            titles.append(new)
        seconds.append(duration)
        # a Wait as long as the slot keeps manim from stretching short reveals to fill it: reveals take their own second, then hold
        chapters.append(cv.chapter(duration, mn.Wait(run_time=0.8 * duration), mn.AnimationGroup(*swaps, *animations, lag_ratio=lag_ratio)))

    # 1. the trace, and the neuron it came from
    mean, mask = a["mean"], a["mask"]
    patch = cv.image_mobject(mean, W_RIGHT, H, 0.0, float(np.percentile(mean, 99.5)), "movie")
    put(patch, X_RIGHT, Y_TOP)
    outline = roi_outline(patch, mask)
    name_patch = cv.label(f"{a['label']}, {fs:.0f} frames / s", style.DIM, 18).next_to(patch, mn.DOWN, buff=0.12)
    spread = 1.4826 * float(np.median(np.abs(a["dff"] - np.median(a["dff"]))))  # the trace's noise, in %
    lo, hi = -(100 * A_SPIKE + 3.5 * spread), 4.0 * spread
    ax_rec = put(axes(0, WINDOW, lo, hi, W_LEFT, H), X_LEFT, Y_TOP)
    trace = path(ax_rec, t, np.clip(a["dff"], lo, hi), TRACE, 1.2)
    name_rec = cv.label("brightness, % change", TRACE, 18).next_to(ax_rec, mn.UP, buff=0.08).align_to(ax_rec, mn.LEFT)
    zero = hline(ax_rec, 0.0, 0, WINDOW, style.DIM, 1.0, 0.5)
    scale_bar = hline(ax_rec, lo + 0.06 * (hi - lo), WINDOW - 0.1, WINDOW, style.DIM, 2.0)
    scale_text = cv.label("100 ms", style.DIM, 16).next_to(scale_bar, mn.DOWN, buff=0.06)
    beat(5.0, "d': spike detectability",
         mn.FadeIn(patch), mn.Create(outline), mn.FadeIn(name_patch), mn.Create(ax_rec), mn.FadeIn(zero), mn.Create(trace),
         mn.FadeIn(name_rec), mn.FadeIn(scale_bar), mn.FadeIn(scale_text))

    # 2. marks drop onto the spikes; the title stays
    ticks = mn.VGroup(*[vline(ax_rec, s / fs, hi - 0.16 * (hi - lo), hi, SPIKE, 3.0) for s in a["onsets"]])
    name_ticks = cv.label(f"3 spikes, added: {100 * A_SPIKE:.0f}% dips", SPIKE, 18).next_to(ax_rec, mn.UP, buff=0.08).align_to(ax_rec, mn.RIGHT)
    beat(5.0, None, mn.FadeIn(ticks, shift=0.3 * mn.DOWN, lag_ratio=0.3), mn.FadeIn(name_ticks))

    # 3. one spike.  a window around the first dip grows into the shape on the right
    shape = a["shape"]
    ms = np.arange(shape.size) / fs * 1e3
    s0 = int(a["onsets"][0])
    seg = slice(max(0, s0 - int(0.002 * fs)), s0 + shape.size)
    segment = path(ax_rec, t[seg], np.clip(a["dff"][seg], lo, hi), SHAPE, 2.5)
    box = mn.SurroundingRectangle(segment, color=style.manim_color(SHAPE), buff=0.12, stroke_width=2.0)
    ax_h = put(axes(-2.0, ms[-1] + 1.0, -1.25, 0.35, W_RIGHT, H), X_RIGHT, Y_TOP)
    h_zero = hline(ax_h, 0.0, -2.0, ms[-1] + 1.0, style.DIM, 1.0, 0.6)
    h_target = path(ax_h, ms, shape, SHAPE, 3.0)
    width_bar = hline(ax_h, -1.18, 0.0, 2.0, SHAPE, 2.0)
    width_text = cv.label("2 ms", SHAPE, 18).next_to(width_bar, mn.DOWN, buff=0.06)
    name_h = cv.label("the shape of a spike", SHAPE, 18).next_to(ax_h, mn.UP, buff=0.08)
    beat(7.0, "One spike",
         mn.FadeOut(patch), mn.FadeOut(outline), mn.FadeOut(name_patch), mn.Create(box), mn.FadeIn(segment),
         mn.Create(ax_h), mn.FadeIn(h_zero), mn.Transform(segment, h_target, run_time=1.5), mn.FadeOut(box), mn.FadeIn(name_h),
         mn.FadeIn(width_bar), mn.FadeIn(width_text), lag_ratio=0.3)

    # 4. moving 2 ms window.  a copy of the shape sits on the trace and moves; the match draws underneath
    z = a["z"]
    base = cv.axes_points(ax_rec, np.arange(shape.size) / fs, A_SPIKE * 100 * shape)
    slider = mn.VMobject(stroke_color=style.manim_color(KEY), stroke_width=3.5).set_points_as_corners(base)
    cursors = [cv.cursor(ax_rec, fs, lo, hi)]
    z_lo, z_hi = -4.5, max(float(z.max()) + 1.0, 8.0)
    ax_z = put(axes(0, WINDOW, z_lo, z_hi, W_LEFT, H), X_LEFT, Y_BOTTOM)
    z_zero = hline(ax_z, 0.0, 0, WINDOW, style.DIM, 1.0, 0.5)
    z_path = path(ax_z, t, np.clip(z, z_lo, z_hi), MATCH, 1.4)
    name_z = cv.label("match", MATCH, 18).next_to(ax_z, mn.UP, buff=0.08).align_to(ax_z, mn.LEFT)
    cursors.append(cv.cursor(ax_z, fs, z_lo, z_hi))
    create_z = mn.Create(z_path, rate_func=mn.linear, run_time=6.0)
    beat(9.0, "Moving 2 ms window",
         mn.GrowFromPoint(slider, h_target.get_center(), run_time=0.6), *[mn.FadeIn(k.line, run_time=0.6) for k in cursors],
         mn.Create(ax_z, run_time=0.6), mn.FadeIn(z_zero, run_time=0.6), mn.FadeIn(name_z, run_time=0.6), create_z, lag_ratio=0.0)

    # 5. no spike: jitter.  the trace dims; the match's spread is 1 SD; the pile of its values
    sd_band = band(ax_z, -1.0, 1.0, 0, WINDOW, MATCH, 0.22)
    brace = mn.Brace(sd_band, mn.RIGHT, buff=0.08, color=style.manim_color(MATCH))
    sd_text = cv.label("1 SD", MATCH, 18).next_to(brace, mn.RIGHT, buff=0.06)
    noise = a["noise"]
    edges = np.linspace(z_lo, z_hi, 51)
    density_noise = np.histogram(noise, edges)[0] / noise.size
    density_noise = 0.8 * density_noise / density_noise.max()
    ax_hist = put(axes(z_lo, z_hi, 0.0, 1.6, W_RIGHT, H), X_RIGHT, Y_BOTTOM)
    hist_noise = bars(ax_hist, edges, density_noise, MATCH)
    name_hist = cv.label("how often each match value occurs", style.DIM, 18).next_to(ax_hist, mn.UP, buff=0.08)
    # the two words sit under their piles, pushed apart only when they would touch (at a low d')
    mid, gap, per_sd = 0.5 * a["dprime"], 1.0, W_RIGHT / (z_hi - z_lo)
    legend_noise = cv.label("no spike", MATCH, 18)
    legend_noise.move_to(cv.axes_points(ax_hist, np.array([min(0.0, mid - gap - 0.5 * legend_noise.width / per_sd)]), np.array([-0.22]))[0])
    dim = [trace.animate.set_stroke(opacity=0.3), ticks.animate.set_stroke(opacity=0.3), name_ticks.animate.set_opacity(0.3),
           name_rec.animate.set_opacity(0.3)]
    beat(7.0, "No spike: jitter",
         *dim, mn.FadeIn(sd_band), mn.GrowFromCenter(brace), mn.FadeIn(sd_text), mn.Create(ax_hist), mn.FadeIn(name_hist),
         mn.FadeIn(hist_noise, lag_ratio=0.02), mn.FadeIn(legend_noise))

    # 6. spike: a jump.  the peaks light up; their pile; the distance between the piles is d'
    peaks = a["peaks"]
    density_spike = np.histogram(peaks, edges)[0] / peaks.size
    density_spike = 0.8 * density_spike / density_spike.max()
    hist_spike = bars(ax_hist, edges, density_spike, SPIKE, 0.75)
    peak_dots = mn.VGroup(*[dot(cv.axes_points(ax_z, np.array([s / fs]), np.array([min(z[s], z_hi)]))[0], SPIKE, 0.09) for s in a["peak_at"]])
    legend_spike = cv.label("spike", SPIKE, 18)
    legend_spike.move_to(cv.axes_points(ax_hist, np.array([max(a["dprime"], mid + gap + 0.5 * legend_spike.width / per_sd)]), np.array([-0.22]))[0])
    start, end = cv.axes_points(ax_hist, np.array([0.0, a["dprime"]]), np.full(2, 1.15))
    arrow = mn.DoubleArrow(start, end, color=style.manim_color(KEY), stroke_width=3.0, tip_length=0.16, buff=0)
    name_d = cv.label(f"d' = {a['dprime']:.1f}", KEY, 26).next_to(arrow, mn.UP, buff=0.05)
    sub_d = cv.label("SDs apart", KEY, 16).next_to(arrow, mn.DOWN, buff=0.04)
    undim = [trace.animate.set_stroke(opacity=1.0), ticks.animate.set_stroke(opacity=1.0), name_ticks.animate.set_opacity(1.0),
             name_rec.animate.set_opacity(1.0)]
    beat(7.0, "Spike: a jump",
         *undim, mn.FadeIn(peak_dots, scale=2.0, lag_ratio=0.2), mn.FadeIn(hist_spike, lag_ratio=0.05), mn.FadeIn(legend_spike),
         mn.GrowArrow(arrow), mn.FadeIn(name_d), mn.FadeIn(sub_d))

    # 7. pick a threshold.  it rises (drawn live) while the hit and false-alarm counters change
    beat(9.0, "Pick a threshold")

    # 8. is 6 good?  the scale
    x0, x1 = -5.5, 5.5
    per = (x1 - x0) / GRADES[-1][1]
    scale = mn.VGroup()
    scale_names = mn.VGroup()
    for low, high, color, text in GRADES:
        segment_bar = mn.Rectangle(width=(high - low) * per, height=0.22, fill_color=style.manim_color(color), fill_opacity=0.85, stroke_width=0)
        scale.add(segment_bar.move_to(np.array([x0 + low * per, Y_SCALE, 0.0]), aligned_edge=mn.LEFT))
        scale_names.add(put(cv.label(f"{low:g} to {high:g}: {text}" if high < GRADES[-1][1] else f"{low:g} and up: {text}", color, 18),
                            x0 + (low + high) / 2 * per, Y_SCALE - 0.32))
    marker = mn.Triangle(color=style.manim_color(KEY), fill_color=style.manim_color(KEY), fill_opacity=1.0, stroke_width=0)
    marker.scale(0.12).rotate(math.pi).move_to(np.array([x0 + a["dprime"] * per, Y_SCALE + 0.25, 0.0]))
    marker_name = cv.label(f"d' = {a['dprime']:.1f}", KEY, 18).next_to(marker, mn.UP, buff=0.05)
    beat(7.0, f"Is {a['dprime']:.0f} good?", mn.FadeIn(scale, lag_ratio=0.2), mn.FadeIn(scale_names, lag_ratio=0.2),
         mn.FadeIn(marker, shift=0.3 * mn.DOWN), mn.FadeIn(marker_name))

    starts = np.concatenate([[0.0], np.cumsum(seconds)[:-1]])
    live = dict(ax_z=ax_z, ax_hist=ax_hist, z_lo=z_lo, z_hi=z_hi, z=z)
    return cv.story([], chapters, cursors=cursors), starts, create_z, slider, base, live


class Live:
    """The elements drawn outside the story: the match readout by the cursor, and the threshold with its two counters."""

    def __init__(self, a: dict, live: dict, t_threshold: float):
        self.a, self.live, self.t_threshold = a, live, t_threshold
        ax_z, ax_hist = live["ax_z"], live["ax_hist"]
        self.line = hline(ax_z, 1.0, 0, WINDOW, KEY, 1.5, 0.9)
        self.line_hist = vline(ax_hist, 1.0, 0.0, 1.0, KEY, 1.5, 0.9)
        self.name = cv.label("threshold", KEY, 18)

    def threshold(self, t: float) -> float:
        return 1.0 + (THRESHOLD - 1.0) * smooth((t - self.t_threshold - 0.8) / SECONDS_RISE)

    def mobjects(self, t: float, sample: int, sliding: bool) -> list:
        out = []
        ax_z, ax_hist, z = self.live["ax_z"], self.live["ax_hist"], self.live["z"]
        if sliding:
            top = cv.axes_points(ax_z, np.array([sample / self.a["fs"]]), np.array([self.live["z_hi"]]))[0]
            readout = cv.label(f"{z[sample]:+.1f} SD", MATCH, 16).next_to(top, mn.UR if sample < 0.85 * self.a["n"] else mn.UL, buff=0.05)
            out.append(readout)
        if t >= self.t_threshold:
            k = self.threshold(t)
            left, right = cv.axes_points(ax_z, np.array([0.0, WINDOW]), np.full(2, k))
            self.line.put_start_and_end_on(left, right)
            bottom, top = cv.axes_points(ax_hist, np.full(2, k), np.array([0.0, 1.0]))
            self.line_hist.put_start_and_end_on(bottom, top)
            self.name.next_to(right, mn.UL, buff=0.06)
            caught = 100 * q(k - self.a["dprime"])
            false = 60 * self.a["fs"] * q(k)
            counters = mn.VGroup(
                cv.label(f"threshold  {k:.1f} SD", KEY, 20),
                cv.label(f"spikes caught  {caught:.0f}%", SPIKE, 20),
                cv.label(f"false alarms  {per_minute(false)} / min", MATCH, 20),
            ).arrange(mn.RIGHT, buff=0.7)
            put(counters, X_LEFT, Y_COUNTERS)
            out += [self.line, self.line_hist, self.name, counters]
        return out


def sample_at(t: float, slide: tuple, n: int) -> int:
    """Which sample of the window the shape sits on at story time t: 0 before the match is drawn, then across it."""
    start, end = slide
    if t < start:
        return 0
    if t < end:
        return int((t - start) / (end - start) * (n - 1))
    return int(n - 1 + (t - end) * n / SECONDS_SLIDE) % n


def record(a: dict, filepath_lossless: Path, width: int) -> tuple:
    """Draw every frame into a lossless FFV1 file; returns (frames, (top, bottom, left, right) the animation covers)."""
    story, starts, create_z, slider, base, live = build(a)
    i = story.leaves.index(create_z)
    slide = (float(story.starts[i]), float(story.ends[i]))
    extra = Live(a, live, float(starts[6]))
    per_sample = story.cursors[0].per_frame
    camera = cv.new_camera(width)
    background = cv.capture(camera, [])[..., :3]
    height = background.shape[0]
    n_frames = int(round((story.duration + SECONDS_HOLD) * FPS))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-c:v", "ffv1", str(filepath_lossless)],
        stdin=subprocess.PIPE,
    )
    top, bottom, left, right = height, 0, width, 0
    n = a["n"]
    for k in range(n_frames):
        t = k / FPS
        sample = sample_at(t, slide, n)
        mobjects = cv.set_story(story, min(t, story.duration), sample)
        sliding = t >= slide[0]
        if sliding:
            slider.set_points_as_corners(base + sample * per_sample)
        mobjects += extra.mobjects(t, sample, sliding)
        rgb = cv.capture(camera, mobjects)[..., :3]
        drawn = np.any(rgb != background, axis=-1)
        rows, cols = np.flatnonzero(drawn.any(axis=1)), np.flatnonzero(drawn.any(axis=0))
        if rows.size:
            top, bottom = min(top, rows[0]), max(bottom, rows[-1])
            left, right = min(left, cols[0]), max(right, cols[-1])
        fade = min(max((t - (n_frames / FPS - SECONDS_FADE)) / SECONDS_FADE, 0.0), 1.0)
        if fade > 0:
            rgb = np.round((1 - fade) * rgb + fade * background).astype(np.uint8)
        ffmpeg.stdin.write(rgb.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the frames")
    return n_frames, (max(top - PAD, 0), min(bottom + PAD, height - 1), max(left - PAD, 0), min(right + PAD, width - 1))


def encode_gif(filepath_lossless: Path, filepath_gif: Path, box: tuple) -> None:
    """Cropped to box, with one palette from every pixel and no dithering: lines on a flat background stay crisp and small."""
    top, bottom, left, right = box
    width, height = (right - left + 1) // 2 * 2, (bottom - top + 1) // 2 * 2
    graph = (f"crop={width}:{height}:{left}:{top},split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
             "[b][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(filepath_lossless), "-vf", graph, str(filepath_gif)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--width", type=int, default=1100, help="manim's frame in px, before the crop")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "quality.gif")
    parser.add_argument("--label", default="Example soma", help="what the patch is, written under it")
    args = parser.parse_args()
    style.setup_manim()
    a = analyse(np.load(args.data))
    a["label"] = args.label
    print(f"{a['peaks'].size} added spikes, measured d' {a['dprime']:.2f} (the metric's D {a['D']:.2f}), "
          f"{a['photons_per_ms']:.0f} photons / ms, window from {a['start'] / a['fs']:.2f} s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as dir_tmp:
        filepath_lossless = Path(dir_tmp) / "quality.mkv"
        n_frames, box = record(a, filepath_lossless, args.width)
        encode_gif(filepath_lossless, args.out, box)
    top, bottom, left, right = box
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB, {right - left + 1} x {bottom - top + 1} px, {n_frames} frames at {FPS} fps")


if __name__ == "__main__":
    main()
