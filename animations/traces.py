# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
Traces as one animation: one spike through the indicator, the camera and the photon count; the whole recording and its
dF/F; the recording as a sum of parts, taken away one at a time; and smoothing. Built from imgui_ophys's trace
simulator (GCaMP6f, 30 Hz); drawn with kit.py.

    uv run -p 3.12 animations/traces.py --out <dir> [--stills-only] [--gif-width 720]
"""

import sys
from pathlib import Path

import manim as mn
import numpy as np

import kit
from kit import ACCENT, DIM, KEY, TEXT, TRACE, TRUTH, Count, at, axes_points, chapter, color, hbar, label, path, plain_axes, put, replace, timeline, vbar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from imgui_ophys import sim  # noqa: E402

TITLE = "Traces: from a spike to a recorded dF/F"
CHAPTERS = (
    ("", 7.0),
    ("dF/F = (F - F0) / F0", 7.0),
    ("F = signal + bleaching + neuropil + shot noise + read noise", 6.0),
    ("Remove neuropil", 4.5),
    ("Remove bleaching", 4.5),
    ("Remove noise", 4.5),
    ("Gaussian smoothing, SD 1 frame", 6.0),
    ("SD 3 frames", 6.0),
)
UNTITLED = {0: "one-spike"}
PARTS = (("signal", TRUTH), ("bleaching", (0.95, 0.55, 0.15)), ("neuropil", (0.58, 0.40, 0.74)), ("shot noise", (0.35, 0.65, 0.95)), ("read noise", (0.75, 0.75, 0.78)))
PART_KEYS = ("signal", "bleaching", "neuropil", "shot", "read")
REMOVALS = (("neuropil",), ("bleaching",), ("shot", "read"))
SIGMAS = (0.3, 1.0, 3.0)  # frames: as good as none, then the two smoothing chapters
X_LEFT, WIDTH = -6.5, 13.0


def row_axes(y: float, height: float, x0: float, x1: float, lo: float, hi: float, x_left: float = X_LEFT, width: float = WIDTH) -> mn.Axes:
    axes = plain_axes(x0, x1, lo, hi, width, height, visible=False)
    return axes.move_to(at(x_left + width / 2, y))


def row_label(text: str, c: tuple, axes: mn.Axes) -> mn.Text:
    return label(text, c, 17).next_to(axes, mn.UP, buff=0.02).align_to(axes, mn.LEFT)


def bars(axes: mn.Axes, t: np.ndarray, values: np.ndarray, dt: float, c: tuple, lo: float, hi: float) -> mn.VGroup:
    """One bar per frame, from 0 to its value."""
    out = mn.VGroup()
    for x, v in zip(t, values):
        a, b = axes_points(axes, np.array([x, x + dt]), np.array([0.0, float(np.clip(v, lo, hi))]))
        out.add(mn.Rectangle(width=max(b[0] - a[0], 0.01), height=abs(b[1] - a[1]), stroke_width=0.8, stroke_color=color(c),
                             fill_color=color(c), fill_opacity=0.35).move_to(0.5 * (a + b)))
    return out


def dff_running(f: np.ndarray, fs: float) -> np.ndarray:
    """dF/F against a running 5 s baseline, so bleaching does not drift it: what the smoothing chapters compare."""
    return sim.to_dff(f, sim.baseline(f, "running window", int(5 * fs)))


def event_average(x: np.ndarray, tr: sim.Trace, fs: float) -> tuple:
    return sim.event_average(x, tr.spikes, fs, 0.3, 1.0, 1.0)


def percent_label(x: float) -> mn.Text:
    return label(f"peak kept {x:.0f}%", ACCENT, 26)


def noise_label(x: float) -> mn.Text:
    return label(f"noise SD {x:.3f}", ACCENT, 26)


def build() -> kit.Timeline:
    p = sim.TraceParams(duration=30.0)
    tr = sim.make_trace(p)
    fs = p.fs
    ind = sim.BY_NAME[p.indicator]
    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)

    # chapter 1: one spike, top to bottom
    first = next(s for s in tr.spikes if s >= 0.5)
    t0, t1 = first - 0.5, first + 3.0
    fine = (tr.fine_t >= t0) & (tr.fine_t <= t1)
    frames = (tr.t >= t0) & (tr.t <= t1)
    f_rec = tr.recorded[frames]
    rows_y = (2.3, 1.0, -0.4, -1.9)
    axes_spike = row_axes(rows_y[0], 0.5, t0, t1, 0.0, 1.0)
    tick = mn.Line(*axes_points(axes_spike, np.array([first, first]), np.array([0.0, 1.0])), color=color(TRUTH), stroke_width=3.0)
    axes_calcium = row_axes(rows_y[1], 1.0, t0, t1, 0.0, float(tr.fine_signal[fine].max()) * 1.05)
    calcium = path(axes_calcium, tr.fine_t[fine], tr.fine_signal[fine], TRUTH, 2.0, 0.0, 1.0)
    axes_frames = row_axes(rows_y[2], 1.0, t0, t1, 0.0, float(tr.fine_signal[fine].max()) * 1.05)
    frame_bars = bars(axes_frames, tr.t[frames], tr.dff[frames], 1 / fs, TRUTH, 0.0, 1.0)
    axes_photons = row_axes(rows_y[3], 1.2, t0, t1, float(f_rec.min()) * 0.98, float(f_rec.max()) * 1.02)
    photons = path(axes_photons, tr.t[frames], f_rec, TRACE, 1.8, 0.0, 1e9)
    names_one = mn.VGroup(row_label("spike", TRUTH, axes_spike), row_label(f"{ind.name}: calcium dF/F", TRUTH, axes_calcium),
                          row_label(f"camera frames, {fs:.0f} Hz", TRUTH, axes_frames), row_label("photons counted per frame", TRACE, axes_photons))
    bar_one = hbar(axes_photons, t1 - 0.5, t1, float(f_rec.min()) * 0.98, "0.5 s", 0.12)
    bars_one = mn.VGroup(bar_one, *vbar(axes_photons, t1, float(f_rec.max()) * 1.02 - 100, float(f_rec.max()) * 1.02, "100 photons"))
    one_spike = [tick, calcium, frame_bars, photons, names_one, bars_one]

    # chapter 2: the whole recording, F with F0, then dF/F against the truth
    f = tr.recorded
    f0 = float(np.percentile(f, 20))
    dff = sim.to_dff(f, f0)
    f_lo, f_hi = float(f.min()) * 0.98, float(f.max()) * 1.02
    axes_f = row_axes(1.6, 2.0, 0.0, float(tr.t[-1]), f_lo, f_hi)
    path_f = path(axes_f, tr.t, f, TRACE, 1.3, f_lo, f_hi)
    line_f0 = mn.Line(*axes_points(axes_f, np.array([0.0, float(tr.t[-1])]), np.array([f0, f0])), color=color(KEY), stroke_width=2.5)
    name_f = row_label("F: photons per frame", TRACE, axes_f)
    name_f0 = label("F0: 20th percentile of F", KEY, 17).next_to(line_f0, mn.DOWN, buff=0.05).align_to(axes_f, mn.LEFT)
    bars_f = mn.VGroup(hbar(axes_f, float(tr.t[-1]) - 5.0, float(tr.t[-1]), f_lo, "5 s", 0.1), *vbar(axes_f, float(tr.t[-1]), f_hi - 200, f_hi, "200 photons"))
    d_lo, d_hi = float(min(dff.min(), tr.dff.min())) - 0.02, float(max(dff.max(), tr.dff.max())) + 0.02
    axes_d = row_axes(-1.6, 2.0, 0.0, float(tr.t[-1]), d_lo, d_hi)
    path_d = path(axes_d, tr.t, dff, TRACE, 1.3, d_lo, d_hi)
    path_truth = path(axes_d, tr.t, tr.dff, TRUTH, 1.2, d_lo, d_hi)
    name_d = mn.VGroup(label("dF/F", TRACE, 17), label("true dF/F (simulated)", TRUTH, 17)).arrange(mn.RIGHT, buff=0.4)
    name_d.next_to(axes_d, mn.UP, buff=0.02).align_to(axes_d, mn.LEFT)
    bars_d = mn.VGroup(hbar(axes_d, float(tr.t[-1]) - 5.0, float(tr.t[-1]), d_lo, "5 s", 0.1), *vbar(axes_d, float(tr.t[-1]), d_hi - 0.2, d_hi, "20% dF/F"))
    whole = [path_f, line_f0, name_f, name_f0, bars_f, path_d, path_truth, name_d, bars_d]

    # chapters 3 to 6: the parts of F, and F as parts leave
    pitch, height = 0.92, 0.7
    part_axes, part_paths, part_names = [], [], []
    for k, ((name, c), key) in enumerate(zip(PARTS, PART_KEYS)):
        values = tr.parts[key]
        lo, hi = float(values.min()), float(values.max())
        pad = 0.1 * (hi - lo) or 1.0
        axes = row_axes(2.35 - k * pitch, height, 0.0, float(tr.t[-1]), lo - pad, hi + pad)
        part_axes.append(axes)
        part_paths.append(path(axes, tr.t, values, c, 1.2, lo - pad, hi + pad))
        part_names.append(row_label(name, c, axes))
    sums = {"all": f}
    on = set(PART_KEYS)
    for removal in REMOVALS:
        on -= set(removal)
        sums[removal[0]] = sum(tr.parts[k] for k in on)
    axes_sum = row_axes(2.35 - len(PARTS) * pitch - 0.25, height + 0.2, 0.0, float(tr.t[-1]), f_lo, f_hi)
    path_sum = path(axes_sum, tr.t, f, TRACE, 1.3, f_lo, f_hi)
    paths_sum = {name: path(axes_sum, tr.t, values, TRACE, 1.3, f_lo, f_hi) for name, values in sums.items()}
    name_sum = row_label("F recorded", TRACE, axes_sum)
    bars_sum = mn.VGroup(hbar(axes_sum, float(tr.t[-1]) - 5.0, float(tr.t[-1]), f_lo, "5 s", 0.1), *vbar(axes_sum, float(tr.t[-1]), f_hi - 200, f_hi, "200 photons"))
    plus = mn.VGroup(*[label("+", DIM, 22).move_to(at(X_LEFT - 0.35, 2.35 - (k + 0.5) * pitch)) for k in range(len(PARTS) - 1)],
                     label("=", DIM, 22).move_to(at(X_LEFT - 0.35, 2.35 - (len(PARTS) - 0.5) * pitch - 0.12)))
    parts_all = [*part_paths, *part_names, path_sum, name_sum, bars_sum, plus]

    # chapters 7 and 8: smoothing, the whole dF/F and the average event
    dff_run = dff_running(f, fs)
    smooth = {sigma: sim.denoise(dff_run, "gaussian", sigma) for sigma in SIGMAS}
    axes_s = row_axes(1.9, 1.8, 0.0, float(tr.t[-1]), d_lo, d_hi)
    path_raw = path(axes_s, tr.t, dff_run, TRACE, 1.2, d_lo, d_hi)
    paths_smooth = {sigma: path(axes_s, tr.t, values, ACCENT, 1.8, d_lo, d_hi) for sigma, values in smooth.items()}
    name_s = mn.VGroup(label("dF/F", TRACE, 17), label("smoothed", ACCENT, 17)).arrange(mn.RIGHT, buff=0.4)
    name_s.next_to(axes_s, mn.UP, buff=0.02).align_to(axes_s, mn.LEFT)
    bars_s = mn.VGroup(hbar(axes_s, float(tr.t[-1]) - 5.0, float(tr.t[-1]), d_lo, "5 s", 0.1), *vbar(axes_s, float(tr.t[-1]), d_hi - 0.2, d_hi, "20% dF/F"))
    lag, ev_truth = event_average(tr.dff, tr, fs)
    _, ev_raw = event_average(dff_run, tr, fs)
    ev_smooth = {sigma: event_average(values, tr, fs)[1] for sigma, values in smooth.items()}
    e_lo, e_hi = float(min(ev_raw.min(), ev_truth.min())) - 0.02, float(max(ev_raw.max(), ev_truth.max())) + 0.02
    axes_e = row_axes(-1.6, 2.4, float(lag[0]), float(lag[-1]), e_lo, e_hi, x_left=-6.5, width=6.0)
    path_e_truth = path(axes_e, lag, ev_truth, TRUTH, 2.0, e_lo, e_hi)
    path_e_raw = path(axes_e, lag, ev_raw, TRACE, 1.3, e_lo, e_hi)
    paths_e_smooth = {sigma: path(axes_e, lag, values, ACCENT, 2.0, e_lo, e_hi) for sigma, values in ev_smooth.items()}
    name_e = mn.VGroup(label("average isolated event", DIM, 17), label("true", TRUTH, 17)).arrange(mn.RIGHT, buff=0.4)
    name_e.next_to(axes_e, mn.UP, buff=0.02).align_to(axes_e, mn.LEFT)
    bars_e = mn.VGroup(hbar(axes_e, float(lag[-1]) - 0.5, float(lag[-1]), e_lo, "0.5 s", 0.1), *vbar(axes_e, float(lag[-1]), e_hi - 0.1, e_hi, "10% dF/F"))
    spike_e = mn.Line(*axes_points(axes_e, np.array([0.0, 0.0]), np.array([e_lo, e_hi])), color=color(TRUTH), stroke_width=1.0, stroke_opacity=0.5)
    i_peak = int(np.argmax(ev_raw))
    kept = {sigma: 100 * float(values[i_peak]) / float(ev_raw[i_peak]) for sigma, values in ev_smooth.items()}
    quiet = tr.dff < 0.01  # frames with no response: what varies there is noise
    noise = {sigma: float(np.std(values[quiet])) for sigma, values in smooth.items()}
    peak_text = percent_label(kept[SIGMAS[0]]).move_to(at(1.2, -1.0), aligned_edge=mn.LEFT)
    noise_text = noise_label(noise[SIGMAS[0]]).move_to(at(1.2, -1.9), aligned_edge=mn.LEFT)
    smoothing = [path_raw, paths_smooth[SIGMAS[0]], name_s, bars_s, path_e_truth, path_e_raw, paths_e_smooth[SIGMAS[0]], name_e, bars_e, spike_e, peak_text, noise_text]

    seconds = [d for _, d in CHAPTERS]
    chapters = [
        chapter(seconds[0], mn.FadeIn(title),
                mn.AnimationGroup(mn.FadeIn(names_one[0]), mn.FadeIn(tick)),
                mn.AnimationGroup(mn.FadeIn(names_one[1]), mn.Create(calcium)),
                mn.AnimationGroup(mn.FadeIn(names_one[2]), mn.FadeIn(frame_bars, lag_ratio=0.02)),
                mn.AnimationGroup(mn.FadeIn(names_one[3]), mn.Create(photons), mn.FadeIn(bars_one)), lag_ratio=0.35),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), *[mn.FadeOut(m) for m in one_spike], mn.FadeIn(titles[1]), run_time=1.0),
                mn.AnimationGroup(mn.Create(path_f), mn.FadeIn(name_f), mn.FadeIn(bars_f), run_time=1.2),
                mn.AnimationGroup(mn.Create(line_f0), mn.FadeIn(name_f0), run_time=0.8),
                mn.AnimationGroup(mn.Create(path_d), mn.FadeIn(name_d[0]), mn.FadeIn(bars_d), run_time=1.2),
                mn.AnimationGroup(mn.Create(path_truth), mn.FadeIn(name_d[1]), run_time=1.0), lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(titles[1], titles[2]), *[mn.FadeOut(m) for m in whole], run_time=1.0),
                mn.LaggedStart(*[mn.AnimationGroup(mn.Create(part_paths[k]), mn.FadeIn(part_names[k]), *([mn.FadeIn(plus[k])] if k < len(PARTS) - 1 else []))
                                 for k in range(len(PARTS))], lag_ratio=0.5),
                mn.AnimationGroup(mn.FadeIn(plus[-1]), mn.Create(path_sum), mn.FadeIn(name_sum), mn.FadeIn(bars_sum)), lag_ratio=1.0),
    ]
    for k, removal in enumerate(REMOVALS):
        gone = [part_paths[PART_KEYS.index(key)].animate.set_stroke(opacity=0.15) for key in removal]
        gone += [part_names[PART_KEYS.index(key)].animate.set_fill(opacity=0.15) for key in removal]
        chapters.append(chapter(seconds[3 + k],
                                mn.AnimationGroup(*replace(titles[2 + k], titles[3 + k]), run_time=1.0),
                                mn.AnimationGroup(*gone,
                                                  mn.Transform(path_sum, paths_sum[removal[0]]), run_time=2.5), lag_ratio=1.0))
    chapters.append(chapter(seconds[6],
                            mn.AnimationGroup(*replace(titles[5], titles[6]), *[mn.FadeOut(m) for m in parts_all], run_time=1.0),
                            mn.AnimationGroup(mn.Create(path_raw), mn.FadeIn(name_s[0]), mn.FadeIn(bars_s), mn.Create(path_e_truth), mn.Create(path_e_raw),
                                              mn.FadeIn(name_e), mn.FadeIn(bars_e), mn.FadeIn(spike_e), run_time=1.2),
                            mn.AnimationGroup(mn.Create(paths_smooth[SIGMAS[0]]), mn.FadeIn(name_s[1]), mn.Create(paths_e_smooth[SIGMAS[0]]),
                                              mn.FadeIn(peak_text), mn.FadeIn(noise_text), run_time=0.8),
                            mn.AnimationGroup(mn.Transform(paths_smooth[SIGMAS[0]], paths_smooth[SIGMAS[1]]),
                                              mn.Transform(paths_e_smooth[SIGMAS[0]], paths_e_smooth[SIGMAS[1]]),
                                              Count(peak_text, kept[SIGMAS[0]], kept[SIGMAS[1]], percent_label, 0),
                                              Count(noise_text, noise[SIGMAS[0]], noise[SIGMAS[1]], noise_label, 3), run_time=2.0), lag_ratio=1.0))
    chapters.append(chapter(seconds[7],
                            mn.AnimationGroup(*replace(titles[6], titles[7]), run_time=1.0),
                            mn.AnimationGroup(mn.Transform(paths_smooth[SIGMAS[0]], paths_smooth[SIGMAS[2]]),
                                              mn.Transform(paths_e_smooth[SIGMAS[0]], paths_e_smooth[SIGMAS[2]]),
                                              Count(peak_text, kept[SIGMAS[1]], kept[SIGMAS[2]], percent_label, 0),
                                              Count(noise_text, noise[SIGMAS[1]], noise[SIGMAS[2]], noise_label, 3), run_time=3.0), lag_ratio=1.0))
    print(f"spike at {first:.2f} s; F0 {f0:.0f} photons; peak kept {kept}; noise {noise}")
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("traces", CHAPTERS, UNTITLED, build, __doc__)
