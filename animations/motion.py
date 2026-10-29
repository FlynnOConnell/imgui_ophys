# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
Motion correction as one animation: a field of cells that moves, the reference image, the correlation that finds each
frame's shift, every frame moved back, patches for a field that bends, and how a reference is built. Built from
imgui_ophys's motion model (32 x 32 pixels, 60 frames); drawn with kit.py.

    uv run -p 3.12 animations/motion.py --out <dir> [--stills-only] [--gif-width 720]
"""

import manim as mn
import numpy as np

import kit
from kit import ACCENT, DIM, HIGHLIGHT, KEY, TEXT, TRACE, Play, at, axes_points, chapter, color, diverging_rgb, gray_rgb, hbar, image, label, path, plain_axes, put, replace, timeline, tinted_rgb, vbar

M = kit.lesson_model("motion")

TITLE = "Motion correction: registering every frame to a reference image"
CHAPTERS = (
    ("", 7.0),
    ("Rigid registration: the correlation peak is the shift", 8.0),
    ("Every frame moved back", 7.0),
    ("Non-rigid registration: 16 x 16 patches, overlap 8", 8.0),
    ("Building the reference image", 7.0),
)
UNTITLED = {0: "the-movie"}
FRAMES = 60
BEND = 2.0  # pixels, how far the bending field's control points wander
PASSES = 8
SIDE = 2.6  # scene units per image
GAIN = 2.5  # arrows are drawn this many times the shift


def arrows(points: np.ndarray, shifts: np.ndarray, img: mn.ImageMobject, c: tuple) -> mn.VGroup:
    """An arrow per (y, x) in points (pixels) showing (dy, dx) from shifts, GAIN times longer, on img."""
    scale = SIDE / M.SIDE
    corner = img.get_corner(mn.UL)
    out = mn.VGroup()
    for i, y in enumerate(points):
        for j, x in enumerate(points):
            dy, dx = shifts[i, j]
            start = corner + at((x + 0.5) * scale, -(y + 0.5) * scale)
            end = start + at(GAIN * dx * scale, -GAIN * dy * scale)
            if np.hypot(dx, dy) < 0.15:
                out.add(mn.Dot(start, radius=0.04, color=color(c)))
            else:
                out.add(mn.Arrow(start, end, buff=0, stroke_width=3.0, max_tip_length_to_length_ratio=0.35, color=color(c)))
    return out


def under(img: mn.Mobject, text: str, c: tuple = DIM) -> mn.Text:
    return label(text, c, 18).next_to(img, mn.DOWN, buff=0.12)


def build(seconds: list) -> kit.Timeline:
    ref = M.field()
    rigid_motion = M.make_motion(FRAMES, bend=0.0)
    bent_motion = M.make_motion(FRAMES, bend=BEND)
    registered, found = M.register(rigid_motion.frames, ref)
    hi = float(max(ref.max(), rigid_motion.frames.max()))
    diff_scale = float(np.abs(rigid_motion.frames - ref).max())
    t_star = int(np.argmax(np.abs(rigid_motion.rigid).sum(axis=1)))
    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)

    # chapter 1: the movie beside the reference, and the difference
    xs = (-4.4, 0.0, 4.4)
    y_img = 0.9
    movie = image(gray_rgb(rigid_motion.frames[0], 0, hi), SIDE, xs[0], y_img)
    movie_frames = [gray_rgb(f, 0, hi) for f in rigid_motion.frames]
    reference = image(gray_rgb(ref, 0, hi), SIDE, xs[1], y_img)
    diff = image(diverging_rgb(rigid_motion.frames[0] - ref, diff_scale, TEXT), SIDE, xs[2], y_img)
    diff_frames = [diverging_rgb(f - ref, diff_scale, TEXT) for f in rigid_motion.frames]
    frames_of = [kit.frame_of(m) for m in (movie, reference, diff)]
    names = mn.VGroup(under(movie, "recorded frames", HIGHLIGHT), under(reference, "reference image"), under(diff, "frame minus reference"))
    legend_diff = label("white brighter, pink dimmer", DIM, 15).next_to(names[2], mn.DOWN, buff=0.08)

    # chapter 2: one frame's correlation with the reference; the peak is the shift; the frame moves back
    frame_star = rigid_motion.frames[t_star]
    corr = M.correlation(ref, frame_star)
    peak = M.find_shift(ref, frame_star)
    corr_img = image(tinted_rgb(np.maximum(corr, 0), 0, float(corr.max()), ACCENT), SIDE, xs[1], -2.1)
    corr_frame = kit.frame_of(corr_img)
    scale = SIDE / M.SIDE
    center = corr_img.get_center() + at(0.5 * scale, -0.5 * scale)  # pixel (16, 16) is one past the middle
    origin_mark = mn.Circle(radius=0.08, color=color(DIM), stroke_width=1.5).move_to(center)
    peak_point = center + at(peak[1] * scale, -peak[0] * scale)
    peak_arrow = mn.Arrow(center, peak_point, buff=0, stroke_width=3.0, max_tip_length_to_length_ratio=0.3, color=color(HIGHLIGHT))
    corr_name = under(corr_img, "correlation at every offset", ACCENT)
    dy, dx = found[t_star]
    shift_text = label(f"shift {dy:+.1f}, {dx:+.1f} px", HIGHLIGHT, 20).next_to(corr_img, mn.RIGHT, buff=0.4)
    moved_back = image(gray_rgb(registered[t_star], 0, hi), SIDE, xs[0], y_img)
    diff_back = image(diverging_rgb(registered[t_star] - ref, diff_scale, TEXT), SIDE, xs[2], y_img)
    name_back = under(movie, "moved back", HIGHLIGHT)

    # chapter 3: every frame's shift, found against true, while the registered movie plays
    true = rigid_motion.rigid
    lim = float(max(np.abs(true).max(), np.abs(found).max())) * 1.15
    plots, plot_names, plot_bars = mn.VGroup(), mn.VGroup(), mn.VGroup()
    for k, (name, y) in enumerate((("dy", -1.55), ("dx", -2.75))):
        axes = plain_axes(0, FRAMES - 1, -lim, lim, 8.0, 0.95, visible=False).move_to(at(-2.4, y))
        plots.add(path(axes, np.arange(FRAMES), true[:, k], HIGHLIGHT, 2.5, -lim, lim), path(axes, np.arange(FRAMES), found[:, k], ACCENT, 1.5, -lim, lim))
        plot_names.add(label(name, DIM, 17).next_to(axes, mn.LEFT, buff=0.15))
        if k == 1:
            plot_bars.add(hbar(axes, FRAMES - 1 - 20, FRAMES - 1, -lim, "20 frames", 0.08))
        plot_bars.add(*vbar(axes, FRAMES + 3, -1.0, 1.0, "2 px"))
    plot_legend = mn.VGroup(label("true (simulated)", HIGHLIGHT, 17), label("found", ACCENT, 17)).arrange(mn.DOWN, buff=0.15, aligned_edge=mn.LEFT)
    plot_legend.move_to(at(3.0, -1.55), aligned_edge=mn.LEFT)
    error = float(np.abs(found - true).mean())
    error_text = label(f"found minus true: {error:.2f} px", DIM, 17).move_to(at(3.0, -2.75), aligned_edge=mn.LEFT)
    registered_frames = [gray_rgb(f, 0, hi) for f in registered]
    registered_diffs = [diverging_rgb(f - ref, diff_scale, TEXT) for f in registered]
    name_registered = under(movie, "registered frames", HIGHLIGHT)

    # chapter 4: a field that bends: one shift per frame against one per patch
    bent_rigid, bent_rigid_shifts = M.register(bent_motion.frames, ref)
    bent_patched, patch_shifts = M.register_patches(bent_motion.frames, ref)
    t_bend = int(np.argmax([M.mismatch(bent_rigid[t], ref) - M.mismatch(bent_patched[t], ref) for t in range(FRAMES)]))
    pc = M.PATCH_CENTERS
    bend_scale = float(np.abs(bent_motion.frames - ref).max())
    bent = image(gray_rgb(bent_motion.frames[t_bend], 0, hi), SIDE, xs[0], y_img)
    bent_arrows = arrows(pc, bent_motion.true_at(t_bend, pc, pc), bent, HIGHLIGHT)
    rigid_img = image(diverging_rgb(bent_rigid[t_bend] - ref, bend_scale, TEXT), SIDE, xs[1], y_img)
    rigid_arrows = arrows(np.array([(M.SIDE - 1) / 2]), bent_rigid_shifts[t_bend][None, None, :], rigid_img, ACCENT)
    patched_img = image(diverging_rgb(bent_patched[t_bend] - ref, bend_scale, TEXT), SIDE, xs[2], y_img)
    patched_arrows = arrows(pc, patch_shifts[t_bend], patched_img, ACCENT)
    patch_box = mn.Square(side_length=M.PATCH * scale, color=color(KEY), stroke_width=1.5).move_to(bent.get_corner(mn.UL) + at(M.PATCH * scale / 2, -M.PATCH * scale / 2))
    bend_names = mn.VGroup(under(bent, "true shifts (simulated)", HIGHLIGHT), under(rigid_img, "rigid: one shift"),
                           under(patched_img, f"piecewise rigid: {len(pc) ** 2} shifts"))
    mismatches = mn.VGroup(*[label(f"mean |frame - reference| {M.mismatch(img, ref):.3f}", KEY if k == 2 else DIM, 16).next_to(bend_names[k], mn.DOWN, buff=0.1)
                             for k, img in enumerate((bent_motion.frames[t_bend], bent_rigid[t_bend], bent_patched[t_bend]))])
    bend_all = [bent, bent_arrows, rigid_img, rigid_arrows, patched_img, patched_arrows, patch_box, bend_names, mismatches]

    # chapter 5: the reference built from the frames alone
    refs = M.compute_reference(rigid_motion.frames, iterations=PASSES)
    built = [rigid_motion.frames.mean(axis=0), refs[0], refs[-1]]
    built_imgs = [image(gray_rgb(img, 0, float(img.max())), SIDE, x, y_img) for img, x in zip(built, xs)]
    built_names = mn.VGroup(under(built_imgs[0], "mean of all frames"), under(built_imgs[1], "20 most alike frames, averaged"),
                            under(built_imgs[2], f"{PASSES} passes: register, average", KEY))
    crisp = mn.VGroup(*[label(f"crispness {M.crispness(img):.1f}", KEY if k == 2 else DIM, 17).next_to(built_names[k], mn.DOWN, buff=0.1) for k, img in enumerate(built)])
    built_all = [*built_imgs, built_names, crisp]

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(movie), mn.FadeIn(frames_of[0]), mn.FadeIn(names[0]), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(reference), mn.FadeIn(frames_of[1]), mn.FadeIn(names[1]), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(diff), mn.FadeIn(frames_of[2]), mn.FadeIn(names[2]), mn.FadeIn(legend_diff), run_time=0.8),
                mn.AnimationGroup(Play(movie, movie_frames), Play(diff, diff_frames), run_time=3.0, rate_func=mn.linear), lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), mn.FadeIn(titles[1]), Play(movie, movie_frames[: t_star + 1]), Play(diff, diff_frames[: t_star + 1]),
                                  run_time=1.0, rate_func=mn.linear),
                mn.AnimationGroup(mn.FadeIn(corr_img), mn.FadeIn(corr_frame), mn.FadeIn(corr_name), mn.FadeIn(origin_mark), run_time=1.0),
                mn.AnimationGroup(mn.GrowArrow(peak_arrow), mn.FadeIn(shift_text), run_time=0.8),
                mn.AnimationGroup(mn.Transform(movie, moved_back), mn.Transform(diff, diff_back), *replace(names[0], name_back), run_time=1.5), lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(titles[1], titles[2]), mn.FadeOut(corr_img), mn.FadeOut(corr_frame), mn.FadeOut(corr_name),
                                  mn.FadeOut(origin_mark), mn.FadeOut(peak_arrow), mn.FadeOut(shift_text), *replace(name_back, name_registered), run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(plot_legend), mn.FadeIn(plot_names), mn.FadeIn(plot_bars), mn.Create(plots), run_time=1.5),
                mn.AnimationGroup(mn.FadeIn(error_text), Play(movie, registered_frames), Play(diff, registered_diffs), run_time=3.0, rate_func=mn.linear), lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(titles[2], titles[3]), mn.FadeOut(plots), mn.FadeOut(plot_legend), mn.FadeOut(plot_names), mn.FadeOut(plot_bars),
                                  mn.FadeOut(error_text), mn.FadeOut(movie), mn.FadeOut(name_registered), mn.FadeOut(reference), mn.FadeOut(names[1]),
                                  mn.FadeOut(diff), mn.FadeOut(names[2]), mn.FadeOut(legend_diff), run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(bent), mn.FadeIn(bend_names[0]), mn.FadeIn(bent_arrows, lag_ratio=0.05), run_time=1.2),
                mn.AnimationGroup(mn.FadeIn(rigid_img), mn.FadeIn(bend_names[1]), mn.FadeIn(mismatches[0]), mn.FadeIn(mismatches[1]), mn.FadeIn(rigid_arrows), run_time=1.0),
                mn.AnimationGroup(mn.Create(patch_box), run_time=0.6),
                mn.AnimationGroup(mn.FadeIn(patched_img), mn.FadeIn(bend_names[2]), mn.FadeIn(mismatches[2]), mn.FadeIn(patched_arrows, lag_ratio=0.05), run_time=1.2), lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(titles[3], titles[4]), *[mn.FadeOut(m) for m in bend_all], *[mn.FadeOut(f) for f in frames_of], run_time=1.0),
                mn.LaggedStart(*[mn.AnimationGroup(mn.FadeIn(img), mn.FadeIn(name), mn.FadeIn(c)) for img, name, c in zip(built_imgs, built_names, crisp)], lag_ratio=0.6), lag_ratio=1.0),
    ]
    print(f"frame {t_star}: shift {found[t_star].round(2)} found, {true[t_star].round(2)} true; rigid error {error:.3f} px; bending frame {t_bend}: "
          f"mismatch {[round(M.mismatch(img, ref), 3) for img in (bent_motion.frames[t_bend], bent_rigid[t_bend], bent_patched[t_bend])]}; "
          f"crispness {[round(M.crispness(img), 2) for img in built]}")
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("motion", CHAPTERS, UNTITLED, build, __doc__)
