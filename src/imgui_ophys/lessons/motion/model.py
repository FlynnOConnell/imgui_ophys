"""
The Motion correction stage's numbers: a field of cells that moves as a whole (rigid) or bends smoothly (non-rigid),
registered to a reference image:

- rigid registration: one shift per frame, the peak of the frame's correlation with the reference;
- non-rigid (piecewise rigid, NoRMCorre) registration: a shift for each overlapping patch, found the same way, then
  interpolated into a shift for every pixel, and the frame resampled by it;
- the reference image as suite2p builds it (pick_initial_reference, compute_reference in suite2p/registration).

Shifts are in pixels, (dy, dx), and need not be whole: frames are sampled between pixels with bilinear interpolation.
Here the whole-pixel peak is refined with a parabola; suite2p and NoRMCorre upsample the correlation around it instead.
"""

from dataclasses import dataclass

import numpy as np

SIDE = 32
CONTROL = 3  # the true bending is set at a 3 x 3 grid of points and interpolated between them
PATCH = 16  # piecewise rigid: patches of 16 x 16 ...
STRIDE = 8  # ... every 8 pixels, so neighbours overlap by half
SEARCH = 6  # largest shift tried, pixels


def field(seed: int = 4) -> np.ndarray:
    """The still image: a dozen cells of different brightness on a dim, uneven background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    img = 0.15 + 0.1 * xx / SIDE
    for _ in range(12):
        cx, cy = rng.uniform(4, SIDE - 4, 2)
        img = img + rng.uniform(0.4, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(1.0, 1.8) ** 2))
    return img


def shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """img moved down dy and right dx whole pixels, wrapping at the edges (for the correlation figure)."""
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """img at (y, x), between pixels by bilinear interpolation, edges extended."""
    h, w = img.shape
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    y0 = np.minimum(np.floor(y).astype(int), h - 2)
    x0 = np.minimum(np.floor(x).astype(int), w - 2)
    fy, fx = y - y0, x - x0
    return (img[y0, x0] * (1 - fy) * (1 - fx) + img[y0 + 1, x0] * fy * (1 - fx)
            + img[y0, x0 + 1] * (1 - fy) * fx + img[y0 + 1, x0 + 1] * fy * fx)


GRID = np.mgrid[0:SIDE, 0:SIDE].astype(float)


def move(img: np.ndarray, dy, dx) -> np.ndarray:
    """img with its content moved by (dy, dx): a number each, or one per pixel."""
    return sample(img, GRID[0] - dy, GRID[1] - dx)


def interpolate(values: np.ndarray, centers: np.ndarray, size: int = SIDE) -> np.ndarray:
    """
    A per-pixel map from values (n, n) set at the points centers (n,) on both axes: bilinear between them, flat
    beyond the outermost ones.
    """
    p = np.arange(size, dtype=float)
    cols = np.array([np.interp(p, centers, values[:, j]) for j in range(values.shape[1])]).T  # (size, n)
    return np.array([np.interp(p, centers, row) for row in cols])


CONTROL_POINTS = np.linspace(0, SIDE - 1, CONTROL)
PATCH_STARTS = np.arange(0, SIDE - PATCH + 1, STRIDE)
PATCH_CENTERS = PATCH_STARTS + (PATCH - 1) / 2


@dataclass
class Motion:
    frames: np.ndarray  # (T, SIDE, SIDE)
    rigid: np.ndarray  # (T, 2) the whole field's (dy, dx)
    bend: np.ndarray  # (T, CONTROL, CONTROL, 2) the extra shift at each control point
    noise: float

    def shift_map(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """The true (dy, dx) of every pixel in frame t."""
        return (self.rigid[t, 0] + interpolate(self.bend[t, ..., 0], CONTROL_POINTS),
                self.rigid[t, 1] + interpolate(self.bend[t, ..., 1], CONTROL_POINTS))

    def true_at(self, t: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """The true (dy, dx) at the points (ys[i], xs[j]): (len(ys), len(xs), 2)."""
        dy, dx = self.shift_map(t)
        yi, xi = np.round(ys).astype(int), np.round(xs).astype(int)
        return np.stack([dy[np.ix_(yi, xi)], dx[np.ix_(yi, xi)]], axis=-1)


def make_motion(frames: int = 60, max_shift: float = 3.0, bend: float = 1.0, noise: float = 0.08, seed: int = 0) -> Motion:
    """
    The field moving: a slow random walk of whole-field shifts within max_shift, plus a smooth bend that also
    wanders, up to bend pixels at each control point (0 for purely rigid motion), plus noise.
    """
    rng = np.random.default_rng(seed)
    base = field()
    walk = np.clip(np.cumsum(rng.normal(0, 0.7, (frames, 2)), axis=0), -max_shift, max_shift)
    if bend:
        warp = np.clip(np.cumsum(rng.normal(0, 0.4, (frames, CONTROL, CONTROL, 2)), axis=0), -bend, bend)
    else:
        warp = np.zeros((frames, CONTROL, CONTROL, 2))
    m = Motion(np.empty((frames, SIDE, SIDE)), walk, warp, noise)
    for t in range(frames):
        dy, dx = m.shift_map(t)
        m.frames[t] = move(base, dy, dx)
    m.frames += noise * rng.standard_normal(m.frames.shape)
    return m


def correlation(ref: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    How well img matches ref at every whole-pixel offset, zero offset at the center: its peak sits at how far img
    moved. Computed with FFTs, so all offsets cost about as much as one.
    """
    a, b = img - img.mean(), ref - ref.mean()
    corr = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    return np.fft.fftshift(corr / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def find_shift(ref: np.ndarray, img: np.ndarray, max_shift: int = SEARCH) -> tuple[int, int]:
    """(dy, dx) img moved from ref, whole pixels: the correlation peak within max_shift of the center."""
    corr = correlation(ref, img)
    c = np.array(corr.shape) // 2
    win = corr[c[0] - max_shift : c[0] + max_shift + 1, c[1] - max_shift : c[1] + max_shift + 1]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    return int(dy - max_shift), int(dx - max_shift)


CANDIDATES = np.array([(dy, dx) for dy in range(-SEARCH, SEARCH + 1) for dx in range(-SEARCH, SEARCH + 1)], dtype=float)


def moved_templates(ref: np.ndarray) -> np.ndarray:
    """The template moved by every candidate shift: (candidates, SIDE, SIDE), made once per registration."""
    return np.stack([move(ref, dy, dx) for dy, dx in CANDIDATES])


def scores(frames: np.ndarray, moved: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    """(T, candidates): how well each frame's region matches the template moved by each candidate shift."""
    a = frames[:, rows, cols].reshape(len(frames), -1)
    b = moved[:, rows, cols].reshape(len(moved), -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + 1e-12)


def best(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (T, 2) best shift per frame, refined between pixels by fitting a parabola through the peak and its neighbours,
    and (T,) the peak correlation.
    """
    n = 2 * SEARCH + 1
    grid = score.reshape(len(score), n, n)
    out = np.empty((len(score), 2))
    for t, g in enumerate(grid):
        i, j = np.unravel_index(np.argmax(g), g.shape)
        sub = []
        for k, line in ((i, g[:, j]), (j, g[i, :])):
            off = 0.0
            if 0 < k < n - 1:
                a, b, c = line[k - 1], line[k], line[k + 1]
                d = a - 2 * b + c
                off = 0.5 * (a - c) / d if d < 0 else 0.0
            sub.append(k - SEARCH + off)
        out[t] = sub
    return out, score.max(axis=1)


def register(frames: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid: one shift per frame from the whole frame; (frames moved back, (T, 2) shifts)."""
    shifts, _ = best(scores(frames, moved_templates(ref), slice(None), slice(None)))
    return np.array([move(f, -dy, -dx) for f, (dy, dx) in zip(frames, shifts)]), shifts


def register_patches(frames: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise rigid: a shift for each overlapping patch, interpolated into a smooth shift for every pixel, each frame
    warped back by it. Returns (frames moved back, (T, n, n, 2) patch shifts at PATCH_CENTERS).
    """
    moved = moved_templates(ref)
    n = len(PATCH_STARTS)
    shifts = np.zeros((len(frames), n, n, 2))
    for i, r0 in enumerate(PATCH_STARTS):
        for j, c0 in enumerate(PATCH_STARTS):
            shifts[:, i, j], _ = best(scores(frames, moved, slice(r0, r0 + PATCH), slice(c0, c0 + PATCH)))
    out = np.empty_like(frames)
    for t, f in enumerate(frames):
        dy = interpolate(shifts[t, ..., 0], PATCH_CENTERS)
        dx = interpolate(shifts[t, ..., 1], PATCH_CENTERS)
        out[t] = sample(f, GRID[0] + dy, GRID[1] + dx)
    return out, shifts


def frame_correlations(frames: np.ndarray) -> np.ndarray:
    """(T, T) correlation of every frame with every other."""
    z = frames.reshape(len(frames), -1)
    z = z - z.mean(axis=1, keepdims=True)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return z @ z.T


def pick_initial_reference(frames: np.ndarray, n: int = 20) -> tuple[np.ndarray, int, np.ndarray]:
    """
    suite2p's seed: the frame whose n most-correlated other frames correlate with it best on average, and the mean of
    it and those frames. Returns (reference, seed frame, the frames averaged).
    """
    cc = frame_correlations(frames)
    best_cc = -np.sort(-cc, axis=1)[:, 1 : n + 1].mean(axis=1)  # column 0 is the frame itself
    seed = int(np.argmax(best_cc))
    chosen = np.argsort(-cc[seed])[: n + 1]
    return frames[chosen].mean(axis=0), seed, chosen


def compute_reference(frames: np.ndarray, n: int = 20, iterations: int = 8) -> list[np.ndarray]:
    """
    suite2p's reference image: the initial reference, then iterations times: register every frame rigidly to it,
    average the best-matching (by peak correlation) frames, a larger share each time up to half, and move the result
    to their mean position. Returns the initial reference and each iteration's.
    """
    refs = [pick_initial_reference(frames, n)[0]]
    for k in range(iterations):
        moved = moved_templates(refs[-1])
        shifts, peaks = best(scores(frames, moved, slice(None), slice(None)))
        keep = np.argsort(-peaks)[: max(2, int(len(frames) * (1 + k) / (2 * iterations)))]
        ref = np.mean([move(frames[t], -dy, -dx) for t, (dy, dx) in zip(keep, shifts[keep])], axis=0)
        dy, dx = np.round(shifts[keep].mean(axis=0))
        refs.append(move(ref, dy, dx))
    return refs


def crispness(img: np.ndarray, margin: int = 4) -> float:
    """The norm of the image's gradient (CaImAn's crispness): a blurred image scores lower."""
    sl = slice(margin, SIDE - margin)
    gy, gx = np.gradient(img[sl, sl])
    return float(np.sqrt(np.sum(gy**2 + gx**2)))


def mismatch(img: np.ndarray, truth: np.ndarray, margin: int = 4) -> float:
    """Mean |img - truth| away from the edges, where shifted-in pixels are made up."""
    sl = slice(margin, SIDE - margin)
    return float(np.mean(np.abs(img[sl, sl] - truth[sl, sl])))
