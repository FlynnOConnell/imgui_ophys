"""
The NMF lesson's numbers, no GUI: a small synthetic imaging movie with known parts, rank-K NMF fitted by HALS one
iteration at a time, PCA for contrast, and matching fitted parts to the true ones.

Shapes follow the demixing literature: the movie Y is (pixels, frames), footprints A are (pixels, K), traces C are
(K, frames), and Y ~ A C.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

SIDE = 24  # the image is SIDE x SIDE pixels
N_TRUE = 4  # 3 cells + 1 background

# (x, y, radius) of each cell; the second overlaps both others
CELLS = ((8, 10, 2.6), (13, 13, 2.6), (17, 6, 2.4))
SPIKES = ((30, 95, 98, 101, 180, 240), (60, 130, 200, 203, 270), (45, 150, 220))
DRIFT_PERIODS = (120, 80, 200)
KERNEL = np.array([1.0, 0.5, 0.22, 0.08])  # a fast calcium transient, in frames


@dataclass
class MovieParams:
    frames: int = 300
    noise: float = 0.12  # SD of the white noise added to every pixel
    bleach: float = 0.5  # how much brighter the background starts than it ends
    seed: int = 7


@dataclass
class Movie:
    Y: np.ndarray  # (pixels, frames), clipped at 0 like photon counts
    A: np.ndarray  # true footprints (pixels, N_TRUE)
    C: np.ndarray  # true traces (N_TRUE, frames)
    side: int = SIDE

    @property
    def frames(self) -> int:
        return self.Y.shape[1]

    def image(self, column: np.ndarray) -> np.ndarray:
        """One (pixels,) column as a (side, side) image."""
        return column.reshape(self.side, self.side)


def make_movie(params: MovieParams = MovieParams()) -> Movie:
    """Three overlapping cells on an uneven, slowly bleaching background, plus noise."""
    rng = np.random.default_rng(params.seed)
    T = params.frames
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]

    A = np.zeros((SIDE * SIDE, N_TRUE))
    for k, (cx, cy, r) in enumerate(CELLS):
        A[:, k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2)).ravel()
    A[:, 3] = (0.45 * (0.4 + 0.6 * xx / SIDE) * (0.6 + 0.4 * yy / SIDE)).ravel()

    t = np.arange(T)
    C = np.zeros((N_TRUE, T))
    for k in range(3):
        c = 0.12 + 0.1 * (1 + np.sin(2 * np.pi * t / DRIFT_PERIODS[k] + k)) / 2  # slow drift
        if k == 2:
            c[(t > 100) & (t < 160)] += 0.3  # a plateau
        for s in SPIKES[k]:
            if s < T:
                n = min(len(KERNEL), T - s)
                c[s : s + n] += KERNEL[:n]
        C[k] = c
    C[3] = 0.8 + params.bleach * np.exp(-t / 120)  # bleaching background

    Y = np.maximum(0, A @ C + params.noise * rng.standard_normal((SIDE * SIDE, T)))
    return Movie(Y, A, C)


class NMF:
    """
    Rank-k NMF fitted with HALS (hierarchical alternating least squares).

    Minimizes ||Y - A C||^2 with A >= 0 and C >= 0, updating one component at a time with the others held fixed.
    Each footprint is rescaled to a max of 1 so traces carry the amplitude.
    """

    def __init__(self, Y: np.ndarray, k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Y = Y
        self.A = rng.random((Y.shape[0], k))
        self.C = rng.random((k, Y.shape[1]))
        self.iter = 0
        self.norm_Y = float(np.linalg.norm(Y))

    def step(self) -> None:
        Y, A, C = self.Y, self.A, self.C
        AtY, AtA = A.T @ Y, A.T @ A
        for j in range(A.shape[1]):
            C[j] = np.maximum(1e-9, C[j] + (AtY[j] - AtA[j] @ C) / (AtA[j, j] + 1e-9))
        YCt, CCt = Y @ C.T, C @ C.T
        for j in range(A.shape[1]):
            A[:, j] = np.maximum(0, A[:, j] + (YCt[:, j] - A @ CCt[:, j]) / (CCt[j, j] + 1e-9))
        m = A.max(axis=0)
        m[m == 0] = 1
        A /= m
        C *= m[:, None]
        self.iter += 1

    def rel_error(self) -> float:
        return float(np.linalg.norm(self.Y - self.A @ self.C)) / self.norm_Y


def rel_error(Y: np.ndarray, A: np.ndarray, C: np.ndarray, offset: np.ndarray | float = 0.0) -> float:
    """||Y - (A C + offset)|| / ||Y||: the share of the movie the parts leave unexplained."""
    return float(np.linalg.norm(Y - (A @ C + offset)) / np.linalg.norm(Y))


def cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between every column of A and every column of B, (A cols, B cols)."""

    def unit(M):
        return M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)

    return unit(A).T @ unit(B)


def match_to_truth(A: np.ndarray, C: np.ndarray, A_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorder components so each true footprint, in order, gets its closest unclaimed fitted one. With more fitted
    components than true ones the leftovers go last; with fewer, the later true parts go unmatched.
    """
    sim = cosine(A_true, A)
    order, left = [], list(range(A.shape[1]))
    for j in range(A_true.shape[1]):
        if not left:
            break
        best = max(left, key=lambda k: sim[j, k])
        left.remove(best)
        order.append(best)
    order += left
    return A[:, order], C[order]


def pca(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Top-k PCA of the mean-subtracted movie: (A, C, mean), Y ~ A C + mean. Each footprint's strongest pixel is +1,
    so its sign and scale move into the trace.
    """
    mu = Y.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    A = U[:, :k].copy()
    C = S[:k, None] * Vt[:k]
    for j in range(A.shape[1]):
        i = np.argmax(np.abs(A[:, j]))
        s = A[i, j]
        A[:, j] /= s
        C[j] *= s
    return A, C, mu


# ----------------------------------------------------------------------
# the guide's worked example: small enough to draw every value
# ----------------------------------------------------------------------

TINY_SIDE = 6
TINY_FRAMES = 24
TINY_NAMES = ("cell 1", "cell 2", "background")


def make_tiny(seed: int = 3) -> Movie:
    """
    A 6 x 6 movie of 24 frames as a camera would save it: uint16 photon counts. Two overlapping cells with calcium
    transients and a dim background that bleaches.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:TINY_SIDE, 0:TINY_SIDE]
    A = np.zeros((TINY_SIDE * TINY_SIDE, 3))
    for k, (cx, cy, r) in enumerate(((1.6, 1.8, 1.0), (3.6, 3.4, 1.15))):
        A[:, k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2)).ravel()
    A[:, 2] = (0.7 + 0.3 * xx / TINY_SIDE).ravel()
    A /= A.max(axis=0)
    t = np.arange(TINY_FRAMES)
    C = np.zeros((3, TINY_FRAMES))
    for k, spikes in enumerate(((3, 14), (8, 9, 19))):
        for s in spikes:
            C[k, s:] += 400 * np.exp(-(t[s:] - s) / 2.5)
    C[2] = 120 + 30 * np.exp(-t / 10)
    Y = np.clip(np.round(A @ C + 8 * rng.standard_normal((TINY_SIDE * TINY_SIDE, TINY_FRAMES))), 0, 65535).astype(np.uint16)
    return Movie(Y, A, C, TINY_SIDE)


class HalsWalk:
    """
    HALS one update at a time, for the guide: each sweep updates trace 1..K, then footprint 1..K, then rescales.
    Each update solves for one part with every other part held fixed:

        R_k = Y - sum_{j != k} a_j c_j          what part k has to explain
        c_k = max(0, a_k' R_k / a_k' a_k)       its best trace for its footprint
        a_k = max(0, R_k c_k' / c_k c_k')       its best footprint for its trace

    The same arithmetic as NMF.step, so a sweep here equals one NMF.step from the same start.
    """

    def __init__(self, Y: np.ndarray, k: int, seed: int = 0):
        self.Y = Y.astype(float)
        start = NMF(self.Y, k, seed=seed)
        self.A, self.C = start.A, start.C
        self.k = k
        self.pos = 0
        self.errors = [self.error()]
        self.residual = self.Y - self.A @ self.C  # R_k of the last update
        self.previous = None  # the updated row or column before the last update

    @property
    def steps(self) -> tuple:
        return tuple(("trace", j) for j in range(self.k)) + tuple(("footprint", j) for j in range(self.k)) + (("scale", -1),)

    @property
    def sweep(self) -> int:
        return self.pos // len(self.steps)

    def next(self) -> tuple[str, int]:
        return self.steps[self.pos % len(self.steps)]

    def last(self) -> Optional[tuple[str, int]]:
        return self.steps[(self.pos - 1) % len(self.steps)] if self.pos else None

    def error(self) -> float:
        return rel_error(self.Y, self.A, self.C)

    def advance(self) -> None:
        kind, j = self.next()
        A, C = self.A, self.C
        if kind == "trace":
            R = self.Y - A @ C + np.outer(A[:, j], C[j])
            self.previous = C[j].copy()
            C[j] = np.maximum(1e-9, (A[:, j] @ R) / (A[:, j] @ A[:, j] + 1e-9))
            self.residual = R
        elif kind == "footprint":
            R = self.Y - A @ C + np.outer(A[:, j], C[j])
            self.previous = A[:, j].copy()
            A[:, j] = np.maximum(0, (R @ C[j]) / (C[j] @ C[j] + 1e-9))
            self.residual = R
        else:
            self.previous = None
            m = A.max(axis=0)
            m[m == 0] = 1
            A /= m
            C *= m[:, None]
        self.pos += 1
        if kind == "scale":
            self.errors.append(self.error())

    def advance_sweep(self) -> None:
        self.advance()
        while self.pos % len(self.steps):
            self.advance()
