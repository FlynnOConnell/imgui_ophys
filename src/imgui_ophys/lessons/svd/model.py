"""
The SVD page's numbers: a small image to take apart value by value, and a larger one to rebuild at any rank.

Y = U S V' = sum_k s_k u_k v_k': each term is one column u_k times one row v_k', scaled by s_k. The s_k are sorted
largest first, and keeping the first k terms is the closest any rank-k matrix gets to Y.
"""

from dataclasses import dataclass

import numpy as np

from imgui_ophys.lessons.motion.model import field

SMALL = 8


@dataclass
class Svd:
    Y: np.ndarray
    U: np.ndarray  # (rows, r) columns of unit length, at right angles to each other
    S: np.ndarray  # (r,) largest first
    Vt: np.ndarray  # (r, cols) rows of unit length, at right angles to each other

    def piece(self, k: int) -> np.ndarray:
        """Term k: s_k u_k v_k', a rank-1 matrix."""
        return self.S[k] * np.outer(self.U[:, k], self.Vt[k])

    def rank(self, k: int) -> np.ndarray:
        """The sum of the first k terms."""
        return (self.U[:, :k] * self.S[:k]) @ self.Vt[:k]

    def error(self, k: int) -> float:
        """||Y - rank k|| / ||Y||, which is sqrt(sum of the dropped s^2) / sqrt(sum of all s^2)."""
        return float(np.sqrt(np.sum(self.S[k:] ** 2) / np.sum(self.S**2)))


def svd(Y: np.ndarray) -> Svd:
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    # sign is arbitrary: make each column's largest entry positive, so pictures read the same every run
    flip = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])])
    flip[flip == 0] = 1
    return Svd(Y, U * flip, S, Vt * flip[:, None])


def small_image() -> np.ndarray:
    """An 8 x 8 corner of the cell field, values rounded to one decimal so every one can be printed."""
    return np.round(field()[4 : 4 + SMALL, 10 : 10 + SMALL], 1)


def image() -> np.ndarray:
    return field()


class Peel:
    """
    How Y is split, one piece at a time, as the guide steps through it:

    1. guess a row v (unit length);
    2. u = R v: each row of the leftover R scored against v (how much that row looks like v), scaled to unit length;
    3. v = R' u: each column scored against u, scaled to unit length; s = how long R' u was;
    4. repeat 2 and 3 until v stops changing: u, s, v are the strongest (column x row) pattern left in R;
    5. subtract s u v' from R, and start again on what is left for the next piece.

    The pieces found this way are the SVD's, in the same order (tested against np.linalg.svd).
    """

    MAX_ITERATIONS = 60
    TOLERANCE = 1e-6

    def __init__(self, Y: np.ndarray):
        self.Y = Y.astype(float)
        self.R = self.Y.copy()
        self.pieces: list[tuple[float, np.ndarray, np.ndarray]] = []  # (s, u, v) found so far
        self.start_piece()

    def start_piece(self) -> None:
        n = self.Y.shape[1]
        v = 1.0 + 0.1 * np.arange(n)  # a flat row with a slight tilt, so it is never at right angles to the answer
        self.v = v / np.linalg.norm(v)
        self.u = np.zeros(self.Y.shape[0])
        self.s = 0.0
        self.next = "u"  # "u", "v" or "subtract"
        self.iterations = 0
        self.history: list[float] = []  # s after each iteration
        self.change = np.inf

    @property
    def done(self) -> bool:
        return len(self.pieces) >= min(self.Y.shape) or float(np.linalg.norm(self.R)) < 1e-9

    def piece(self) -> np.ndarray:
        """The piece as it stands: s u v'."""
        return self.s * np.outer(self.u, self.v)

    def step(self) -> str:
        """One half-step; returns what it did: "u", "v" or "subtract"."""
        did = self.next
        if did == "u":
            raw = self.R @ self.v
            self.s = float(np.linalg.norm(raw))
            self.u = raw / (self.s or 1.0)
            self.next = "v"
        elif did == "v":
            raw = self.R.T @ self.u
            self.s = float(np.linalg.norm(raw))
            previous = self.v
            self.v = raw / (self.s or 1.0)
            # a piece's sign is arbitrary: keep u's largest entry positive so pictures read the same every time
            i = int(np.argmax(np.abs(self.u)))
            if self.u[i] < 0:
                self.u, self.v = -self.u, -self.v
                previous = -previous
            self.change = float(np.linalg.norm(self.v - previous))
            self.iterations += 1
            self.history.append(self.s)
            self.next = "subtract" if self.change < self.TOLERANCE or self.iterations >= self.MAX_ITERATIONS else "u"
        else:
            self.R = self.R - self.piece()
            self.pieces.append((self.s, self.u.copy(), self.v.copy()))
            self.start_piece()
        return did

    def finish_piece(self) -> None:
        """Iterate until this piece settles, then subtract it."""
        while self.next != "subtract":
            self.step()
        self.step()
