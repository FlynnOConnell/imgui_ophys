import numpy as np

from imgui_ophys.lessons.svd import model as M


def test_pieces_add_up_to_the_matrix():
    d = M.svd(M.small_image())
    assert np.allclose(sum(d.piece(k) for k in range(len(d.S))), d.Y)
    assert np.allclose(d.rank(len(d.S)), d.Y)
    assert np.all(np.diff(d.S) <= 1e-12)


def test_u_and_v_are_orthonormal():
    d = M.svd(M.image())
    assert np.allclose(d.U.T @ d.U, np.eye(d.U.shape[1]), atol=1e-10)
    assert np.allclose(d.Vt @ d.Vt.T, np.eye(d.Vt.shape[0]), atol=1e-10)


def test_error_is_the_dropped_singular_values():
    d = M.svd(M.image())
    for k in (1, 5, 12):
        assert np.isclose(np.linalg.norm(d.Y - d.rank(k)) / np.linalg.norm(d.Y), d.error(k))


def test_peeling_finds_the_svd():
    Y = M.small_image()
    peel = M.Peel(Y)
    for _ in range(4):
        peel.finish_piece()
    U, S, Vt = np.linalg.svd(Y)
    for k, (s, u, v) in enumerate(peel.pieces):
        assert np.isclose(s, S[k], rtol=1e-4)
        assert abs(u @ U[:, k]) > 0.999 and abs(v @ Vt[k]) > 0.999
    # what is left is Y minus the pieces
    assert np.allclose(peel.R, Y - sum(s * np.outer(u, v) for s, u, v in peel.pieces))


def test_peel_steps_alternate_then_subtract():
    peel = M.Peel(M.small_image())
    assert [peel.step() for _ in range(4)] == ["u", "v", "u", "v"]
    peel.finish_piece()
    assert len(peel.pieces) == 1 and peel.next == "u"
