import numpy as np

from imgui_ophys.lessons.nmf.model import N_TRUE, NMF, MovieParams, cosine, make_movie, match_to_truth, pca, rel_error


def fit(movie, k=N_TRUE, iters=150, seed=0):
    nmf = NMF(movie.Y, k, seed=seed)
    for _ in range(iters):
        nmf.step()
    return nmf


def test_movie_is_non_negative_and_shaped():
    movie = make_movie()
    assert movie.Y.shape == (movie.side**2, 300)
    assert movie.A.shape == (movie.side**2, N_TRUE) and movie.C.shape == (N_TRUE, 300)
    assert movie.Y.min() >= 0


def test_nmf_stays_non_negative_and_error_falls():
    movie = make_movie()
    nmf = NMF(movie.Y, N_TRUE)
    errors = []
    for _ in range(60):
        nmf.step()
        errors.append(nmf.rel_error())
    assert nmf.A.min() >= 0 and nmf.C.min() >= 0
    assert np.allclose(nmf.A.max(axis=0), 1)
    assert all(b <= a + 1e-9 for a, b in zip(errors, errors[1:]))


def test_nmf_finds_the_true_footprints():
    movie = make_movie()
    nmf = fit(movie)
    A, C = match_to_truth(nmf.A, nmf.C, movie.A)
    # each cell footprint soaks up a little background, the overlapped middle cell most
    assert np.diag(cosine(movie.A, A)).min() > 0.8
    # no better than the truth by more than the noise it can fit
    assert rel_error(movie.Y, A, C) < rel_error(movie.Y, movie.A, movie.C) + 0.01


def test_match_keeps_every_component_for_any_k():
    movie = make_movie()
    for k in (2, 6):
        nmf = fit(movie, k=k, iters=20)
        A, C = match_to_truth(nmf.A, nmf.C, movie.A)
        assert A.shape[1] == C.shape[0] == k
        assert sorted(map(tuple, A.T.round(12))) == sorted(map(tuple, nmf.A.T.round(12)))


def test_pca_rebuilds_the_movie_and_goes_negative():
    movie = make_movie(MovieParams(noise=0.0))
    A, C, mu = pca(movie.Y, 8)
    assert rel_error(movie.Y, A, C, mu) < 0.02
    assert np.allclose(np.abs(A).max(axis=0), 1)
    assert (A < -0.05).any()


def test_tiny_movie_is_uint16_and_rank_three():
    from imgui_ophys.lessons.nmf.model import TINY_FRAMES, TINY_SIDE, make_tiny

    movie = make_tiny()
    assert movie.Y.dtype == np.uint16 and movie.Y.shape == (TINY_SIDE**2, TINY_FRAMES)
    assert rel_error(movie.Y.astype(float), movie.A, movie.C) < 0.05


def test_hals_walk_sweep_equals_nmf_step():
    from imgui_ophys.lessons.nmf.model import HalsWalk, make_tiny

    Y = make_tiny().Y.astype(float)
    walk, nmf = HalsWalk(Y, 3, seed=4), NMF(Y, 3, seed=4)
    for _ in range(5):
        walk.advance_sweep()
        nmf.step()
    assert np.allclose(walk.A, nmf.A) and np.allclose(walk.C, nmf.C)
    assert len(walk.errors) == 6 and walk.errors[-1] < walk.errors[0]
