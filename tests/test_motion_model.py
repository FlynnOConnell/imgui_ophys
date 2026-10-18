import time

import numpy as np

from imgui_ophys.lessons.motion.model import (PATCH_CENTERS, field, find_shift, make_motion, mismatch, move, register,
                                              register_patches, shift, compute_reference, crispness,
                                              pick_initial_reference)


def test_correlation_finds_a_known_shift():
    base = field()
    for dy, dx in ((0, 0), (2, -3), (-4, 1)):
        assert find_shift(base, shift(base, dy, dx)) == (dy, dx)


def test_move_by_whole_pixels_matches_roll_inside():
    base = field()
    assert np.allclose(move(base, 2, -1)[4:-4, 4:-4], shift(base, 2, -1)[4:-4, 4:-4])


def test_rigid_registration_recovers_the_walk_between_pixels():
    m = make_motion(bend=0, noise=0.02)
    _, shifts = register(m.frames, field())
    assert np.abs(shifts - m.rigid).max() < 0.25


def test_patches_follow_the_bend_and_beat_rigid():
    m = make_motion(bend=1.0, noise=0.02)
    truth = field()
    rigid, _ = register(m.frames, truth)
    patched, shifts = register_patches(m.frames, truth)
    errors = [np.abs(shifts[t] - m.true_at(t, PATCH_CENTERS, PATCH_CENTERS)).mean() for t in range(len(m.frames))]
    assert np.mean(errors) < 0.35  # a patch holds one shift; the bend inside it is what is left
    assert np.mean([mismatch(p, truth) for p in patched]) < 0.85 * np.mean([mismatch(r, truth) for r in rigid])


def test_reference_is_crisper_than_the_raw_mean_and_sits_where_most_frames_are():
    m = make_motion(bend=0, noise=0.02)
    refs = compute_reference(m.frames, n=10)
    assert crispness(refs[-1]) > 1.3 * crispness(m.frames.mean(axis=0))
    _, shifts = register(refs[-1][None], field())
    assert np.abs(shifts[0] - np.median(m.rigid, axis=0)).max() < 1.0  # sits where most frames are


def test_seed_frames_include_the_seed():
    m = make_motion()
    _, seed, chosen = pick_initial_reference(m.frames, n=10)
    assert chosen[0] == seed and len(chosen) == 11


def test_fast_enough_to_recompute_on_a_click():
    start = time.perf_counter()
    m = make_motion()
    compute_reference(m.frames, n=10)
    register(m.frames, field())
    register_patches(m.frames, field())
    assert time.perf_counter() - start < 1.0
