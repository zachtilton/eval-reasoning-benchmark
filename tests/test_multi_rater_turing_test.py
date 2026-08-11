"""
Unit tests for reliability/multi_rater_turing_test.py.

Covers the two end-to-end scenarios the equivalence test is meant to
distinguish:
- A candidate that behaves like a third correlated human rater (drawn from
  the same latent-signal generative process) should pass (EQUIVALENT).
- A candidate that rates uniformly at random, independent of the latent
  signal the human raters agree on, should fail (NOT EQUIVALENT).
"""

from __future__ import annotations

import numpy as np

from reliability.multi_rater_turing_test import run_turing_test


def _make_correlated_raters(n_items: int, n_raters: int, flip_prob: float, seed: int):
    """
    Simulate `n_raters` raters of a shared binary latent "true quality"
    signal, each with independent noise: a rater flips their label away
    from the latent value with probability `flip_prob`.
    """
    rng = np.random.default_rng(seed)
    latent = rng.integers(0, 2, size=n_items)
    raters = []
    for _ in range(n_raters):
        flips = rng.random(n_items) < flip_prob
        raters.append(np.where(flips, 1 - latent, latent))
    return raters


class TestRunTuringTestPeerLikeCandidate:
    def test_correlated_anchor_passes_as_equivalent(self):
        # 3 raters, all drawn from the same latent-signal + noise process --
        # the anchor is statistically just another peer rater.
        rater_1, rater_2, anchor = _make_correlated_raters(
            n_items=300, n_raters=3, flip_prob=0.1, seed=1
        )
        result = run_turing_test(
            human_ratings={"rater_1": rater_1, "rater_2": rater_2},
            candidate_ratings=anchor,
            n_boot=1000,
            seed=7,
        )
        assert result["decision"] == "EQUIVALENT (passes)"
        assert result["delta_kappa_5th_percentile"] > -result["margin"]


class TestRunTuringTestRandomNoiseCandidate:
    def test_random_anchor_fails_as_not_equivalent(self):
        # 2 correlated human raters, plus an anchor that rates uniformly
        # at random -- no relationship to the latent signal the humans agree on.
        rater_1, rater_2 = _make_correlated_raters(
            n_items=300, n_raters=2, flip_prob=0.1, seed=2
        )
        anchor = np.random.default_rng(3).integers(0, 2, size=300)

        result = run_turing_test(
            human_ratings={"rater_1": rater_1, "rater_2": rater_2},
            candidate_ratings=anchor,
            n_boot=1000,
            seed=7,
        )
        assert result["decision"] == "NOT EQUIVALENT (fails)"
        assert result["delta_kappa_5th_percentile"] <= -result["margin"]
