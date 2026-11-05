"""Tests for the 2D Random Hierarchy Model (src.utils.random_hierarchy)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import (
    RandomHierarchyModel2D,
    RandomHierarchy2DDataset,
    sample_random_hierarchy_model_2d,
)


class TestRandomHierarchyModel2D:

    def test_shapes_and_ranges(self):
        model = RandomHierarchyModel2D(v=4, s=2, L=3, m=2, seed=0)
        grid, root = model.sample(5, seed=1)
        assert grid.shape == (5, 8, 8)
        assert root.shape == (5,)
        assert grid.dtype == np.int64
        assert grid.min() >= 0 and grid.max() < 4
        assert root.min() >= 0 and root.max() < 4
        assert model.grid_size == 8

    def test_reproducible_with_same_seed(self):
        m1 = RandomHierarchyModel2D(v=6, s=2, L=3, m=3, seed=42)
        m2 = RandomHierarchyModel2D(v=6, s=2, L=3, m=3, seed=42)
        for r1, r2 in zip(m1.rules, m2.rules):
            np.testing.assert_array_equal(r1, r2)

        grid1, root1 = m1.sample(10, seed=7)
        grid2, root2 = m2.sample(10, seed=7)
        np.testing.assert_array_equal(grid1, grid2)
        np.testing.assert_array_equal(root1, root2)

    def test_different_seeds_give_different_rules(self):
        m1 = RandomHierarchyModel2D(v=6, s=2, L=3, m=3, seed=1)
        m2 = RandomHierarchyModel2D(v=6, s=2, L=3, m=3, seed=2)
        assert any(not np.array_equal(r1, r2) for r1, r2 in zip(m1.rules, m2.rules))

    def test_different_sample_seeds_give_different_grids(self):
        model = RandomHierarchyModel2D(v=8, s=2, L=4, m=2, seed=0)
        grid1, _ = model.sample(4, seed=1)
        grid2, _ = model.sample(4, seed=2)
        assert not np.array_equal(grid1, grid2)

    def test_return_hierarchy(self):
        model = RandomHierarchyModel2D(v=4, s=2, L=3, m=2, seed=0)
        grid, root, levels = model.sample(3, seed=1, return_hierarchy=True)
        assert len(levels) == model.L + 1
        for l, level in enumerate(levels):
            assert level.shape == (3, 2 ** l, 2 ** l)
        np.testing.assert_array_equal(levels[0][:, 0, 0], root)
        np.testing.assert_array_equal(levels[-1], grid)

    def test_functional_wrapper(self):
        grid, root = sample_random_hierarchy_model_2d(4, v=4, s=2, L=2, m=2, seed=0)
        assert grid.shape == (4, 4, 4)
        assert root.shape == (4,)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": 1},
            {"s": 1},
            {"L": 0},
            {"m": 0},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        base = dict(v=4, s=2, L=2, m=2, seed=0)
        base.update(kwargs)
        with pytest.raises(ValueError):
            RandomHierarchyModel2D(**base)


class TestRandomHierarchy2DDataset:

    def test_token_mode(self):
        ds = RandomHierarchy2DDataset(n_samples=6, v=4, s=2, L=3, m=2, seed=0, onehot=False)
        assert len(ds) == 6
        x, label = ds[0]
        assert x.dtype == torch.long
        assert x.shape == (8, 8)
        assert label.dtype == torch.long
        assert label.shape == ()
        assert int(x.min()) >= 0 and int(x.max()) < ds.v

    def test_onehot_mode(self):
        ds = RandomHierarchy2DDataset(n_samples=3, v=5, s=2, L=2, m=2, seed=0, onehot=True)
        x, label = ds[0]
        assert x.dtype == torch.float32
        assert x.shape == (5, 4, 4)
        # exactly one-hot along the channel dimension at every site
        torch.testing.assert_close(x.sum(dim=0), torch.ones(4, 4))

    def test_precomputed_grids(self):
        model = RandomHierarchyModel2D(v=4, s=2, L=2, m=2, seed=0)
        grids, labels = model.sample(5, seed=1)
        ds = RandomHierarchy2DDataset(model=model, grids=grids, labels=labels)
        assert len(ds) == 5
        x, label = ds[2]
        np.testing.assert_array_equal(x.numpy(), grids[2])
        assert int(label) == int(labels[2])

    def test_requires_n_samples_or_grids(self):
        with pytest.raises(ValueError):
            RandomHierarchy2DDataset(v=4, s=2, L=2, m=2, seed=0)


def _brute_force_bp_L1(model, xt, mask_token):
    """Exact marginals for an L=1 model via full enumeration over (root
    symbol, word choice) -- independent of RandomHierarchyModel2D.belief_propagation."""
    v, s, m = model.v, model.s, model.m
    rule = model.rules[0]  # (v, m, s, s)
    n = xt.shape[0]
    marg_root = np.zeros((n, 1, 1, v))
    marg_leaf = np.zeros((n, s, s, v))
    for idx in range(n):
        obs = xt[idx]
        joint = np.zeros((v, m))
        for a in range(v):
            for w in range(m):
                pattern = rule[a, w]
                if np.all((obs == mask_token) | (obs == pattern)):
                    joint[a, w] = 1.0 / (v * m)
        joint /= joint.sum()
        marg_root[idx, 0, 0] = joint.sum(axis=1)
        for di in range(s):
            for dj in range(s):
                for a in range(v):
                    for w in range(m):
                        marg_leaf[idx, di, dj, rule[a, w, di, dj]] += joint[a, w]
    return [marg_root, marg_leaf]


def _brute_force_bp_L2(model, xt, mask_token):
    """Exact marginals for an L=2 model via full enumeration over every
    node's (symbol, word) choice in the tree -- independent reference for
    belief_propagation's recursive up/down passes."""
    import itertools

    v, s, m = model.v, model.s, model.m
    rule0, rule1 = model.rules[0], model.rules[1]
    n = xt.shape[0]
    grid_size = s * s
    marg = [np.zeros((n, 1, 1, v)), np.zeros((n, s, s, v)), np.zeros((n, grid_size, grid_size, v))]

    for idx in range(n):
        obs = xt[idx]
        total = 0.0
        entries = []  # (prob, a0, level1, leaf)
        for a0 in range(v):
            for w0 in range(m):
                level1 = rule0[a0, w0]  # (s, s)
                for combo in itertools.product(range(m), repeat=s * s):
                    leaf = np.zeros((grid_size, grid_size), dtype=np.int64)
                    idxc = 0
                    for bi in range(s):
                        for bj in range(s):
                            sym1 = level1[bi, bj]
                            w1 = combo[idxc]
                            idxc += 1
                            leaf[bi * s : (bi + 1) * s, bj * s : (bj + 1) * s] = rule1[sym1, w1]
                    if np.all((obs == mask_token) | (obs == leaf)):
                        prob = (1.0 / (v * m)) * (1.0 / m) ** (s * s)
                        entries.append((prob, a0, level1, leaf))
                        total += prob
        for prob, a0, level1, leaf in entries:
            p_norm = prob / total
            marg[0][idx, 0, 0, a0] += p_norm
            for bi in range(s):
                for bj in range(s):
                    marg[1][idx, bi, bj, level1[bi, bj]] += p_norm
            for i in range(grid_size):
                for j in range(grid_size):
                    marg[2][idx, i, j, leaf[i, j]] += p_norm
    return marg


class TestBeliefPropagation:

    def test_matches_brute_force_L1(self):
        model = RandomHierarchyModel2D(v=3, s=2, L=1, m=2, seed=0)
        grid, root = model.sample(6, seed=1)
        mask_token = model.v
        rng = np.random.default_rng(2)
        xt = grid.copy()
        mask = rng.random(xt.shape) < 0.5
        xt[mask] = mask_token

        got = model.belief_propagation(xt, mask_token=mask_token)
        expected = _brute_force_bp_L1(model, xt, mask_token)
        for l in range(2):
            np.testing.assert_allclose(got[l], expected[l], atol=1e-8)

    def test_matches_brute_force_L2(self):
        model = RandomHierarchyModel2D(v=3, s=2, L=2, m=2, seed=3)
        grid, root = model.sample(3, seed=4)
        mask_token = model.v
        rng = np.random.default_rng(5)
        xt = grid.copy()
        mask = rng.random(xt.shape) < 0.4
        xt[mask] = mask_token

        got = model.belief_propagation(xt, mask_token=mask_token)
        expected = _brute_force_bp_L2(model, xt, mask_token)
        for l in range(3):
            np.testing.assert_allclose(got[l], expected[l], atol=1e-8)

    def test_fully_observed_gives_certainty(self):
        model = RandomHierarchyModel2D(v=6, s=2, L=3, m=2, seed=10)
        grid, root, levels = model.sample(5, seed=11, return_hierarchy=True)
        marginals = model.belief_propagation(grid, mask_token=model.v)
        for l in range(model.L + 1):
            expected_onehot = np.eye(model.v)[levels[l]]
            np.testing.assert_allclose(marginals[l], expected_onehot, atol=1e-6)

    def test_fully_masked_root_is_exactly_uniform(self):
        # The root symbol is drawn uniformly by construction, and with zero
        # observations there is *no* other source of information about it,
        # so its marginal must be exactly uniform regardless of the (fixed,
        # random) rule table realization.
        model = RandomHierarchyModel2D(v=5, s=2, L=2, m=3, seed=12)
        xt = np.full((4, model.grid_size, model.grid_size), model.v, dtype=np.int64)
        marginals = model.belief_propagation(xt, mask_token=model.v)
        np.testing.assert_allclose(marginals[0], np.full_like(marginals[0], 1.0 / model.v), atol=1e-8)

    def test_fully_masked_deeper_levels_match_empirical_prior(self):
        # Unlike the root, a level l > 0 marginal need *not* be exactly
        # uniform under total masking: it's built from summing over only
        # v*m fixed, randomly-drawn rule-table entries feeding into that
        # position, not a symmetric ensemble average, so finite-size
        # imbalances in the realized rules are expected. What BP's
        # zero-information marginal *must* equal is the model's true
        # marginal prior at that level, i.e. the population frequency you'd
        # get by sampling many instances -- cross-check via Monte Carlo.
        model = RandomHierarchyModel2D(v=5, s=2, L=2, m=3, seed=12)
        xt = np.full((1, model.grid_size, model.grid_size), model.v, dtype=np.int64)
        marginals = model.belief_propagation(xt, mask_token=model.v)

        _, _, levels = model.sample(50_000, seed=99, return_hierarchy=True)
        for l in range(1, model.L + 1):
            empirical = np.bincount(levels[l][:, 0, 0], minlength=model.v) / levels[l].shape[0]
            np.testing.assert_allclose(marginals[l][0, 0, 0], empirical, atol=0.01)

    def test_expected_correct_probability_bounds_and_perfect_at_zero_corruption(self):
        model = RandomHierarchyModel2D(v=5, s=2, L=3, m=2, seed=13)
        grid, root, levels = model.sample(6, seed=14, return_hierarchy=True)
        mask_token = model.v

        p_correct_clean = model.bp_expected_correct_probability(grid, levels, mask_token=mask_token)
        for p in p_correct_clean:
            assert p == pytest.approx(1.0, abs=1e-6)

        rng = np.random.default_rng(15)
        xt = grid.copy()
        xt[rng.random(xt.shape) < 0.5] = mask_token
        p_correct_noisy = model.bp_expected_correct_probability(xt, levels, mask_token=mask_token)
        for p in p_correct_noisy:
            assert 0.0 <= p <= 1.0 + 1e-9
