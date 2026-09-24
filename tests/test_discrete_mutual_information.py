"""Tests for src.utils.discrete_mutual_information.

The key correctness property being tested is the exact identity used to
avoid estimating the (analytically known) conditional entropy term:

    I(X0; Xt) = H(Xt) - n_sites * H_b(p)

which is checked against hand-derived closed forms for two toy cases where
the true mutual information is known exactly:
1. A single i.i.d. uniform site over v symbols: I = (1-p) * log2(v).
2. Two perfectly correlated binary sites (b = a): I = 1 - p**2 bits.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.discrete_mutual_information import (
    binary_entropy_bits,
    empirical_entropy_bits,
    extract_all_patches,
    extract_block_contained_patches,
    label_patch_mutual_information,
    patch_erasure_mutual_information,
    patch_size_mutual_information_curves,
    resolvable_latent_height_grid,
)
from src.utils.random_hierarchy import RandomHierarchyModel2D


class TestBasics:

    def test_binary_entropy_endpoints_and_peak(self):
        vals = binary_entropy_bits(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(vals, [0.0, 1.0, 0.0], atol=1e-10)

    def test_empirical_entropy_uniform(self):
        # 4 equally frequent symbols -> exactly 2 bits, no correction needed.
        values = [0, 1, 2, 3] * 10000
        h, n, n_distinct = empirical_entropy_bits(values, correction=False)
        assert n == 40000
        assert n_distinct == 4
        assert h == pytest.approx(2.0, abs=1e-9)

    def test_extract_all_patches_values_and_shape(self):
        grid = torch.arange(9).reshape(1, 3, 3)
        patches = extract_all_patches(grid, k=2)
        assert patches.shape == (4, 2, 2)
        expected = [
            [[0, 1], [3, 4]],
            [[1, 2], [4, 5]],
            [[3, 4], [6, 7]],
            [[4, 5], [7, 8]],
        ]
        np.testing.assert_array_equal(patches.numpy(), np.array(expected))


class TestExactErasureMutualInformation:

    def test_single_iid_site_matches_closed_form(self):
        rng = np.random.default_rng(0)
        v = 4
        n = 300_000
        flat = rng.integers(0, v, size=(n, 1))
        ps = [0.0, 0.25, 0.5, 0.75, 0.9]
        results = patch_erasure_mutual_information(flat, n_sites=1, p_values=ps, rng=rng)
        for p in ps:
            expected = (1 - p) * np.log2(v)
            assert results[p]["mi_bits"] == pytest.approx(expected, abs=0.03)

    def test_two_perfectly_correlated_sites_matches_closed_form(self):
        rng = np.random.default_rng(1)
        n = 300_000
        a = rng.integers(0, 2, size=(n,))
        b = a.copy()  # perfectly redundant second site
        flat = np.stack([a, b], axis=1)
        ps = [0.0, 0.2, 0.5, 0.8, 1.0]
        results = patch_erasure_mutual_information(flat, n_sites=2, p_values=ps, rng=rng)
        for p in ps:
            expected = 1 - p**2  # derived by hand, see module docstring
            assert results[p]["mi_bits"] == pytest.approx(expected, abs=0.03)

    def test_fully_masked_gives_zero_mutual_information(self):
        rng = np.random.default_rng(2)
        flat = rng.integers(0, 5, size=(50_000, 3))
        results = patch_erasure_mutual_information(flat, n_sites=3, p_values=[1.0], rng=rng)
        assert results[1.0]["mi_bits"] == pytest.approx(0.0, abs=1e-6)

    def test_no_masking_recovers_full_patch_entropy(self):
        rng = np.random.default_rng(3)
        v = 3
        flat = rng.integers(0, v, size=(300_000, 2))  # two independent sites
        results = patch_erasure_mutual_information(flat, n_sites=2, p_values=[0.0], rng=rng, correction=False)
        expected = 2 * np.log2(v)  # I(X0;X0) = H(X0) for two independent uniform sites
        assert results[0.0]["mi_bits"] == pytest.approx(expected, abs=0.02)


class TestPatchSizeCurvesOnRHM:

    def test_curves_shape_and_basic_sanity(self):
        rhm = RandomHierarchyModel2D(v=4, s=2, L=3, m=2, seed=0)
        grid, _ = rhm.sample(1500, seed=1)
        grids = torch.from_numpy(grid).long()

        k_values = [1, 2, 3]
        p_values = [0.2, 0.5, 0.8]
        curves = patch_size_mutual_information_curves(grids, k_values, p_values, seed=5)

        assert set(curves.keys()) == set(k_values)
        for k in k_values:
            assert set(curves[k].keys()) == set(p_values)
            for p in p_values:
                mi = curves[k][p]["mi_bits"]
                assert mi >= -1e-9
                # sanity bound (with slack for finite-sample estimator noise):
                # can't greatly exceed the max possible entropy of k*k v=4 sites
                assert mi <= k * k * np.log2(4) + 0.5

        # more corruption should not produce (meaningfully) more information,
        # for a fixed k; allow slack for finite-sample entropy-estimator noise
        for k in k_values:
            mi_low_p = curves[k][0.2]["mi_bits"]
            mi_high_p = curves[k][0.8]["mi_bits"]
            assert mi_low_p >= mi_high_p - 0.2


class TestBlockContainedPatchesAndLabelMI:

    def test_extract_block_contained_patches_values(self):
        leaf = np.arange(16).reshape(1, 4, 4)
        block_values = np.array([[[100, 101], [102, 103]]])  # (1, 2, 2)
        flat_patches, labels = extract_block_contained_patches(leaf, block_values, block_size=2, k=1)
        pairs = set(zip(flat_patches[:, 0].tolist(), labels.tolist()))
        expected = {
            (0, 100), (1, 100), (4, 100), (5, 100),
            (2, 101), (3, 101), (6, 101), (7, 101),
            (8, 102), (9, 102), (12, 102), (13, 102),
            (10, 103), (11, 103), (14, 103), (15, 103),
        }
        assert pairs == expected

    def test_extract_block_contained_patches_rejects_oversized_k(self):
        leaf = np.zeros((1, 4, 4), dtype=np.int64)
        block_values = np.zeros((1, 2, 2), dtype=np.int64)
        with pytest.raises(ValueError):
            extract_block_contained_patches(leaf, block_values, block_size=2, k=3)

    def test_label_patch_mi_identity_matches_closed_form(self):
        # the "patch" literally *is* the label -> I = (1-p) * H(Z) = (1-p) bits
        rng = np.random.default_rng(10)
        n = 200_000
        z = rng.integers(0, 2, size=n)
        flat_patches = z.reshape(n, 1)
        results = label_patch_mutual_information(flat_patches, z, p_values=[0.0, 0.3, 0.7], rng=rng)
        for p in [0.0, 0.3, 0.7]:
            expected = (1 - p) * 1.0
            assert results[p]["mi_bits"] == pytest.approx(expected, abs=0.03)

    def test_label_patch_mi_independent_gives_zero(self):
        rng = np.random.default_rng(11)
        n = 200_000
        z = rng.integers(0, 3, size=n)
        flat_patches = rng.integers(0, 3, size=(n, 2))  # unrelated to z
        results = label_patch_mutual_information(flat_patches, z, p_values=[0.2, 0.6], rng=rng)
        for p in [0.2, 0.6]:
            assert results[p]["mi_bits"] == pytest.approx(0.0, abs=0.03)


class TestResolvableLatentHeightGrid:

    def test_perfectly_redundant_leaf_resolves_root_too(self):
        # leaf == root everywhere -> a single revealed leaf pixel exactly
        # reveals the root symbol too, so both levels share one resolvability
        # threshold: p=0.3 (clearly below it) resolves the root; p=0.8
        # (clearly above it) fails even the leaf.
        rng = np.random.default_rng(20)
        n = 20_000
        root = rng.integers(0, 2, size=(n, 1, 1))
        leaf = np.broadcast_to(root, (n, 2, 2)).copy()
        levels = [root, leaf]  # L = 1, s = 2

        out = resolvable_latent_height_grid(
            levels, s=2, k_values=[1], p_values=[0.3, 0.8], threshold_frac=0.5, seed=21
        )
        grid = out["height_grid"]
        assert grid[0, 0] == 1  # p=0.3: root (height 1) resolvable
        assert grid[1, 0] == -1  # p=0.8: even the leaf fails

    def test_independent_root_caps_height_at_leaf(self):
        # leaf is unrelated to root -> the root can never be resolved from a
        # leaf patch, so height should cap at 0 (leaf only) instead of
        # incorrectly crediting the root.
        rng = np.random.default_rng(22)
        n = 20_000
        root = rng.integers(0, 2, size=(n, 1, 1))
        leaf = rng.integers(0, 2, size=(n, 2, 2))
        levels = [root, leaf]

        out = resolvable_latent_height_grid(
            levels, s=2, k_values=[1], p_values=[0.3, 0.8], threshold_frac=0.5, seed=23
        )
        grid = out["height_grid"]
        assert grid[0, 0] == 0  # p=0.3: leaf resolvable, root is not
        assert grid[1, 0] == -1  # p=0.8: leaf also fails

    def test_rejects_unsorted_p_values(self):
        levels = [np.zeros((10, 1, 1), dtype=np.int64), np.zeros((10, 2, 2), dtype=np.int64)]
        with pytest.raises(ValueError):
            resolvable_latent_height_grid(levels, s=2, k_values=[1], p_values=[0.5, 0.1])

    def test_height_is_monotone_non_increasing_in_p(self):
        # Data-processing inequality: more corruption can only make a level
        # harder (never easier) to resolve, for a fixed k. The function
        # enforces this explicitly to guard against finite-sample noise, so
        # this should hold exactly regardless of sample size.
        rhm = RandomHierarchyModel2D(v=4, s=2, L=3, m=2, seed=30)
        _, _, levels = rhm.sample(3000, seed=31, return_hierarchy=True)
        p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        out = resolvable_latent_height_grid(levels, s=2, k_values=[1, 2, 3], p_values=p_values, seed=32)
        height_grid = out["height_grid"]
        assert np.all(np.diff(height_grid, axis=0) <= 0)
