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
