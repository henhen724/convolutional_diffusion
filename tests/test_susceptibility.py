"""Tests for src.utils.susceptibility (dynamical correlation/susceptibility).

chi = sum_ij C_ij / sum_i C_ii is bounded between 1 (every site changes
independently -- each change is its own isolated "cluster") and N (every
site changes together as one perfectly correlated block), so both
extremes give sharp, exactly-known targets to test against.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.susceptibility import dynamical_susceptibility, find_peak, pool_sigma_field


class TestDynamicalSusceptibility:

    def test_independent_sites_give_chi_near_one(self):
        rng = np.random.default_rng(0)
        n_x0, r, n = 5, 20_000, 10
        sigma = rng.integers(0, 2, size=(n_x0, r, n)).astype(np.float64)
        chi, c_mean = dynamical_susceptibility(sigma)
        assert chi == pytest.approx(1.0, abs=0.05)
        assert c_mean.shape == (n, n)

    def test_perfectly_correlated_sites_give_chi_near_n(self):
        rng = np.random.default_rng(1)
        n_x0, r, n = 5, 20_000, 12
        shared = rng.integers(0, 2, size=(n_x0, r, 1)).astype(np.float64)
        sigma = np.broadcast_to(shared, (n_x0, r, n)).copy()
        chi, _ = dynamical_susceptibility(sigma)
        assert chi == pytest.approx(n, rel=0.02)

    def test_two_independent_clusters_give_chi_near_half_n(self):
        # N sites split into two independently-fluctuating blocks of N/2:
        # each block is a single "cluster", so chi should be ~ N/2.
        rng = np.random.default_rng(2)
        n_x0, r, half = 5, 20_000, 6
        cluster_a = rng.integers(0, 2, size=(n_x0, r, 1)).astype(np.float64)
        cluster_b = rng.integers(0, 2, size=(n_x0, r, 1)).astype(np.float64)
        sigma = np.concatenate(
            [np.broadcast_to(cluster_a, (n_x0, r, half)), np.broadcast_to(cluster_b, (n_x0, r, half))],
            axis=2,
        ).copy()
        chi, _ = dynamical_susceptibility(sigma)
        assert chi == pytest.approx(half, rel=0.05)

    def test_averages_covariance_across_x0_not_across_trajectories(self):
        # Two starting data: one with independent sites (chi=1), one with
        # fully correlated sites (chi=N). Averaging the *covariance
        # matrices* (not the raw sigma) means the result is NOT just the
        # mean of the two chi values -- construct it by hand and check.
        rng = np.random.default_rng(3)
        r, n = 20_000, 8
        indep = rng.integers(0, 2, size=(r, n)).astype(np.float64)
        shared = rng.integers(0, 2, size=(r, 1)).astype(np.float64)
        correlated = np.broadcast_to(shared, (r, n)).copy()
        sigma = np.stack([indep, correlated], axis=0)  # (2, r, n)

        chi, c_mean = dynamical_susceptibility(sigma)
        c_indep = np.cov(indep, rowvar=False)
        c_corr = np.cov(correlated, rowvar=False)
        expected_c_mean = (c_indep + c_corr) / 2
        np.testing.assert_allclose(c_mean, expected_c_mean, atol=1e-6)
        expected_chi = expected_c_mean.sum() / np.trace(expected_c_mean)
        assert chi == pytest.approx(expected_chi, rel=1e-6)

    def test_rejects_too_few_trajectories(self):
        sigma = np.zeros((3, 1, 5))
        with pytest.raises(ValueError):
            dynamical_susceptibility(sigma)

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError):
            dynamical_susceptibility(np.zeros((3, 5)))


class TestPoolSigmaField:

    def test_pooling_shape_and_values(self):
        field = torch.zeros(2, 4, 4)
        field[:, :2, :2] = 1.0
        field[:, :2, 2:] = 3.0
        field[:, 2:, :2] = 5.0
        field[:, 2:, 2:] = 7.0
        pooled = pool_sigma_field(field, pool_size=2)
        assert pooled.shape == (2, 2, 2)
        expected = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        torch.testing.assert_close(pooled[0], expected)
        torch.testing.assert_close(pooled[1], expected)


class TestFindPeak:

    def test_find_peak_basic(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        chi = np.array([1.0, 2.0, 9.0, 3.0, 1.0])
        t_star, chi_star, idx = find_peak(t, chi)
        assert t_star == 0.5
        assert chi_star == 9.0
        assert idx == 2
