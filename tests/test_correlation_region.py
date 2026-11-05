"""Tests for src.utils.correlation_region."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.correlation_region import (
    connected_region_containing_target,
    correlation_region,
    fit_radial_power_law,
    nested_correlation_curve,
    pixel_correlation_map,
    radial_correlation_profile,
    region_mask,
    region_shape_stats,
    sample_x0_xt,
)


class _TinyImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0


class TestPixelCorrelationMap:

    def test_matches_np_corrcoef(self):
        rng = np.random.default_rng(0)
        n, h, w = 500, 4, 5
        x0 = rng.normal(size=(n, h, w))
        xt = rng.normal(size=(n, h, w)) + 0.5 * x0  # correlated with x0 pointwise
        target = (1, 2)
        corr_map = pixel_correlation_map(x0, xt, target)
        for i in range(h):
            for j in range(w):
                expected = np.corrcoef(x0[:, target[0], target[1]], xt[:, i, j])[0, 1]
                assert corr_map[i, j] == pytest.approx(expected, abs=1e-8)

    def test_perfect_correlation_when_xt_equals_x0(self):
        rng = np.random.default_rng(1)
        n, h, w = 200, 3, 3
        x0 = rng.normal(size=(n, h, w))
        xt = x0.copy()  # noiseless copy: alpha_t=1, beta_t=0
        corr_map = pixel_correlation_map(x0, xt, (1, 1))
        # every pixel that's a copy of x0 at the SAME location as target is corr=1;
        # off-target pixels are only correlated insofar as x0 itself correlates
        # across locations, which for i.i.d. noise is ~0.
        assert corr_map[1, 1] == pytest.approx(1.0, abs=1e-8)

    def test_near_zero_when_xt_independent_of_x0(self):
        rng = np.random.default_rng(2)
        n, h, w = 5000, 3, 3
        x0 = rng.normal(size=(n, h, w))
        xt = rng.normal(size=(n, h, w))  # independent of x0 entirely
        corr_map = pixel_correlation_map(x0, xt, (1, 1))
        assert np.all(np.abs(corr_map) < 0.05)

    def test_shape_and_constant_column_handled(self):
        rng = np.random.default_rng(3)
        n, h, w = 100, 2, 2
        x0 = rng.normal(size=(n, h, w))
        xt = rng.normal(size=(n, h, w))
        xt[:, 0, 0] = 1.0  # zero variance -> correlation undefined, should be 0 not NaN
        corr_map = pixel_correlation_map(x0, xt, (0, 1))
        assert corr_map.shape == (h, w)
        assert not np.isnan(corr_map).any()
        assert corr_map[0, 0] == 0.0


class TestRegionMask:

    def test_threshold_matches_bonferroni_corrected_formula(self):
        from scipy.stats import norm

        corr_map = np.array([[0.5, 0.05], [-0.5, -0.05]])
        n_samples = 100
        fwer_alpha = 0.05
        n_pixels = corr_map.size
        z = norm.ppf(1.0 - fwer_alpha / (2.0 * n_pixels))
        threshold = z / np.sqrt(n_samples)

        mask = region_mask(corr_map, n_samples, fwer_alpha=fwer_alpha)
        np.testing.assert_array_equal(mask, np.abs(corr_map) > threshold)
        # sanity check: the strong entries clear the corrected threshold,
        # the weak ones don't (same qualitative result as the naive
        # threshold here, since 0.5 and 0.05 aren't close to the boundary)
        np.testing.assert_array_equal(mask, np.array([[True, False], [True, False]]))
        # the correction should be strictly more conservative than a bare
        # single-comparison z=1 threshold
        assert threshold > 1.0 / np.sqrt(n_samples)

    def test_null_false_positive_rate_stays_below_percolation_threshold(self):
        # Under the null of zero true correlation, a Pearson correlation
        # estimated from n_samples i.i.d. draws is approximately
        # N(0, 1/n_samples). A bare z=1 threshold would flag ~32% of pixels
        # by chance alone -- comfortably above the ~0.59 4-connectivity
        # site-percolation threshold once combined with any real weak
        # correlation. The Bonferroni-corrected threshold should keep the
        # null false-positive rate near fwer_alpha, far below percolation.
        rng = np.random.default_rng(42)
        n_samples = 5000
        h, w = 32, 32
        corr_map = rng.normal(scale=1.0 / np.sqrt(n_samples), size=(h, w))
        mask = region_mask(corr_map, n_samples)
        assert mask.mean() < 0.1

    def test_more_pixels_tested_raises_the_threshold(self):
        rng = np.random.default_rng(1)
        corr_map_small = rng.normal(scale=0.1, size=(4, 4))
        corr_map_big = rng.normal(scale=0.1, size=(64, 64))
        n_samples = 100
        # same per-pixel correlation values pattern, but Bonferroni
        # correction over more simultaneous tests must be stricter
        from scipy.stats import norm

        z_small = norm.ppf(1.0 - 0.05 / (2 * corr_map_small.size))
        z_big = norm.ppf(1.0 - 0.05 / (2 * corr_map_big.size))
        assert z_big > z_small


class TestConnectedRegionContainingTarget:

    def test_isolates_blob_touching_target_from_disconnected_speckle(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:13, 8:13] = True  # a 5x5 blob touching the target
        target = (10, 10)
        assert mask[target]
        # scattered, disconnected "look-elsewhere" false positives elsewhere
        mask[0, 0] = True
        mask[19, 19] = True
        mask[0, 19] = True
        mask[3, 15] = True

        connected = connected_region_containing_target(mask, target)
        expected = np.zeros((20, 20), dtype=bool)
        expected[8:13, 8:13] = True
        np.testing.assert_array_equal(connected, expected)
        assert connected.sum() == 25
        assert connected[8:13, 8:13].all()
        assert not connected[0, 0]
        assert not connected[19, 19]
        assert not connected[0, 19]
        assert not connected[3, 15]

    def test_returns_all_false_when_target_not_in_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True  # speckle far from target, target itself not set
        connected = connected_region_containing_target(mask, (5, 5))
        assert not connected.any()

    def test_diagonal_neighbors_not_connected(self):
        # 4-connectivity: a pixel touching target only diagonally is a
        # separate component, not part of the region.
        mask = np.zeros((5, 5), dtype=bool)
        target = (2, 2)
        mask[target] = True
        mask[1, 1] = True  # diagonal neighbor only
        connected = connected_region_containing_target(mask, target)
        assert connected.sum() == 1
        assert connected[target]
        assert not connected[1, 1]


class TestRegionShapeStats:

    def test_empty_mask(self):
        stats = region_shape_stats(np.zeros((5, 5), dtype=bool))
        assert stats["area"] == 0
        assert stats["r_eff"] == 0.0
        assert np.isnan(stats["aspect_ratio"])

    def test_circular_mask_is_isotropic(self):
        n = 41
        yy, xx = np.mgrid[0:n, 0:n]
        cy = cx = n // 2
        radius = 10
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        stats = region_shape_stats(mask)
        assert stats["area"] == pytest.approx(np.pi * radius**2, rel=0.05)
        assert stats["aspect_ratio"] == pytest.approx(1.0, abs=0.05)
        assert stats["eccentricity"] == pytest.approx(0.0, abs=0.1)
        assert stats["centroid"][0] == pytest.approx(cy, abs=0.5)
        assert stats["centroid"][1] == pytest.approx(cx, abs=0.5)

    def test_elongated_mask_has_high_eccentricity(self):
        mask = np.zeros((41, 41), dtype=bool)
        mask[20, 5:36] = True  # a thin horizontal line: very elongated
        stats = region_shape_stats(mask)
        assert stats["aspect_ratio"] > 5.0
        assert stats["eccentricity"] > 0.95

    def test_square_mask_aspect_ratio_near_one(self):
        mask = np.zeros((41, 41), dtype=bool)
        mask[15:26, 15:26] = True  # 11x11 square
        stats = region_shape_stats(mask)
        assert stats["aspect_ratio"] == pytest.approx(1.0, abs=0.05)


class TestRadialCorrelationProfile:

    def test_constant_map_gives_constant_profile(self):
        corr_map = np.full((21, 21), 0.7)
        radii, values = radial_correlation_profile(corr_map, target=(10, 10))
        assert radii[0] == 1
        np.testing.assert_allclose(values, 0.7, atol=1e-10)

    def test_r_max_bounded_by_distance_to_nearest_edge(self):
        corr_map = np.zeros((21, 21))
        radii, _ = radial_correlation_profile(corr_map, target=(2, 10))
        # target is 2 rows from the top edge -> the largest radius whose
        # ring still fully fits inside the image is r_max=2 (touching row 0)
        assert radii.max() == 2

    def test_recovers_known_power_law_exponent(self):
        h, w = 41, 41
        target = (20, 20)
        true_epsilon = 1.2
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - target[0]) ** 2 + (xx - target[1]) ** 2)
        r_safe = np.where(r == 0, 1.0, r)  # avoid 0**(-eps) at the target itself
        corr_map = r_safe ** (-true_epsilon)

        radii, values = radial_correlation_profile(corr_map, target)
        epsilon, r2 = fit_radial_power_law(radii, values, r_min=2)
        assert epsilon == pytest.approx(true_epsilon, rel=0.1)
        assert r2 > 0.99


class TestFitRadialPowerLaw:

    def test_insufficient_points_returns_nan(self):
        radii = np.array([1, 2])
        values = np.array([0.5, 0.3])
        epsilon, r2 = fit_radial_power_law(radii, values, r_min=1)
        assert np.isnan(epsilon)
        assert np.isnan(r2)

    def test_zero_values_excluded_not_errored(self):
        radii = np.array([1, 2, 3, 4, 5])
        values = np.array([0.5, 0.0, 0.2, 0.1, 0.05])
        epsilon, r2 = fit_radial_power_law(radii, values, r_min=1)
        assert np.isfinite(epsilon)
        assert np.isfinite(r2)


class TestSampleX0Xt:

    def test_shapes_and_no_replacement(self):
        rng = np.random.default_rng(4)
        images = [torch.from_numpy(rng.normal(size=(3, 8, 8)).astype(np.float32)) for _ in range(20)]
        ds = _TinyImageDataset(images)
        from src.utils.noise_schedules import cosine_noise_schedule

        x0, xt = sample_x0_xt(ds, n_samples=10, t=0.5, schedule=cosine_noise_schedule, seed=0)
        assert x0.shape == (10, 8, 8)
        assert xt.shape == (10, 8, 8)

    def test_t_zero_gives_xt_equal_x0(self):
        rng = np.random.default_rng(5)
        images = [torch.from_numpy(rng.normal(size=(1, 6, 6)).astype(np.float32)) for _ in range(10)]
        ds = _TinyImageDataset(images)
        from src.utils.noise_schedules import cosine_noise_schedule

        x0, xt = sample_x0_xt(ds, n_samples=10, t=0.0, schedule=cosine_noise_schedule, seed=0)
        np.testing.assert_allclose(x0, xt, atol=1e-5)

    def test_rejects_n_samples_too_large(self):
        images = [torch.zeros(1, 4, 4) for _ in range(5)]
        ds = _TinyImageDataset(images)
        from src.utils.noise_schedules import cosine_noise_schedule

        with pytest.raises(ValueError):
            sample_x0_xt(ds, n_samples=10, t=0.5, schedule=cosine_noise_schedule)


class TestCorrelationRegionEndToEnd:

    def test_runs_and_is_internally_consistent(self):
        rng = np.random.default_rng(6)
        # a dataset with real spatial structure: smoothly-varying random fields,
        # so nearby pixels are genuinely correlated and distant ones are not.
        n_img, h, w = 300, 12, 12
        base = rng.normal(size=(n_img, 1, 1))
        field = 0.7 * base + 0.3 * rng.normal(size=(n_img, h, w))
        images = [torch.from_numpy(field[i][None].astype(np.float32)) for i in range(n_img)]
        ds = _TinyImageDataset(images)
        from src.utils.noise_schedules import cosine_noise_schedule

        out = correlation_region(ds, n_samples=300, t=0.3, schedule=cosine_noise_schedule, target=(6, 6), seed=7)
        assert out["mask"][6, 6]  # target pixel always correlates with itself
        assert out["area"] >= 1
        assert 0.0 <= out["corr_map"].max() <= 1.0
        # connected mask is a (subset-or-equal) of the raw thresholded mask
        assert out["raw_mask"][6, 6]
        assert np.all(out["mask"] <= out["raw_mask"])


class TestNestedCorrelationCurve:

    def test_prefix_property_uses_slice_of_the_largest_draw(self):
        # curve[50] must be computed from exactly the first 50 rows of the
        # SAME draw used for n_max=100 -- not a fresh, independently-seeded
        # 50-sample draw (numpy's choice(..., replace=False) is not
        # prefix-consistent across different sample sizes even with the same
        # seed, so this has to be checked against a manual slice, not
        # against an independent sample_x0_xt(..., n_samples=50, ...) call).
        rng = np.random.default_rng(8)
        n_img, h, w = 200, 8, 8
        base = rng.normal(size=(n_img, 1, 1))
        field = 0.6 * base + 0.4 * rng.normal(size=(n_img, h, w))
        images = [torch.from_numpy(field[i][None].astype(np.float32)) for i in range(n_img)]
        ds = _TinyImageDataset(images)
        from src.utils.noise_schedules import cosine_noise_schedule

        curve = nested_correlation_curve(
            ds, n_values=[20, 50, 100], t=0.4, schedule=cosine_noise_schedule, target=(4, 4), seed=9
        )
        x0_full, xt_full = sample_x0_xt(ds, 100, t=0.4, schedule=cosine_noise_schedule, seed=9)
        expected_50 = pixel_correlation_map(x0_full[:50], xt_full[:50], (4, 4))
        np.testing.assert_allclose(curve[50]["corr_map"], expected_50, atol=1e-10)

    def test_returns_all_requested_n(self):
        rng = np.random.default_rng(10)
        images = [torch.from_numpy(rng.normal(size=(1, 6, 6)).astype(np.float32)) for _ in range(60)]
        ds = _TinyImageDataset(images)
        from src.utils.noise_schedules import cosine_noise_schedule

        curve = nested_correlation_curve(
            ds, n_values=[10, 30, 60], t=0.5, schedule=cosine_noise_schedule, target=(3, 3), seed=11
        )
        assert set(curve.keys()) == {10, 30, 60}
        for n, stats in curve.items():
            assert stats["n_samples"] == n
