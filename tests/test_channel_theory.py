"""Tests for channel theory scale calibration (pixel-space ELS/LS)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.channel_theory import (
    compute_posterior_stats_at_t,
    infer_beta_at_k_star,
    load_scale_calibration,
    collect_stats_vs_k,
    infer_betas_for_rate_distortion,
    cost_at_beta,
    t_to_snr,
)
from src.utils.data import get_dataset
from src.utils.idealscore import LocalEquivScoreModule, LocalScoreModule
from src.utils.noise_schedules import cosine_noise_schedule


class TestChannelTheoryUtils:
    """Test src.utils.channel_theory functions."""

    def test_t_to_snr(self):
        """SNR = α_t/β_t; scalar and array; decreases as t increases (β_t grows)."""
        snr_early = float(t_to_snr(0.05, cosine_noise_schedule))
        snr_mid = float(t_to_snr(0.5, cosine_noise_schedule))
        snr_late = float(t_to_snr(0.95, cosine_noise_schedule))
        assert snr_early >= 0 and snr_mid >= 0 and snr_late >= 0
        assert snr_early > snr_mid > snr_late
        arr = t_to_snr(np.array([0.1, 0.5, 0.9]), cosine_noise_schedule)
        assert arr.shape == (3,)
        assert np.all(arr[1:] < arr[:-1])

    def test_infer_beta_at_k_star_interior(self):
        k_vals = [3, 5, 7, 9, 11]
        # Cost = β×reg − S; minimizer gives β = d(S)/dk / d(reg)/dk.
        # At k=7 (idx 2): d_entropy = (12-7)/4 = 1.25, d_reg = (2.5-1.5)/4 = 0.25 => beta = 5.
        entropy = np.array([5.0, 7.0, 10.0, 12.0, 13.0])
        reg = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        beta = infer_beta_at_k_star(entropy, reg, k_vals, 7)
        assert not np.isnan(beta)
        assert abs(beta - 5.0) < 0.01

    def test_infer_beta_at_k_star_boundary_returns_nan(self):
        k_vals = [3, 5, 7]
        entropy = np.array([1.0, 2.0, 3.0])
        reg = np.array([1.0, 2.0, 3.0])
        assert np.isnan(infer_beta_at_k_star(entropy, reg, k_vals, 3))
        assert np.isnan(infer_beta_at_k_star(entropy, reg, k_vals, 7))

    def test_infer_beta_at_k_star_not_in_list_returns_nan(self):
        k_vals = [3, 5, 7]
        entropy = np.array([1.0, 2.0, 3.0])
        reg = np.array([1.0, 2.0, 3.0])
        assert np.isnan(infer_beta_at_k_star(entropy, reg, k_vals, 4))

    def test_load_scale_calibration_missing_returns_none(self):
        assert load_scale_calibration("nonexistent_median.pt", tld=str(ROOT / "checkpoints")) is None
        assert load_scale_calibration(None, tld=str(ROOT)) is None

    def test_cost_at_beta(self):
        # Cost = β×⟨d⟩ − S
        e = np.array([1.0, 2.0, 3.0])
        r = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(cost_at_beta(e, r, 0.0), -e)
        np.testing.assert_allclose(cost_at_beta(e, r, 1.0), r - e)
        np.testing.assert_allclose(cost_at_beta(e, r, 2.0), 2 * r - e)

    def test_compute_posterior_stats_at_t_returns_dict_or_none(self):
        dataset, metadata = get_dataset("mnist", root=str(ROOT / "data"))
        mod = LocalScoreModule(
            dataset,
            kernel_size=3,
            image_size=metadata["image_size"],
            batch_size=min(16, len(dataset)),
            schedule=cosine_noise_schedule,
            max_samples=30,
        )
        x = torch.randn(2, metadata["num_channels"], metadata["image_size"], metadata["image_size"])
        t = torch.ones(1) * 0.5
        s = compute_posterior_stats_at_t(mod, t, x, device=torch.device("cpu"))
        assert s is not None
        assert set(s.keys()) == {"avg_entropy", "center_variance", "center_binder", "total_variance", "total_binder"}
        assert all(isinstance(v, (int, float)) for v in s.values())

    def test_compute_posterior_stats_at_t_module_without_method_returns_none(self):
        class NoStats(torch.nn.Module):
            pass
        s = compute_posterior_stats_at_t(NoStats(), torch.ones(1), torch.randn(1, 1, 8, 8))
        assert s is None

    def test_els_center_variance_nonnegative_and_finite(self):
        """ELS center_variance is posterior Var(center pixel of patch); must be >= 0 and finite."""
        dataset, metadata = get_dataset("mnist", root=str(ROOT / "data"))
        mod = LocalEquivScoreModule(
            dataset,
            kernel_size=3,
            image_size=metadata["image_size"],
            channels=metadata["num_channels"],
            batch_size=8,
            schedule=cosine_noise_schedule,
            max_samples=16,
        )
        x = torch.randn(1, metadata["num_channels"], metadata["image_size"], metadata["image_size"])
        t = torch.ones(1) * 0.5
        s = compute_posterior_stats_at_t(mod, t, x, device=torch.device("cpu"))
        assert s is not None
        cv = s["center_variance"]
        assert isinstance(cv, (int, float))
        assert np.isfinite(cv), "center_variance should be finite"
        assert cv >= 0.0, "center_variance (Var) should be nonnegative"

    def test_collect_stats_vs_k_returns_arrays(self):
        dataset, metadata = get_dataset("mnist", root=str(ROOT / "data"))
        modules = []
        for k in [3, 5]:
            mod = LocalScoreModule(
                dataset,
                kernel_size=k,
                image_size=metadata["image_size"],
                batch_size=min(16, len(dataset)),
                schedule=cosine_noise_schedule,
                max_samples=40,
            )
            modules.append((k, mod))
        t = torch.ones(1) * 0.5
        x = torch.randn(2, metadata["num_channels"], metadata["image_size"], metadata["image_size"])
        stats = collect_stats_vs_k(modules, t, x, torch.device("cpu"), None)
        assert stats["k_vals"] == [3, 5]
        assert stats["avg_entropy"].shape == (2,)
        assert stats["total_variance"].shape == (2,)

    def test_infer_betas_for_rate_distortion_without_median_empty(self):
        stats = {
            "k_vals": [3, 5, 7],
            "avg_entropy": np.array([1.0, 2.0, 3.0]),
            "center_variance": np.array([1.0, 2.0, 3.0]),
            "center_binder": np.array([1.0, 2.0, 3.0]),
            "total_variance": np.array([1.0, 2.0, 3.0]),
            "total_binder": np.array([1.0, 2.0, 3.0]),
        }
        betas = infer_betas_for_rate_distortion(stats, stats, None, None, 0)
        assert betas == {}


class TestPosteriorStats:
    """Test forward_with_posterior_stats for LS and ELS (idealscore)."""

    def test_ls_forward_with_posterior_stats_returns_five_tensors(self):
        dataset, metadata = get_dataset("mnist", root=str(ROOT / "data"))
        mod = LocalScoreModule(
            dataset,
            kernel_size=3,
            image_size=metadata["image_size"],
            batch_size=min(32, len(dataset)),
            schedule=cosine_noise_schedule,
            max_samples=50,
        )
        x = torch.randn(2, metadata["num_channels"], metadata["image_size"], metadata["image_size"])
        t = torch.ones(1) * 0.5
        out = mod.forward_with_posterior_stats(t, x, device=torch.device("cpu"))
        assert len(out) in (5, 7)
        score, E_x0, entropy_map, variance_map, binder_map = out[:5]
        assert score.shape == x.shape
        assert E_x0.shape == x.shape
        assert entropy_map.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert variance_map.shape == x.shape
        assert binder_map.shape == x.shape
        if len(out) == 7:
            patch_variance_map, patch_binder_map = out[5], out[6]
            assert patch_variance_map.shape == (x.shape[0], x.shape[2], x.shape[3])
            assert patch_binder_map.shape == (x.shape[0], x.shape[2], x.shape[3])

    def test_els_forward_with_posterior_stats_returns_five_tensors(self):
        dataset, metadata = get_dataset("mnist", root=str(ROOT / "data"))
        mod = LocalEquivScoreModule(
            dataset,
            kernel_size=3,
            batch_size=16,
            image_size=metadata["image_size"],
            channels=metadata["num_channels"],
            schedule=cosine_noise_schedule,
            max_samples=50,
        )
        x = torch.randn(2, metadata["num_channels"], metadata["image_size"], metadata["image_size"])
        t = torch.ones(1) * 0.5
        out = mod.forward_with_posterior_stats(t, x, device=torch.device("cpu"))
        assert len(out) in (5, 7)
        score, E_x0, entropy_map, variance_map, binder_map = out[:5]
        assert score.shape == x.shape
        assert E_x0.shape == x.shape
        assert entropy_map.shape == (x.shape[0], x.shape[2], x.shape[3])
        assert variance_map.shape == x.shape
        assert binder_map.shape == x.shape
        if len(out) == 7:
            patch_variance_map, patch_binder_map = out[5], out[6]
            assert patch_variance_map.shape == (x.shape[0], x.shape[2], x.shape[3])
            assert patch_binder_map.shape == (x.shape[0], x.shape[2], x.shape[3])

    def test_ls_entropy_bounded_by_log_N(self):
        """Discrete posterior entropy should satisfy 0 <= H <= log(N)."""
        from torch.utils.data import TensorDataset
        N, C, H, W = 24, 1, 8, 8
        images = torch.randn(N, C, H, W)
        labels = torch.zeros(N, dtype=torch.long)
        dset = TensorDataset(images, labels)
        mod = LocalScoreModule(
            dset,
            kernel_size=3,
            image_size=H,
            batch_size=8,
            schedule=cosine_noise_schedule,
            max_samples=N,
        )
        x = torch.randn(2, C, H, W)
        t = torch.ones(1) * 0.5
        out = mod.forward_with_posterior_stats(t, x, device=torch.device("cpu"))
        entropy_map = out[2]
        log_N = float(torch.log(torch.tensor(N)).item())
        tol = 0.5
        assert float(entropy_map.min()) >= -tol, "entropy should be >= 0 (discrete)"
        assert float(entropy_map.max()) <= log_N + tol, "entropy should be <= log N"

    def test_els_entropy_bounded_by_log_N(self):
        """Discrete posterior entropy should satisfy 0 <= H <= log(N_states). ELS uses patches: N_states = N_train * L with L = (H-k+1)*(W-k+1)."""
        from torch.utils.data import TensorDataset
        N, C, H, W = 24, 1, 8, 8
        k = 3
        L = (H - k + 1) * (W - k + 1)  # patch positions per image
        images = torch.randn(N, C, H, W)
        labels = torch.zeros(N, dtype=torch.long)
        dset = TensorDataset(images, labels)
        mod = LocalEquivScoreModule(
            dset,
            kernel_size=k,
            batch_size=8,
            image_size=H,
            channels=C,
            schedule=cosine_noise_schedule,
            max_samples=N,
        )
        x = torch.randn(2, C, H, W)
        t = torch.ones(1) * 0.5
        out = mod.forward_with_posterior_stats(t, x, device=torch.device("cpu"))
        entropy_map = out[2]
        log_N_states = float(torch.log(torch.tensor(N * L, dtype=torch.float64)).item())
        tol = 0.5
        assert float(entropy_map.min()) >= -tol, "entropy should be >= 0 (discrete)"
        assert float(entropy_map.max()) <= log_N_states + tol, "entropy should be <= log(N_states)"


@pytest.mark.slow
class TestChannelTheoryScript:
    """Test channel_theory_scale_calibration script (slow)."""

    def test_script_import_and_run_channel_theory(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import channel_theory_scale_calibration as cts
            assert hasattr(cts, "run_channel_theory")
            assert hasattr(cts, "main")
        finally:
            if str(ROOT / "scripts") in sys.path:
                sys.path.remove(str(ROOT / "scripts"))

    def test_run_channel_theory_smoke(self, tmp_path):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import channel_theory_scale_calibration as cts
            result = cts.run_channel_theory(
                dataset_name="mnist",
                kernelsizes=(3, 5),
                scale_file_ls=None,
                scale_file_els=None,
                tld=str(ROOT / "checkpoints"),
                nsteps=5,
                nsamples=2,
                timestep_idx=2,
                max_samples=80,
                score_batch_size=16,
                out_dir=str(tmp_path),
                cpu=True,
            )
            assert "k_vals" in result
            assert result["k_vals"] == [3, 5]
            assert "ls_entropy" in result
            assert len(result["ls_entropy"]) == 2
            assert "els_entropy" in result
            assert "betas" in result
            out_png = tmp_path / "channel_theory_stats_vs_k_mnist_t2.png"
            assert out_png.is_file()
        finally:
            if str(ROOT / "scripts") in sys.path:
                sys.path.remove(str(ROOT / "scripts"))
