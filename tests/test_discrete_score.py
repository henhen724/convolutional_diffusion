"""Tests for src.utils.discrete_score (LocalDiscreteScoreMachine and friends).

Correctness of LocalDiscreteScoreMachine is checked against a slow,
independent brute-force reference implementation (plain Python loops, no
tensor tricks) rather than against golden numbers, since the tensor-based
implementation and the brute-force one are unlikely to share a bug.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.discrete_score import (
    LocalDiscreteScoreMachine,
    ScheduledDiscreteScoreMachine,
    cosine_mask_schedule,
    linear_mask_schedule,
    mask_tokens,
)


class ListGridDataset(Dataset):
    """Minimal (grid, label) dataset for tests: grids is a list of 2D int arrays."""

    def __init__(self, grids, labels=None):
        self.grids = [np.asarray(g, dtype=np.int64) for g in grids]
        self.labels = [0] * len(self.grids) if labels is None else list(labels)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return torch.from_numpy(self.grids[idx]).long(), torch.tensor(self.labels[idx], dtype=torch.long)


def brute_force_posterior(xt, mask_token, images, k, V, alpha):
    """Independent reference implementation of the exact Bayes posterior."""
    b, h, w = xt.shape
    d = k // 2
    posterior = np.zeros((b, V, h, w))
    for bi in range(b):
        for i in range(h):
            for j in range(w):
                counts = np.zeros(V)
                n_match = 0
                for img in images:
                    H, W = img.shape
                    for pi in range(H - k + 1):
                        for pj in range(W - k + 1):
                            ok = True
                            for di in range(k):
                                for dj in range(k):
                                    qi, qj = i - d + di, j - d + dj
                                    if 0 <= qi < h and 0 <= qj < w and xt[bi, qi, qj] != mask_token:
                                        if xt[bi, qi, qj] != img[pi + di, pj + dj]:
                                            ok = False
                                            break
                                if not ok:
                                    break
                            if ok:
                                counts[img[pi + d, pj + d]] += 1
                                n_match += 1
                posterior[bi, :, i, j] = (counts + alpha) / (n_match + alpha * V)
    for bi in range(b):
        for i in range(h):
            for j in range(w):
                if xt[bi, i, j] != mask_token:
                    posterior[bi, :, i, j] = 0.0
                    posterior[bi, xt[bi, i, j], i, j] = 1.0
    return posterior


class TestMaskSchedulesAndCorruption:

    def test_linear_schedule_endpoints(self):
        assert linear_mask_schedule(0.0) == 0.0
        assert linear_mask_schedule(1.0) == 1.0

    def test_cosine_schedule_endpoints_and_monotone(self):
        assert cosine_mask_schedule(0.0) == pytest.approx(0.0, abs=1e-8)
        assert cosine_mask_schedule(1.0) == pytest.approx(1.0, abs=1e-8)
        ts = torch.linspace(0, 1, 11)
        vals = cosine_mask_schedule(ts)
        assert torch.all(vals[1:] >= vals[:-1])

    def test_mask_tokens_p0_p1(self):
        x0 = torch.randint(0, 4, (3, 5, 5))
        xt0, mask0 = mask_tokens(x0, 0.0, mask_token=4)
        assert not mask0.any()
        torch.testing.assert_close(xt0, x0)

        xt1, mask1 = mask_tokens(x0, 1.0, mask_token=4)
        assert bool(mask1.all())
        assert bool((xt1 == 4).all())

    def test_mask_tokens_matches_mask_flag(self):
        x0 = torch.randint(0, 4, (2, 6, 6))
        xt, mask = mask_tokens(x0, 0.5, mask_token=4, generator=torch.Generator().manual_seed(0))
        assert torch.equal(mask, xt == 4)
        assert torch.equal(xt[~mask], x0[~mask])


class TestLocalDiscreteScoreMachine:

    def test_posterior_sums_to_one(self):
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 4, size=(5, 5)) for _ in range(3)]
        ds = ListGridDataset(images)
        machine = LocalDiscreteScoreMachine(ds, vocab_size=4, kernel_size=3, batch_size=8, alpha_smooth=1.0)

        xt = torch.from_numpy(images[0]).clone().unsqueeze(0)
        xt[0, 1, 1] = machine.mask_token
        xt[0, 2, 3] = machine.mask_token
        posterior = machine(torch.tensor([0.5]), xt)
        assert posterior.shape == (1, 4, 5, 5)
        torch.testing.assert_close(posterior.sum(dim=1), torch.ones(1, 5, 5), atol=1e-5, rtol=1e-5)

    def test_observed_sites_collapse_to_delta(self):
        rng = np.random.default_rng(1)
        images = [rng.integers(0, 3, size=(4, 4)) for _ in range(2)]
        ds = ListGridDataset(images)
        machine = LocalDiscreteScoreMachine(ds, vocab_size=3, kernel_size=3, alpha_smooth=1.0)
        xt = torch.from_numpy(images[0]).clone().unsqueeze(0)  # fully observed
        posterior = machine(torch.tensor([0.1]), xt)
        expected = torch.nn.functional.one_hot(xt, num_classes=3).permute(0, 3, 1, 2).float()
        torch.testing.assert_close(posterior, expected)

    def test_matches_brute_force_reference(self):
        img0 = np.array(
            [
                [0, 1, 2, 0],
                [1, 2, 0, 1],
                [2, 0, 1, 2],
                [0, 1, 2, 0],
            ]
        )
        img1 = np.array(
            [
                [1, 0, 2, 1],
                [0, 2, 1, 0],
                [2, 1, 0, 2],
                [1, 0, 2, 1],
            ]
        )
        images = [img0, img1]
        ds = ListGridDataset(images)
        V, k, alpha, mask_token = 3, 3, 1.0, 3
        machine = LocalDiscreteScoreMachine(
            ds, vocab_size=V, kernel_size=k, batch_size=8, alpha_smooth=alpha, mask_token=mask_token
        )

        xt = img0.copy()
        xt[1, 1] = mask_token
        xt[1, 2] = mask_token
        xt_t = torch.from_numpy(xt).long().unsqueeze(0)

        got = machine(torch.tensor([0.4]), xt_t).numpy()
        expected = brute_force_posterior(xt[None, :, :], mask_token, images, k, V, alpha)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_fully_masked_gives_translation_invariant_marginal(self):
        rng = np.random.default_rng(2)
        images = [rng.integers(0, 3, size=(5, 5)) for _ in range(4)]
        ds = ListGridDataset(images)
        V, k, alpha, mask_token = 3, 3, 1.0, 3
        machine = LocalDiscreteScoreMachine(ds, vocab_size=V, kernel_size=k, alpha_smooth=alpha, mask_token=mask_token)

        xt = torch.full((1, 5, 5), mask_token, dtype=torch.long)
        posterior = machine(torch.tensor([1.0]), xt).numpy()

        expected = brute_force_posterior(xt.numpy(), mask_token, images, k, V, alpha)
        np.testing.assert_allclose(posterior, expected, atol=1e-5)
        # translation invariance: every site should see the identical marginal
        flat = posterior.reshape(V, -1).T
        for row in flat[1:]:
            np.testing.assert_allclose(row, flat[0], atol=1e-6)

    def test_label_filtering(self):
        rng = np.random.default_rng(3)
        img_a = rng.integers(0, 3, size=(4, 4))
        img_b = rng.integers(0, 3, size=(4, 4))
        ds = ListGridDataset([img_a, img_b], labels=[0, 1])
        V, k, alpha, mask_token = 3, 3, 1.0, 3
        machine = LocalDiscreteScoreMachine(ds, vocab_size=V, kernel_size=k, alpha_smooth=alpha, mask_token=mask_token)

        xt = img_a.copy()
        xt[1, 1] = mask_token
        xt_t = torch.from_numpy(xt).long().unsqueeze(0)

        got = machine(torch.tensor([0.4]), xt_t, label=0).numpy()
        expected = brute_force_posterior(xt[None, :, :], mask_token, [img_a], k, V, alpha)
        np.testing.assert_allclose(got, expected, atol=1e-5)


class TestScheduledDiscreteScoreMachine:

    def test_sample_fully_reveals_and_stays_in_vocab(self):
        rng = np.random.default_rng(4)
        images = [rng.integers(0, 3, size=(4, 4)) for _ in range(5)]
        ds = ListGridDataset(images)
        machine = LocalDiscreteScoreMachine(ds, vocab_size=3, kernel_size=3, alpha_smooth=1.0)
        sampler = ScheduledDiscreteScoreMachine(machine, vocab_size=3, grid_size=4, default_time_steps=8)

        gen = torch.Generator().manual_seed(0)
        out = sampler.sample(n_samples=2, device=torch.device("cpu"), generator=gen)
        assert out.shape == (2, 4, 4)
        assert bool((out != machine.mask_token).all())
        assert int(out.min()) >= 0 and int(out.max()) < 3

    def test_sample_is_deterministic_given_generator_seed(self):
        rng = np.random.default_rng(5)
        images = [rng.integers(0, 3, size=(4, 4)) for _ in range(5)]
        ds = ListGridDataset(images)
        machine = LocalDiscreteScoreMachine(ds, vocab_size=3, kernel_size=3, alpha_smooth=1.0)
        sampler = ScheduledDiscreteScoreMachine(machine, vocab_size=3, grid_size=4, default_time_steps=8)

        out1 = sampler.sample(n_samples=2, device=torch.device("cpu"), generator=torch.Generator().manual_seed(123))
        out2 = sampler.sample(n_samples=2, device=torch.device("cpu"), generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(out1, out2)

    def test_t_start_partial_corruption_leaves_observed_sites_untouched(self):
        rng = np.random.default_rng(6)
        images = [rng.integers(0, 3, size=(4, 4)) for _ in range(5)]
        ds = ListGridDataset(images)
        machine = LocalDiscreteScoreMachine(ds, vocab_size=3, kernel_size=3, alpha_smooth=1.0)
        sampler = ScheduledDiscreteScoreMachine(machine, vocab_size=3, grid_size=4, default_time_steps=10)

        x0 = torch.from_numpy(images[0]).long().unsqueeze(0)
        xt, mask = mask_tokens(x0, linear_mask_schedule(0.5), machine.mask_token, generator=torch.Generator().manual_seed(1))

        out = sampler(xt, nsteps=10, t_start=0.5, device=torch.device("cpu"), generator=torch.Generator().manual_seed(2))
        assert bool((out != machine.mask_token).all())
        # sites already observed at t_start must be left exactly as they were
        torch.testing.assert_close(out[~mask], x0[~mask])

    def test_t_start_zero_is_a_no_op_on_clean_input(self):
        rng = np.random.default_rng(7)
        images = [rng.integers(0, 3, size=(4, 4)) for _ in range(5)]
        ds = ListGridDataset(images)
        machine = LocalDiscreteScoreMachine(ds, vocab_size=3, kernel_size=3, alpha_smooth=1.0)
        sampler = ScheduledDiscreteScoreMachine(machine, vocab_size=3, grid_size=4, default_time_steps=10)

        x0 = torch.from_numpy(images[0]).long().unsqueeze(0)
        out = sampler(x0.clone(), nsteps=10, t_start=0.0, device=torch.device("cpu"))
        torch.testing.assert_close(out, x0)
