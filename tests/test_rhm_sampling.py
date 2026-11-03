"""Tests for the patch-count score machine and the RHM grammar oracle."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.discrete_score import (
    LocalDiscreteScoreMachine,
    ScheduledDiscreteScoreMachine,
    mask_tokens,
)
from src.utils.random_hierarchy import RandomHierarchy2DDataset, RandomHierarchyModel2D
from src.utils.rhm_grammar import evaluate_samples, is_grammatical, is_novel, parse_height
from src.utils.rhm_sampling import PatchCountDiscreteScoreMachine, distinct_patch_table

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def rhm():
    return RandomHierarchyModel2D(v=4, s=2, L=3, m=3, seed=1000)


class TestGrammar:
    def test_true_samples_are_grammatical(self, rhm):
        grids, _ = rhm.sample(64, seed=1)
        assert is_grammatical(rhm, grids).all()
        assert (parse_height(rhm, grids) == rhm.L).all()

    def test_random_grids_are_not_grammatical(self, rhm):
        rng = np.random.default_rng(0)
        grids = rng.integers(0, rhm.v, size=(64, rhm.grid_size, rhm.grid_size))
        assert is_grammatical(rhm, grids).mean() < 0.05

    def test_novelty_is_exact_set_membership(self, rhm):
        train, _ = rhm.sample(32, seed=2)
        fresh, _ = rhm.sample(32, seed=3)
        assert not is_novel(train, train).any()
        assert is_novel(train, fresh).mean() > 0.5

    def test_evaluate_samples_reports_both_criteria(self, rhm):
        train, _ = rhm.sample(32, seed=2)
        res = evaluate_samples(rhm, train, train)
        assert res["grammatical"] == 1.0
        assert res["novel"] == 0.0
        assert res["success"] == 0.0
        assert len(res["level_parse_fraction"]) == rhm.L + 1


class TestPatchCountMachine:
    @pytest.mark.parametrize("p_mask", [0.3, 0.7, 1.0])
    def test_matches_reference_machine(self, rhm, p_mask):
        """block_align=1 must reproduce LocalDiscreteScoreMachine exactly."""
        grids, labels = rhm.sample(200, seed=5)
        ds = RandomHierarchy2DDataset(model=rhm, grids=grids, labels=labels, onehot=False)
        ref = LocalDiscreteScoreMachine(ds, vocab_size=rhm.v, kernel_size=3)
        fast = PatchCountDiscreteScoreMachine(grids, vocab_size=rhm.v, kernel_size=3, device=CPU)

        x = torch.as_tensor(grids[:8])
        xt, _ = mask_tokens(x, p_mask, mask_token=rhm.v, generator=torch.Generator().manual_seed(0))
        a = ref(torch.tensor([p_mask]), xt, device=CPU)
        b = fast(torch.tensor([p_mask]), xt)
        assert torch.allclose(a, b, atol=1e-6)

    def test_patch_table_counts_every_window(self, rhm):
        grids, _ = rhm.sample(10, seed=5)
        k = 3
        patches, counts = distinct_patch_table(grids, k, rhm.v)
        npos = (rhm.grid_size - k + 1) ** 2
        assert counts.sum() == 10 * npos
        assert patches.shape[1:] == (k, k)

    def test_phase_split_partitions_the_windows(self, rhm):
        """The s**2 phase tables together hold exactly the pooled table's windows."""
        grids, _ = rhm.sample(10, seed=5)
        k, s = 3, rhm.s
        total = distinct_patch_table(grids, k, rhm.v)[1].sum()
        split = sum(
            distinct_patch_table(grids, k, rhm.v, phase=(pi, pj), block_align=s)[1].sum()
            for pi in range(s) for pj in range(s)
        )
        assert total == split

    def test_full_window_machine_memorizes(self, rhm):
        """k = 2d - 1 is the smallest window that sees the whole grid from every site.

        A window of side k centred on a site reaches only k // 2, so k = d still
        leaves corner sites blind to the far half of the grid; 2d - 1 is the real
        full-window machine. There the ancestral sampler conditions each reveal on
        every earlier one over the entire grid, so its joint is exactly the
        empirical distribution of the training set: samples are grammatical and
        never novel. This is the ceiling the sweep measures against -- perfect
        grammar bought by zero generalization.
        """
        grids, _ = rhm.sample(64, seed=5)
        mach = PatchCountDiscreteScoreMachine(
            grids, vocab_size=rhm.v, kernel_size=2 * rhm.grid_size - 1, alpha_smooth=1e-9,
            device=CPU, pad_mode='outside',
        )
        sampler = ScheduledDiscreteScoreMachine(
            mach, vocab_size=rhm.v, grid_size=rhm.grid_size, mask_token=rhm.v, reveal_one=True,
        )
        gen = sampler.sample(n_samples=32, device=CPU, generator=torch.Generator().manual_seed(1)).numpy()
        assert is_grammatical(rhm, gen).all()
        assert not is_novel(grids, gen).any()

    @pytest.mark.parametrize("pad_mode", ["zeros", "outside"])
    def test_even_window_keeps_the_grid_shape(self, rhm, pad_mode):
        """Even k needs asymmetric padding, or the posterior comes back the wrong size."""
        grids, _ = rhm.sample(16, seed=5)
        for k in (2, 4, 8):
            mach = PatchCountDiscreteScoreMachine(
                grids, vocab_size=rhm.v, kernel_size=k, device=CPU, pad_mode=pad_mode
            )
            xt = torch.as_tensor(grids[:2])
            assert mach(torch.tensor([0.0]), xt).shape == (2, rhm.v, rhm.grid_size, rhm.grid_size)

    def test_outside_padding_is_a_proper_posterior(self, rhm):
        """Every site's posterior must still normalise, border sites included."""
        grids, _ = rhm.sample(64, seed=5)
        mach = PatchCountDiscreteScoreMachine(
            grids, vocab_size=rhm.v, kernel_size=5, device=CPU, pad_mode='outside'
        )
        xt, _ = mask_tokens(torch.as_tensor(grids[:4]), 0.6, mask_token=rhm.v,
                            generator=torch.Generator().manual_seed(0))
        post = mach(torch.tensor([0.6]), xt)
        assert torch.allclose(post.sum(1), torch.ones_like(post.sum(1)), atol=1e-5)

    def test_outside_padding_recovers_true_tokens(self, rhm):
        """Lightly masked true grids must denoise back accurately at a window
        the training set can actually fill (k = 3 here; k = 5 needs far more
        than 2000 grids, which is the sampling complexity this all measures)."""
        grids, _ = rhm.sample(2000, seed=5)
        mach = PatchCountDiscreteScoreMachine(
            grids, vocab_size=rhm.v, kernel_size=3, device=CPU, pad_mode='outside', block_align=2
        )
        held, _ = rhm.sample(32, seed=77)
        xt, msk = mask_tokens(torch.as_tensor(held), 0.15, mask_token=rhm.v,
                              generator=torch.Generator().manual_seed(0))
        rec = mach(torch.tensor([0.15]), xt).argmax(1).numpy()
        acc = (rec[msk.numpy()] == held[msk.numpy()]).mean()
        assert acc > 0.75, f"token accuracy at 15% masking was only {acc:.3f}"
