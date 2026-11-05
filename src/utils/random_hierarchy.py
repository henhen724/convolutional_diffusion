"""
2D Random Hierarchy Model (RHM).

The Random Hierarchy Model (Cagnetta et al., "Deep Networks as Denoisers"
/ "How Deep Networks Learn Compositional Data") is a synthetic generative
model of discrete, compositional data defined by a random probabilistic
grammar: a fixed alphabet of ``v`` symbols, a small number ``m`` of
synonymous production rules per symbol at each of ``L`` levels, and
recursive expansion of a root symbol down to a grid of leaf tokens.

This module implements the natural 2D generalization used for image-shaped
discrete data (as opposed to the usual 1D sequence version): each symbol
expands into an ``s`` x ``s`` block of child symbols instead of a length-``s``
tuple, so after ``L`` levels a single root symbol has grown into a
``s**L`` x ``s**L`` grid of leaf tokens.

Generative process
-------------------
1. Fix, once, ``m`` random "words" per symbol per level: for level
   ``l = 1..L`` and every symbol ``a`` in ``{0, ..., v-1}``, draw ``m``
   i.i.d. uniform ``s`` x ``s`` arrays of symbols in ``{0, ..., v-1}``. These
   are the production rules and are shared by every sample (they define the
   hidden compositional structure of the dataset).
2. For each sample, draw a root symbol ``z_0 ~ Uniform({0, ..., v-1})``.
3. For ``l = 1, ..., L``: replace every current symbol independently by one
   of its ``m`` words, chosen uniformly at random, laying out the word's
   ``s`` x ``s`` entries as the corresponding block of the (``s``-times
   larger) next grid.
4. After ``L`` levels the grid has shape ``(s**L, s**L)`` and its entries are
   the leaf tokens; ``z_0`` is a natural root-level class label.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Array = np.ndarray


class RandomHierarchyModel2D:
    """Fixed random production rules for a 2D Random Hierarchy Model.

    Parameters
    ----------
    v : alphabet size; symbols take values in ``{0, ..., v-1}`` at every level.
    s : branching factor; each symbol expands into an ``s`` x ``s`` block of children.
    L : number of hierarchy levels; final grid is ``s**L`` x ``s**L``.
    m : number of synonymous production rules ("words") per symbol per level.
    seed : RNG seed for drawing the production rules. ``None`` -> non-deterministic.
    """

    def __init__(self, v: int = 8, s: int = 2, L: int = 3, m: int = 2, seed: Optional[int] = None):
        if v < 2:
            raise ValueError("v must be >= 2")
        if s < 2:
            raise ValueError("s must be >= 2")
        if L < 1:
            raise ValueError("L must be >= 1")
        if m < 1:
            raise ValueError("m must be >= 1")

        self.v = v
        self.s = s
        self.L = L
        self.m = m
        self.grid_size = s ** L

        rng = np.random.default_rng(seed)
        # rules[l] has shape (v, m, s, s): m candidate s x s child blocks per parent symbol.
        self.rules = [rng.integers(0, v, size=(v, m, s, s), dtype=np.int64) for _ in range(L)]

    def sample(self, n_samples: int, seed: Optional[int] = None, return_hierarchy: bool = False):
        """Draw ``n_samples`` i.i.d. grids from the model.

        Returns
        -------
        grid : np.ndarray, shape (n_samples, s**L, s**L), dtype int64, leaf tokens in [0, v).
        root : np.ndarray, shape (n_samples,), dtype int64, root symbols in [0, v).
        levels : list of np.ndarray (only if return_hierarchy=True), one per level
            0..L, with level ``l`` having shape (n_samples, s**l, s**l).
        """
        if n_samples < 0:
            raise ValueError("n_samples must be nonnegative")

        rng = np.random.default_rng(seed)
        root = rng.integers(0, self.v, size=(n_samples,), dtype=np.int64)
        grid = root[:, None, None]
        levels = [grid]

        for rule in self.rules:
            n, h, w = grid.shape
            word_idx = rng.integers(0, self.m, size=(n, h, w))
            words = rule[grid, word_idx]  # (n, h, w, s, s) via fancy indexing
            grid = words.transpose(0, 1, 3, 2, 4).reshape(n, h * self.s, w * self.s)
            levels.append(grid)

        if return_hierarchy:
            return grid, root, levels
        return grid, root

    def belief_propagation(self, xt: Array, mask_token: Optional[int] = None):
        """Exact posterior marginals at every level, via belief propagation.

        This is the same exact-inference method used for Figs. 5/7 of
        Sclocchi, Favero & Wyart (PNAS 2025) and Fig. 2 of Sclocchi, Favero,
        Levi & Wyart (ICLR 2025): since the generative model is a tree (each
        node's ``s x s`` block of children is drawn from one of its ``m``
        known, fixed candidate words, independent of everything outside its
        own subtree given its own symbol), belief propagation is *exact*,
        not an approximation -- there are no loops to break sum-product's
        exactness.

        Parameters
        ----------
        xt : (N, s**L, s**L) int array of observed leaf tokens, with sites
            equal to ``mask_token`` treated as unobserved (erased).
        mask_token : sentinel marking an unobserved site; defaults to ``v``.

        Returns
        -------
        marginals : list of (N, s**l, s**l, v) float arrays, l = 0..L
            (root..leaf), the exact posterior P(z_l = a | observed leaves)
            for every node.
        """
        if mask_token is None:
            mask_token = self.v
        xt = np.asarray(xt)
        n = xt.shape[0]
        v, s, L, m = self.v, self.s, self.L, self.m

        observed = xt != mask_token
        safe_xt = np.where(observed, xt, 0)
        leaf_up = np.eye(v, dtype=np.float64)[safe_xt]  # (n, gs, gs, v): one-hot if observed
        leaf_up[~observed] = 1.0 / v  # uniform likelihood if masked

        up = self._bp_upward(leaf_up)
        down = self._bp_downward(up)

        marginals = []
        for l in range(L + 1):
            combined = up[l] * down[l]
            combined /= combined.sum(axis=-1, keepdims=True).clip(min=1e-300)
            marginals.append(combined)
        return marginals

    def _bp_upward(self, leaf_up: Array):
        """up[l][n, i, j, a] ~ P(observed leaves under this subtree | z_l = a), normalized."""
        v, s, L, m = self.v, self.s, self.L, self.m
        up = [None] * (L + 1)
        up[L] = leaf_up
        current = leaf_up
        for l in range(L - 1, -1, -1):
            rule = self.rules[l]  # (v, m, s, s)
            n, h, w, _ = current.shape
            nb = h // s
            # (n, nb, nb, s, s, v): group leaf/child grid into this level's s x s blocks
            blocks = current.reshape(n, nb, s, nb, s, v).transpose(0, 1, 3, 2, 4, 5)
            new_up = np.zeros((n, nb, nb, v), dtype=np.float64)
            for a in range(v):
                word_probs = np.empty((m, n, nb, nb), dtype=np.float64)
                for wi in range(m):
                    prod = np.ones((n, nb, nb), dtype=np.float64)
                    for di in range(s):
                        for dj in range(s):
                            prod = prod * blocks[:, :, :, di, dj, rule[a, wi, di, dj]]
                    word_probs[wi] = prod
                new_up[:, :, :, a] = word_probs.mean(axis=0)
            new_up /= new_up.sum(axis=-1, keepdims=True).clip(min=1e-300)
            up[l] = new_up
            current = new_up
        return up

    def _bp_downward(self, up):
        """down[l][n, i, j, a] ~ P(z_l = a | everything outside this node's subtree)."""
        v, s, L, m = self.v, self.s, self.L, self.m
        n = up[0].shape[0]
        down = [None] * (L + 1)
        down[0] = np.full((n, 1, 1, v), 1.0 / v)  # root prior is uniform by construction
        for l in range(L):
            rule = self.rules[l]  # (v, m, s, s)
            parent_down = down[l]  # (n, nb, nb, v)
            nb = parent_down.shape[1]
            # children's own upward messages, grouped into this level's s x s blocks
            blocks_up = up[l + 1].reshape(n, nb, s, nb, s, v).transpose(0, 1, 3, 2, 4, 5)
            new_blocks = np.zeros((n, nb, nb, s, s, v), dtype=np.float64)
            for a in range(v):
                weight = parent_down[:, :, :, a] * (1.0 / m)  # (n, nb, nb)
                for wi in range(m):
                    for di_t in range(s):
                        for dj_t in range(s):
                            # leave-one-out product over siblings (excluding the target child)
                            loo = np.ones((n, nb, nb), dtype=np.float64)
                            for di in range(s):
                                for dj in range(s):
                                    if (di, dj) == (di_t, dj_t):
                                        continue
                                    loo = loo * blocks_up[:, :, :, di, dj, rule[a, wi, di, dj]]
                            target_symbol = rule[a, wi, di_t, dj_t]
                            new_blocks[:, :, :, di_t, dj_t, target_symbol] += weight * loo
            new_down = new_blocks.transpose(0, 1, 3, 2, 4, 5).reshape(n, nb * s, nb * s, v)
            new_down /= new_down.sum(axis=-1, keepdims=True).clip(min=1e-300)
            down[l + 1] = new_down
        return down

    def bp_expected_correct_probability(self, xt: Array, levels: "list[Array]", mask_token: Optional[int] = None):
        """Exact per-level "probability correct": the posterior mass BP
        assigns to the *true* symbol at each node, averaged over all nodes
        and samples at that level.

        This is the quantity plotted in PNAS Figs. 5/7 -- not a thresholded
        MAP-matches-truth indicator, but the exact expected accuracy of a
        Bayes-optimal probabilistic reconstruction, with no Monte Carlo
        sampling noise beyond which corrupted instances are averaged over.

        Parameters
        ----------
        xt : (N, s**L, s**L) observed (possibly masked) leaf grids.
        levels : ground-truth ``levels[l]`` for l = 0..L, as returned by
            ``sample(..., return_hierarchy=True)``, matching ``xt``.

        Returns
        -------
        p_correct : list of floats, one per level 0..L (root..leaf).
        """
        marginals = self.belief_propagation(xt, mask_token=mask_token)
        p_correct = []
        for l, post in enumerate(marginals):
            true_vals = levels[l]  # (N, s^l, s^l)
            p_true = np.take_along_axis(post, true_vals[..., None], axis=-1)[..., 0]
            p_correct.append(float(p_true.mean()))
        return p_correct


class RandomHierarchy2DDataset(Dataset):
    """Torch ``Dataset`` of 2D Random Hierarchy Model samples.

    Each item is ``(x, label)`` where ``label`` is the root symbol and ``x``
    is either the raw leaf-token grid (``LongTensor`` of shape ``(H, W)``,
    the format expected by :class:`~src.utils.discrete_score.LocalDiscreteScoreMachine`)
    or, if ``onehot=True``, a one-hot encoded ``FloatTensor`` of shape ``(v, H, W)``.
    """

    def __init__(
        self,
        n_samples: Optional[int] = None,
        v: int = 8,
        s: int = 2,
        L: int = 3,
        m: int = 2,
        seed: Optional[int] = None,
        model: Optional[RandomHierarchyModel2D] = None,
        grids: Optional[Array] = None,
        labels: Optional[Array] = None,
        onehot: bool = False,
    ):
        if model is None:
            model = RandomHierarchyModel2D(v=v, s=s, L=L, m=m, seed=seed)
        self.model = model
        self.v = model.v
        self.onehot = onehot

        if grids is None:
            if n_samples is None:
                raise ValueError("Provide either n_samples or precomputed grids/labels.")
            sample_seed = None if seed is None else seed + 1
            grids, labels = model.sample(n_samples, seed=sample_seed)

        self.grids = np.asarray(grids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.grids.shape[0])

    def __getitem__(self, idx):
        grid = torch.from_numpy(np.array(self.grids[idx], copy=True)).long()
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.onehot:
            grid = F.one_hot(grid, num_classes=self.v).permute(2, 0, 1).float()
        return grid, label


def sample_random_hierarchy_model_2d(
    n_samples: int,
    v: int = 8,
    s: int = 2,
    L: int = 3,
    m: int = 2,
    seed: Optional[int] = None,
    return_hierarchy: bool = False,
):
    """Functional convenience wrapper around :class:`RandomHierarchyModel2D`.

    Draws a fresh model (production rules) and samples from it in one call;
    see :class:`RandomHierarchyModel2D` for parameter semantics. Use the
    class directly when you need to draw multiple batches from the *same*
    fixed rules.
    """
    model = RandomHierarchyModel2D(v=v, s=s, L=L, m=m, seed=seed)
    sample_seed = None if seed is None else seed + 1
    return model.sample(n_samples, seed=sample_seed, return_hierarchy=return_hierarchy)
