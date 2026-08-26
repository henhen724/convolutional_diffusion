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
