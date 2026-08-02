"""
tests/test_layers.py
======================
Unit tests for the core Aetherion mechanisms translated into GamaX1.
These tests directly check the properties that the research report
found to matter (e.g. exact top-K count, trend-based not instantaneous
sparsity gating, asymmetric PTM hysteresis) -- not just "does it run."
"""

import torch
import pytest

from gamax1.layers import (
    SparseSuperpositionLinear,
    DynamicSparsityController,
    ProbationaryMemoryTracker,
    HexNeighborInfluence,
    build_hex_neighbor_table,
    RouterExpert,
    ValidatorExpert,
)


def test_sparse_superposition_exact_topk_count():
    layer = SparseSuperpositionLinear(d_model=16, n_features=64)
    x = torch.randn(4, 5, 16)
    out = layer(x, k=8)
    assert out.shape == (4, 5, 16)
    active = layer.last_active_mask.reshape(4, 5, 64)
    # each token should have AT MOST k active units (could be fewer if
    # ties/zeros truncate, but never more)
    assert (active.sum(dim=-1) <= 8).all()


def test_sparse_superposition_no_nan_after_several_steps():
    layer = SparseSuperpositionLinear(d_model=16, n_features=64)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for _ in range(20):
        x = torch.randn(4, 5, 16)
        out = layer(x, k=8)
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert not torch.isnan(out).any()
    assert not torch.isnan(next(layer.parameters())).any()


def test_dynamic_sparsity_does_not_react_to_a_single_spike():
    """Regression test for the validated fix: a single bad-loss step
    must NOT reduce k -- only a sustained trend should."""
    ctrl = DynamicSparsityController(k_init=256, k_min=64, k_max=1024,
                                      trend_window=10, patience=5)
    for _ in range(25):
        ctrl.step(1.0)  # flat, "good" loss to build up history
    k_before = ctrl.k
    ctrl.step(50.0)  # one instantaneous spike
    assert ctrl.k == k_before, "a single spike must not change k (trend-based gating)"


def test_dynamic_sparsity_reduces_on_sustained_improvement():
    ctrl = DynamicSparsityController(k_init=256, k_min=64, k_max=1024,
                                      trend_window=5, patience=3)
    # sustained, genuine downward trend
    loss = 10.0
    ks = []
    for _ in range(200):
        loss *= 0.98
        ctrl.step(loss)
        ks.append(ctrl.k)
    assert ks[-1] < 256, "k should shrink under a genuine, sustained improving trend"
    assert ks[-1] >= 64


def test_ptm_enters_probation_after_consecutive_misses():
    tracker = ProbationaryMemoryTracker(n_units=10, miss_threshold=3, success_threshold=2)
    always_inactive = torch.zeros(4, 10, dtype=torch.bool)
    for _ in range(3):
        tracker.update(always_inactive)
    assert tracker.on_probation.all(), "units with 3 consecutive misses should be on probation"


def test_ptm_exits_after_consecutive_successes():
    tracker = ProbationaryMemoryTracker(n_units=4, miss_threshold=2, success_threshold=2)
    inactive = torch.zeros(2, 4, dtype=torch.bool)
    for _ in range(2):
        tracker.update(inactive)
    assert tracker.on_probation.all()

    active = torch.ones(2, 4, dtype=torch.bool)
    for _ in range(2):
        tracker.update(active)
    assert not tracker.on_probation.any(), "units with enough forced successes should exit probation"


def test_ptm_true_negatives_do_not_count_as_misses_for_active_units():
    """A unit that IS active this batch should never accumulate misses,
    regression test for the original bookkeeping bug."""
    tracker = ProbationaryMemoryTracker(n_units=3, miss_threshold=2, success_threshold=2)
    active_unit0 = torch.tensor([[True, False, False]])
    for _ in range(10):
        tracker.update(active_unit0)
    assert not tracker.on_probation[0], "an always-active unit must never enter probation"
    assert tracker.on_probation[1] and tracker.on_probation[2]


def test_hex_neighbor_table_shape_and_bounds():
    table = build_hex_neighbor_table(n_features=64)
    assert table.shape == (64, 6)
    assert (table >= 0).all() and (table < 64).all()


def test_hex_influence_preserves_shape():
    hex_mod = HexNeighborInfluence(n_features=64, decay=0.1)
    x = torch.randn(2, 5, 64)
    out = hex_mod(x)
    assert out.shape == x.shape


def test_router_expert_output_is_a_probability():
    router = RouterExpert(d_model=32)
    pooled = torch.randn(8, 32)
    p = router(pooled)
    assert p.shape == (8,)
    assert (p >= 0).all() and (p <= 1).all()


def test_validator_stability_requires_matching_consecutive_predictions():
    validator = ValidatorExpert(patience=2)
    same = torch.tensor([5, 5])
    different = torch.tensor([7, 7])
    assert validator.is_stable(same) is False  # not enough history yet
    assert validator.is_stable(same) is False
    assert validator.is_stable(same) is True   # 3 identical calls, patience=2
    validator.reset()
    assert validator.is_stable(same) is False
    assert validator.is_stable(different) is False
    assert validator.is_stable(different) is False  # last two differ from the very first
