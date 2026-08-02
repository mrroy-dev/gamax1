"""Tests for engineering safeguards in the training workflow."""

import math

import torch
import torch.nn as nn

from gamax1.model import GamaX1Model
from gamax1.tokenizer import CharTokenizer, word_tokenizer_warning
from gamax1.train import (
    auto_size_model,
    checkpoint_dict,
    create_optimizer,
    evaluate_loss,
    get_lr_schedule,
    is_memorization_detected,
    is_overfitting,
    perplexity,
    split_training_windows,
    tokens_per_parameter,
)


class MeanTargetLossModel(nn.Module):
    """Fixed model whose loss exposes sampling noise for evaluator tests."""

    def forward(self, idx, targets=None, k=None):
        return None, targets.float().mean()


def test_perplexity_is_exponential_and_saturates_instead_of_overflowing():
    assert perplexity(0.0) == 1.0
    assert perplexity(math.log(10.0)) == pytest_approx(10.0)
    assert math.isinf(perplexity(1e6))


def pytest_approx(value):
    """Avoid a broad behavioral test: use a tight numeric tolerance for exp."""
    import pytest
    return pytest.approx(value)


def test_lr_schedule_warms_up_then_decays_to_ten_percent():
    assert get_lr_schedule(0, 100, 1.0, 10) == 0.0
    assert get_lr_schedule(10, 100, 1.0, 10) == 1.0
    assert get_lr_schedule(100, 100, 1.0, 10) == pytest_approx(0.1)
    assert get_lr_schedule(50, 100, 1.0, 10) < 1.0


def test_overfit_detection_requires_opposing_consecutive_trends():
    assert is_overfitting([1.0, 1.1, 1.2, 1.3], [2.0, 1.9, 1.8, 1.7], 3)
    assert not is_overfitting([1.0, 1.1, 1.05, 1.2], [2.0, 1.9, 1.8, 1.7], 3)
    assert not is_overfitting([1.0, 1.1, 1.2, 1.3], [2.0, 1.9, 1.95, 1.7], 3)


def test_memorization_detection_flags_low_perplexity_with_oversized_model():
    synthetic_corpus = torch.arange(100, dtype=torch.long) % 5
    model = GamaX1Model(vocab_size=5, d_model=32, n_heads=2, n_layers=2, n_features=128,
                         max_seq_len=8, sparsity_k_init=64, sparsity_k_min=16)
    assert is_memorization_detected(
        token_count=len(synthetic_corpus),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        train_loss=0.01, val_loss=0.02,
        min_tokens_per_param=10, perplexity_memorization_floor=1.5,
    )


def test_memorization_detection_ignores_healthy_capacity_ratio():
    synthetic_corpus = torch.arange(10_000, dtype=torch.long) % 5
    model = GamaX1Model(vocab_size=5, d_model=1, n_heads=1, n_layers=1, n_features=1,
                         max_seq_len=8, sparsity_k_init=1, sparsity_k_min=1)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert tokens_per_parameter(len(synthetic_corpus), parameter_count) > 10
    assert not is_memorization_detected(
        token_count=len(synthetic_corpus), parameter_count=parameter_count, train_loss=0.01, val_loss=0.02,
        min_tokens_per_param=10, perplexity_memorization_floor=1.5,
    )


def test_create_optimizer_uses_requested_weight_decay():
    model = GamaX1Model(vocab_size=5, d_model=16, n_heads=2, n_layers=1, n_features=32,
                         max_seq_len=8, sparsity_k_init=16, sparsity_k_min=4)
    optimizer = create_optimizer(model, lr=3e-4, weight_decay=0.123)
    assert optimizer.param_groups[0]["weight_decay"] == pytest_approx(0.123)


def test_multibatch_validation_evaluation_has_lower_sampling_variance():
    model = MeanTargetLossModel()
    val_data = torch.arange(512, dtype=torch.long) % 97
    single_batch_estimates = []
    averaged_estimates = []
    for seed in range(40):
        torch.manual_seed(seed)
        single_batch_estimates.append(evaluate_loss(model, val_data, 8, 4, 1, "cpu"))
        torch.manual_seed(seed)
        averaged_estimates.append(evaluate_loss(model, val_data, 8, 4, 10, "cpu"))
    assert torch.tensor(averaged_estimates).var() < torch.tensor(single_batch_estimates).var()


def test_random_window_validation_split_is_reproducible_and_non_overlapping():
    data = torch.arange(200, dtype=torch.long)
    _, _, train_starts, val_starts = split_training_windows(
        data, block_size=8, validation_fraction=0.2, strategy="random_windows", seed=7,
    )
    _, _, repeat_train_starts, repeat_val_starts = split_training_windows(
        data, block_size=8, validation_fraction=0.2, strategy="random_windows", seed=7,
    )
    assert torch.equal(train_starts, repeat_train_starts)
    assert torch.equal(val_starts, repeat_val_starts)
    train_tokens = {token for start in train_starts.tolist() for token in range(start, start + 9)}
    val_tokens = {token for start in val_starts.tolist() for token in range(start, start + 9)}
    assert train_tokens.isdisjoint(val_tokens)


def test_auto_size_model_respects_viable_architecture_floors_at_every_scale():
    for token_count in (100, 100_000, 10_000_000):
        result = auto_size_model(
            token_count=token_count, vocab_size=32, block_size=16, n_heads=1,
            target_tokens_per_param=40, min_tokens_per_param=10,
        )
        assert result.d_model >= 64
        assert result.n_layers >= 2
        assert result.n_heads >= 2
        assert result.d_model % result.n_heads == 0
        assert result.n_features >= 4 * result.d_model


def test_auto_size_model_warns_by_status_instead_of_shrinking_below_floor():
    result = auto_size_model(
        token_count=100, vocab_size=32, block_size=16, n_heads=4,
        target_tokens_per_param=40, min_tokens_per_param=10,
    )
    assert result.status == "hard_floor"
    assert result.d_model == 64 and result.n_layers == 2 and result.n_features == 256


def test_auto_size_model_scales_up_for_large_corpus():
    result = auto_size_model(
        token_count=10_000_000, vocab_size=32, block_size=16, n_heads=4,
        target_tokens_per_param=40, min_tokens_per_param=10,
    )
    assert result.status == "comfortably_above_target"
    assert (result.d_model, result.n_layers) != (64, 2)


def test_checkpoint_captures_optimizer_step_and_sparsity_state():
    model = GamaX1Model(vocab_size=5, d_model=16, n_heads=2, n_layers=1, n_features=32,
                         max_seq_len=8, sparsity_k_init=16, sparsity_k_min=4)
    optimizer = torch.optim.AdamW(model.parameters())
    tok = CharTokenizer("abcde")
    model.sparsity_ctrl.k = 12
    checkpoint = checkpoint_dict(model, optimizer, tok, {"tokenizer": "char"}, step=7)
    assert checkpoint["step"] == 7
    assert checkpoint["sparsity_controller_state"]["k"] == 12
    assert "optimizer_state" in checkpoint and checkpoint["vocab"] == tok.chars


def test_word_tokenizer_warning_distinguishes_small_and_large_corpora():
    assert word_tokenizer_warning("word", 40, 20) is not None
    assert word_tokenizer_warning("char", 40, 20) is None
    assert word_tokenizer_warning("word", 2_400, 300) is None
