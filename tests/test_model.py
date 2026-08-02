"""
tests/test_model.py
=====================
End-to-end tests for GamaX1Model: shapes, a real training step that
reduces loss, generation, and the compute-accounting helper that makes
the sparse-vs-dense efficiency claim directly checkable in code.
"""

import torch
import pytest

from gamax1.model import GamaX1Model, apply_repetition_penalty
from gamax1.tokenizer import CharTokenizer, WordTokenizer


def make_tiny_model(vocab_size=20, hex_influence=False):
    return GamaX1Model(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=2,
        n_layers=2,
        n_features=64,
        max_seq_len=16,
        sparsity_k_init=32,
        sparsity_k_min=8,
        hex_influence=hex_influence,
    )


def test_forward_shapes():
    model = make_tiny_model()
    idx = torch.randint(0, 20, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, 20)
    assert loss is None


def test_language_model_output_weight_is_tied_to_input_embedding():
    model = make_tiny_model(vocab_size=16)
    assert model.token_emb.weight is model.head.weight


def test_forward_with_targets_returns_scalar_loss():
    model = make_tiny_model()
    idx = torch.randint(0, 20, (3, 10))
    targets = torch.randint(0, 20, (3, 10))
    logits, loss = model(idx, targets=targets)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_training_step_reduces_loss_on_a_repeated_pattern():
    """A model that cannot learn at all would be a critical bug --
    this checks genuine optimization happens, not just that the code
    runs without crashing."""
    torch.manual_seed(0)
    model = make_tiny_model(vocab_size=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    pattern = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
    x, y = pattern[:, :-1], pattern[:, 1:]

    losses = []
    for _ in range(150):
        _, loss = model(x, targets=y, k=model.sparsity_ctrl.k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, "loss should drop substantially on a trivial repeated pattern"


def test_generate_produces_correct_length_and_valid_ids():
    model = make_tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=15)
    assert out.shape == (1, 18)
    assert (out >= 0).all() and (out < 20).all()


def test_generate_with_hierarchical_exit_runs_and_produces_valid_output():
    model = make_tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10, use_hierarchical_exit=True)
    assert out.shape == (1, 13)
    assert (out >= 0).all() and (out < 20).all()


def test_active_units_per_token_reflects_current_k():
    model = make_tiny_model()
    model.sparsity_ctrl.k = 16
    assert model.active_units_per_token() == 16 * len(model.blocks)
    assert model.active_units_per_token(k=8) == 8 * len(model.blocks)


def test_sparse_model_uses_less_compute_than_dense_equivalent():
    """Direct, code-level check of the core Aetherion efficiency claim
    (Sections 5.1/5.12): active units per token should be substantially
    less than n_features * n_layers (the dense-equivalent baseline)."""
    model = make_tiny_model()
    dense_equivalent = 64 * len(model.blocks)
    assert model.active_units_per_token() < dense_equivalent


def test_hex_influence_variant_still_produces_finite_output():
    model = make_tiny_model(hex_influence=True)
    idx = torch.randint(0, 20, (2, 8))
    logits, _ = model(idx)
    assert torch.isfinite(logits).all()


def test_tokenizer_roundtrip():
    text = "hello world, this is GamaX1!"
    tok = CharTokenizer(text)
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
    assert tok.vocab_size == len(set(text))


def test_tokenizer_save_and_load(tmp_path):
    text = "abcdefg"
    tok = CharTokenizer(text)
    path = tmp_path / "tok.json"
    tok.save(str(path))
    tok2 = CharTokenizer.load(str(path))
    assert tok2.chars == tok.chars
    assert tok2.encode("abc") == tok.encode("abc")


def test_word_tokenizer_preserves_punctuation_and_uses_unknown_token(tmp_path):
    tok = WordTokenizer("Hello, world!")
    encoded = tok.encode("Hello, unseen!")
    assert encoded[2] == tok.stoi["<unk>"]
    assert tok.decode(tok.encode("Hello, world!")) == "Hello, world!"
    path = tmp_path / "word_tok.json"
    tok.save(path)
    assert WordTokenizer.load(path).tokens == tok.tokens


def test_word_tokenizer_caps_vocab_by_frequency_and_maps_dropped_words_to_unk():
    tokenizer = WordTokenizer("rare common common common medium medium punctuation !", max_vocab_size=3)
    assert tokenizer.tokens == ["<unk>", "common", "medium"]
    assert tokenizer.vocab_size == 3
    assert tokenizer.encode("common rare") == [tokenizer.stoi["common"], tokenizer.stoi["<unk>"]]
    assert tokenizer.decode(tokenizer.encode("rare")) == "<unk>"


def test_repetition_penalty_suppresses_present_positive_logits():
    logits = torch.tensor([[2.0, 1.0, -1.0, 3.0]])
    present = torch.tensor([True, False, True, False])
    penalized = apply_repetition_penalty(logits, present, 2.0)
    assert penalized[0, 0].item() == pytest.approx(1.0)   # 2.0 / 2
    assert penalized[0, 2].item() == pytest.approx(-2.0)  # -1.0 * 2
    assert penalized[0, 1].item() == pytest.approx(1.0)   # untouched
    assert penalized[0, 3].item() == pytest.approx(3.0)   # untouched


def test_repetition_penalty_is_identity_at_one():
    logits = torch.tensor([[2.0, -1.0]])
    present = torch.tensor([True, True])
    out = apply_repetition_penalty(logits, present, 1.0)
    assert torch.equal(out, logits)
    assert out is logits  # identity fast-path returns the input untouched


def test_generate_accepts_repetition_penalty():
    model = make_tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10, repetition_penalty=1.3)
    assert out.shape == (1, 13)
    assert (out >= 0).all() and (out < 20).all()


def test_dense_mode_matches_width_but_counts_all_features_as_active():
    sparse = make_tiny_model()
    dense = GamaX1Model(
        vocab_size=20, d_model=32, n_heads=2, n_layers=2, n_features=64,
        max_seq_len=16, sparsity_k_init=32, sparsity_k_min=8, dense_mode=True,
    )
    assert dense.active_units_per_token() == 64 * len(dense.blocks)
    assert sparse.active_units_per_token() < dense.active_units_per_token()
    assert sum(p.numel() for p in sparse.parameters()) == sum(p.numel() for p in dense.parameters())
