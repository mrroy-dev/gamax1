"""Tests for demo-only tokenizer selection without running a training job."""

from run_demo import build_tokenizer
from gamax1.tokenizer import CharTokenizer, WordTokenizer


def test_demo_selects_character_tokenizer_without_a_warning():
    tokenizer, warning = build_tokenizer("small text", "char")
    assert isinstance(tokenizer, CharTokenizer)
    assert warning is None


def test_demo_selects_word_tokenizer_and_warns_for_tiny_corpus():
    tokenizer, warning = build_tokenizer("small text", "word")
    assert isinstance(tokenizer, WordTokenizer)
    assert "[WARNING]" in warning


def test_demo_word_tokenizer_accepts_a_sufficiently_large_corpus():
    text = " ".join(f"term{i % 300}" for i in range(2_400))
    tokenizer, warning = build_tokenizer(text, "word")
    assert isinstance(tokenizer, WordTokenizer)
    assert tokenizer.vocab_size >= 200
    assert warning is None
