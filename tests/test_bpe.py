"""Tests for the dependency-free byte-level BPE tokenizer."""

import pytest

from gamax1.tokenizer import BPETokenizer


SAMPLE = (
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness. \"Call me Ishmael,\" he said. "
    "The time machine.\n"
)


def test_bpe_roundtrip_exact():
    tok = BPETokenizer(SAMPLE, vocab_size=512)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_bpe_vocab_size_is_256_plus_merges():
    tok = BPETokenizer(SAMPLE, vocab_size=300)
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256


def test_bpe_never_emits_unknown_on_unseen_text():
    tok = BPETokenizer(SAMPLE, vocab_size=512)
    ids = tok.encode("Zephyroth quasars underflowed qvx213!")
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert "<unk>" not in tok.decode(ids)


def test_bpe_merges_frequent_substrings():
    tok = BPETokenizer("the the the the the the the the", vocab_size=300)
    # The most frequent adjacent byte pair of the text must be merged first.
    assert tok.merges[0] in {(ord('t'), ord('h')), (ord('h'), ord('e')), (ord(' '), ord('t'))}
    # A merged multi-byte token must exist in the vocabulary.
    assert any(len(tok.token_bytes[i]) > 1 for i in range(256, tok.vocab_size))


def test_bpe_save_and_load_roundtrip(tmp_path):
    tok = BPETokenizer(SAMPLE, vocab_size=512)
    path = tmp_path / "bpe.json"
    tok.save(str(path))
    tok2 = BPETokenizer.load(str(path))
    assert tok2.merges == tok.merges
    assert tok2.encode(SAMPLE) == tok.encode(SAMPLE)


def test_bpe_preserves_whitespace_and_punctuation():
    tok = BPETokenizer(SAMPLE, vocab_size=512)
    tricky = "a  b   c\t\td\ne . ,  \"quoted\" (paren) ;!"
    assert tok.decode(tok.encode(tricky)) == tricky


def test_bpe_requires_text_or_merges():
    with pytest.raises(ValueError):
        BPETokenizer()
    with pytest.raises(ValueError):
        BPETokenizer(SAMPLE, vocab_size=100)
