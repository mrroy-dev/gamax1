"""One-command GamaX1 demo with character or word-level tokenization."""

import argparse
import os

import torch

from gamax1.model import GamaX1Model
from gamax1.tokenizer import BPETokenizer, CharTokenizer, WordTokenizer, word_tokenizer_warning
from gamax1.train import get_batch, perplexity


def build_tokenizer(text: str, tokenizer_type: str, bpe_vocab_size: int = 1024):
    """Build the selected tokenizer and its non-blocking corpus-size warning."""
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(text, vocab_size=bpe_vocab_size, sample_chars=500_000)
        warning = None
    else:
        if tokenizer_type == "word":
            tokenizer = WordTokenizer(text, max_vocab_size=15_000)
        else:
            tokenizer = CharTokenizer(text)
        warning = word_tokenizer_warning(tokenizer_type, len(WordTokenizer._tokenize(text)), tokenizer.vocab_size)
    return tokenizer, warning


def parse_args():
    parser = argparse.ArgumentParser(description="Train and sample a small GamaX1 model.")
    parser.add_argument("--tokenizer", choices=("char", "word", "bpe"), default="char")
    parser.add_argument("--bpe_vocab_size", type=int, default=1024,
                        help="BPE vocabulary size for --tokenizer bpe (default: 1024).")
    parser.add_argument("--corpus", choices=("small", "large"), default="small",
                        help="Bundled corpus to use; 'small' trains on a 50K-char slice of the combined "
                             "Gutenberg corpus, 'large' uses it in full. For a faster word-tokenizer demo, "
                             "pass --tokenizer bpe instead.")
    parser.add_argument("--data", default=None,
                        help="Optional text file. Overrides --corpus (small or large bundled corpus).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    combined = os.path.join(os.path.dirname(__file__), "data", "sample_corpus_combined.txt")
    data_path = args.data or combined
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    if args.data is None and args.corpus == "small":
        text = text[:50_000]
    tok, warning = build_tokenizer(text, args.tokenizer, args.bpe_vocab_size)
    if warning:
        print(warning)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"Corpus: {data_path} | vocab size {tok.vocab_size} ({args.tokenizer})")

    d_model, n_layers, n_features, block_size = 64, 2, 256, 64
    model = GamaX1Model(vocab_size=tok.vocab_size, d_model=d_model, n_heads=2,
                         n_layers=n_layers, n_features=n_features, max_seq_len=block_size,
                         sparsity_k_init=n_features // 2, sparsity_k_min=n_features // 8).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dense_equivalent = n_features * n_layers

    print("Training for 300 steps on the selected corpus...")
    for step in range(1, 301):
        model.train()
        xb, yb = get_batch(train_data, block_size, 16, device)
        k = model.sparsity_ctrl.k
        _, loss = model(xb, targets=yb, k=k)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.sparsity_ctrl.step(loss.item())
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, block_size, 16, device)
                _, val_loss = model(xv, targets=yv, k=model.sparsity_ctrl.k)
            print(f"  step {step}: train_loss {loss.item():.3f}, train_ppl {perplexity(loss.item()):.2f}, "
                  f"val_loss {val_loss.item():.3f}, val_ppl {perplexity(val_loss.item()):.2f}, sparsity_k {k}")

    active = model.active_units_per_token()
    print(f"\nEfficiency check (Aetherion Sections 5.1/5.12):")
    print(f"  Dense-equivalent compute: {dense_equivalent} units/token")
    print(f"  GamaX1 sparse compute:    {active} units/token")
    print(f"  Compute ratio:            {dense_equivalent / active:.2f}x less compute\n")
    prompt = "The " if args.tokenizer == "char" else "The project"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=20)
    print("Sample generation (standard):\n ", tok.decode(out[0].tolist()).replace("\n", " "))
    out2 = model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=20, use_hierarchical_exit=True)
    print("\nSample generation (hierarchical Router/Validator early exit):\n ",
          tok.decode(out2[0].tolist()).replace("\n", " "))


if __name__ == "__main__":
    main()
