"""
gamax1/generate.py
====================
Load a trained GamaX1 checkpoint and generate text.
"""

import argparse
import os

import torch

from .model import GamaX1Model
from .tokenizer import BPETokenizer, CharTokenizer, WordTokenizer


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("tokenizer") == "bpe":
        tok = BPETokenizer(merges=ckpt["merges"])
    else:
        tok_cls = WordTokenizer if cfg.get("tokenizer") == "word" else CharTokenizer
        tok = tok_cls(vocab=ckpt["vocab"])
    model = GamaX1Model(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_features=cfg["n_features"],
        max_seq_len=cfg["block_size"],
        hex_influence=cfg.get("hex_influence", False),
        sparsity_k_init=cfg["n_features"] // 2,
        sparsity_k_min=cfg["n_features"] // 8,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GamaX1 model.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gamax1.pt")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Penalize reusing tokens already in the sequence. >1.0 suppresses "
                             "repetition (e.g. 1.2); 1.0 disables it (default: 1.0).")
    parser.add_argument("--hierarchical_exit", action="store_true",
                         help="Use Router/Validator hierarchical early exit at inference time.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.ckpt, device)

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        use_hierarchical_exit=args.hierarchical_exit,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
