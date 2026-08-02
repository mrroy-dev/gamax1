# GamaX1 — First Working Version of Aetherion

GamaX1 is the first real, trainable NLP implementation of the **Aetherion**
architecture (see the accompanying Aetherion Technical Report v4). It is
built in PyTorch, runs on Google Colab (CPU or GPU), and translates each
*validated* mechanism from the research report into working code — not
the architecture's original theoretical form, but the version that was
actually tested, debugged, and shown to work.

## What's in here

```
gamax1/
├── gamax1/
│   ├── layers.py       # Sparse superposition, dynamic sparsity, PTM, hex influence, Router/Validator
│   ├── model.py         # GamaX1Block + GamaX1Model (full causal LM)
│   ├── tokenizer.py      # Character, word, and byte-level BPE tokenizers
│   ├── train.py         # Training script (CLI, metrics, scheduling, resume)
│   ├── compare_dense.py # Matched sparse-versus-dense experiment
│   └── generate.py      # Text generation script (CLI)
├── tests/               # 50+ unit + integration tests (pytest)
├── data/sample_corpus_combined.txt  # Combined public-domain Gutenberg corpus (72 MB)
├── run_demo.py          # One-command end-to-end demo
├── compare_dense.py     # One-command sparse-versus-dense comparison
├── GamaX1_Colab.ipynb   # Google Colab notebook
├── requirements.txt
└── setup.py
```

## How this maps to the research report

| Research finding | Code |
|---|---|
| Sparse superposition matches ~96–98% of dense accuracy at a fraction of the compute (Sections 5.1, 5.12) | `layers.SparseSuperpositionLinear`, used as the FFN in every `GamaX1Block` |
| Dynamic sparsity must be gated on a loss **trend**, never instantaneous signal (Section 5.2) | `layers.DynamicSparsityController` |
| Dead-feature prevention needs an explicit, asymmetric entry/exit rule that ignores true negatives (Section 5.3) | `layers.ProbationaryMemoryTracker` |
| A trained Router beats a hand-tuned heuristic (Sections 5.4, 5.11) | `layers.RouterExpert` (trained, not hand-tuned) |
| Early-exit gating must use answer **stability**, not raw confidence (Section 5.6) | `layers.ValidatorExpert.is_stable` |
| Hexagonal neighbor influence only helps on data with matching cluster structure (Section 5.7) | `layers.HexNeighborInfluence` — included, **off by default**, documented condition |
| Plain cross-entropy is the wrong loss for imbalanced multi-label top-K tasks (Section 6.2) | **Not applicable here** — GamaX1 does standard next-token language modeling, a well-posed single-correct-class task, so standard cross-entropy is used deliberately (see `model.py` docstring) |

### Honest design choices (please read before assuming more than is claimed)

- **Attention is retained.** Aetherion's validated claim is about replacing a dense *feed-forward* layer with sparse superposition — it was never tested as an attention replacement. GamaX1 uses standard causal self-attention for token-mixing and Aetherion's sparse mechanism for the FFN. This is the most defensible way to bring validated results into a real model without overclaiming.
- **Hexagonal influence defaults to OFF.** The research report found this mechanism can *hurt* when the underlying data lacks the right clustering structure, and there's no way to guarantee a language model's hidden features cluster at the right scale. Turn it on and evaluate empirically on your own data; don't assume a benefit.
- **Hierarchical Router/Validator early exit is inference-time only.** Training with dynamic per-sample depth would need ragged-batch handling not implemented in this first version — a real limitation, stated plainly rather than hidden.
- **This is GamaX1 — a first version.** The bundled corpus is ~3KB of original demo text, meant to prove the pipeline works end-to-end in seconds on Colab CPU, not to train a high-quality model. Supply your own corpus for real results.

## Quickstart (local)

```bash
pip install -r requirements.txt
python run_demo.py

# Recommended bundled demo (byte-level BPE subword tokenizer)
python run_demo.py --tokenizer bpe --corpus large
```

## Quickstart (Google Colab)

Open `GamaX1_Colab.ipynb` in Colab (Runtime → Change runtime type → GPU
recommended but not required), and run all cells.

## CLI usage

```bash
# Train on the bundled corpus
python -m gamax1.train --max_steps 2000

# Train on your own text file
python -m gamax1.train --data path/to/your.txt --max_steps 5000 --n_layers 6

# Use a word-level vocabulary, linear warmup, and cosine LR decay
python -m gamax1.train --tokenizer word --max_steps 5000 --warmup_steps 250

# Use the built-in dependency-free byte-level BPE subword tokenizer
# (recommended for real corpora: no <unk>, ~1.4x more tokens than word level)
python -m gamax1.train --tokenizer bpe --data path/to/your.txt --max_steps 5000 --bpe_vocab_size 8000

# Cap a large word vocabulary (15,000 entries including <unk>) to keep
# vocabulary parameters from dominating the language model
python -m gamax1.train --tokenizer word --data path/to/novels.txt --max_vocab_size 15000

# Stabilize validation reporting by averaging ten 128-sequence validation batches
python -m gamax1.train --eval_batches 10 --eval_batch_size 128

# Use the stricter final-corpus-tail holdout instead of the representative
# random-window validation split used by default
python -m gamax1.train --validation_split tail

# Let the safety net choose a viable architecture for this corpus
python -m gamax1.train --tokenizer bpe --data data/sample_corpus_combined.txt --auto_size_model

# Word-level demo on the bundled combined corpus
python run_demo.py --tokenizer word --corpus large

# Warn after three worsening validation evaluations, or stop at that point
python -m gamax1.train --overfit_patience 3 --early_stop_on_overfit

# Save recovery checkpoints regularly and continue one after an interruption
python -m gamax1.train --checkpoint_interval 500
python -m gamax1.train --resume_from checkpoints/gamax1_step_500.pt --max_steps 2000

# Generate text from a checkpoint
python -m gamax1.generate --ckpt checkpoints/gamax1.pt --prompt "Once upon a time" --max_new_tokens 300

# Generate using the Router/Validator hierarchical early exit
python -m gamax1.generate --ckpt checkpoints/gamax1.pt --hierarchical_exit

# Suppress repetition: penalize tokens already present in the sequence (>1.0)
python -m gamax1.generate --ckpt checkpoints/gamax1.pt --repetition_penalty 1.2
```

`train` logs train/validation loss, perplexity, current learning rate, and
sparse active-unit compute at each evaluation interval. Checkpoints retain the
optimizer, current training step, and dynamic-sparsity controller state, so a
resumed run follows the same LR schedule rather than starting over. The
checkpoint records the tokenizer type; `generate` restores it automatically.

Word tokenization needs repeated word contexts. A small corpus creates a thin
vocabulary and too few examples per word, which leads to weak next-word output.
Both `train` and `run_demo.py` print a warning for that case. Use
`--tokenizer char` for a small corpus, `--tokenizer bpe` (or a larger `--data`
file) for serious word-level generation.

For real corpora, prefer the built-in byte-level BPE tokenizer
(`--tokenizer bpe`). It needs no external dependency, never emits `<unk>`
(any byte sequence is representable), and yields roughly 1.4x more training
tokens per byte than word tokenization, which directly improves next-token
modeling quality. BPE pair statistics are learned from the first
`--bpe_sample_chars` characters (default 3M, a deliberate speed/quality
trade-off) and are saved in checkpoints so `generate` restores them exactly.

## Avoiding memorization

GamaX1 prints a `Corpus/model size check` before training. It compares the
number of tokens produced by the active tokenizer with the model's trainable
parameter count. As a rough guardrail, aim for at least **10 tokens per
parameter** (`--min_tokens_per_param 10`). This is a warning heuristic, not a
guarantee of generalization: held-out data and generated samples still matter.

Vocabulary size is a third part of this calculation alongside corpus tokens
and model architecture. GamaX1 ties the standard input-embedding and output
projection weights, so it does not pay for two separate vocabulary matrices.
For large word corpora, `--max_vocab_size` defaults to 15,000 entries
(including `<unk>`), retaining the most frequent tokens and mapping the rest to
`<unk>`. Capping vocabulary is often a more useful correction than shrinking a
model below viability or merely adding data when vocabulary matrices dominate
the parameter count. Set `--max_vocab_size` explicitly when comparing runs;
the setting is saved in checkpoints.

At every evaluation, GamaX1 also checks for a distinct memorization pattern.
If the corpus/model ratio is below the threshold and either training or
validation perplexity falls below `1.5` (configurable with
`--perplexity_memorization_floor`), it prints a one-time
`MEMORIZATION DETECTED` warning. This catches the case where both losses fall
together, which the classic "validation loss rises while train loss falls"
warning cannot see. `--early_stop_on_overfit` stops for either warning.

Validation loss is the mean of `--eval_batches` independent batches (default
`10`), each containing `--eval_batch_size` sequences (default `128`). These
controls are separate from training `--batch_size`, so increasing evaluation
coverage reduces noisy validation-perplexity swings without changing optimizer
dynamics. Larger values produce a steadier estimate at the cost of additional
forward-pass time per evaluation.

By default, `--validation_split random_windows` assigns non-overlapping
block-sized windows from throughout the corpus to train or validation using a
reproducible seed. This avoids validating only on the final book in a combined
corpus. Use `--validation_split tail` when a strict final-section or
out-of-domain holdout is specifically desired; expect its perplexity to be
higher and do not compare it directly with random-window validation.

`--auto_size_model` never makes a degenerate model just to meet a ratio. Its
hard minimum is `d_model=64`, `n_layers=2`, at least two attention heads, and
`n_features=4 * d_model`; it also adjusts `n_heads` upward to two if needed.
Above that floor it selects the largest tier that targets 40 tokens per
parameter by default (`--auto_size_target_tokens_per_param 40`). This target is
intended to land in a practical 20–100 tokens-per-parameter range, not merely
barely clear the minimum threshold.

The bundled `sample_corpus_combined.txt` (~72 MB) tokenizes to roughly 15M
word tokens or 20M+ BPE tokens. That comfortably clears the 10-tokens-per-
parameter guardrail for small architectures, but auto-size will still select
the viable `d_model=64`, two-layer, 256-feature floor unless you explicitly
raise capacity — the floor is a safety net, not a production configuration.
Use held-out text to choose a serious configuration, and prefer BPE
(`--tokenizer bpe`) over word tokenization for this corpus.

Visible symptoms of failure are perplexity collapsing toward the theoretical
minimum of 1.0 (especially below about 1.5) and generated text reproducing the
training corpus verbatim, often looping back to its start. Reduce capacity,
add regularization (`--dropout`, `--weight_decay`), and most importantly train
on more data rather than treating such output as successful generation.

## Sparse versus dense comparison

Run a short matched experiment with the bundled corpus:

```bash
python compare_dense.py
# Or tune the small experiment for your corpus/hardware
python -m gamax1.compare_dense --data path/to/your.txt --tokenizer word --steps 500 --d_model 128 --n_features 512
```

The output table reports final train/validation loss, validation perplexity,
parameter count, and active compute units per token for both models. The dense
control has matching projection shapes and parameter count; it differs only by
evaluating all FFN features rather than the sparse model's top-K active ones.
The final line summarizes the validation-perplexity/compute tradeoff.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

The tests check the *specific properties* the research report found
to matter (exact top-K counts, trend-based not instantaneous sparsity
gating, asymmetric PTM hysteresis, answer-stability-based early exit),
not just "does it run without crashing."

## License

Research/educational use. No warranty. This is a first-version ("v1")
prototype accompanying an ongoing research report — expect rough edges,
and see the report's Limitations and Open Questions sections for what
has and hasn't been validated at what scale.
