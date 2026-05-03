# Apollo

A from-scratch, decoder-only GPT-style transformer trained on a multi-register English corpus. Sibling to [Klotho](https://github.com/Rburra1/Klotho), with the same architecture but bigger data, BPE tokenization, and three explicit registers (`literature`, `wiki`, `code`) tagged at training time.

> Named after Apollo, Greek god of poetry, music, prophecy, and knowledge.

The architecture is held constant across versions on purpose. The point of this project is to attribute output quality changes to the corpus and tokenizer, not to a new model.

## Versions

| | v1 | v2 (current) |
|--|--|--|
| Tokenizer | tiktoken GPT-2 BPE (vocab=50260) | Custom SentencePiece BPE trained on the actual corpus (vocab=32000) |
| Code corpus | numpy/pandas/django/scipy (test-fixture-heavy) | flask/requests/fastapi/click/rich/pydantic/etc, test dirs excluded (38 production-grade libs) |
| Literature filter | none | English-only stopword-frequency filter |
| Early stopping | none | patience=8 evals on val plateau |
| Body params | 25.31M | 25.31M |
| Total params | 51.05M | 41.70M |
| Best val loss | 3.67 (random baseline 10.82) | 3.95 (random baseline 10.37) |

See [SAMPLES.md](SAMPLES.md) for full sample outputs and a detailed comparison of failure modes between versions.

## Three-register data

Three corpora at ~50 MB each, each document tagged with a register marker:

```
<|literature|>{Project Gutenberg book text}<|endoftext|>
<|wiki|>{Wikipedia article text}<|endoftext|>
<|code|>{Python source code}<|endoftext|>
```

At sample time you can prompt with a register marker to bias the output:

```bash
python sample.py "<|literature|>The night was"
python sample.py "<|wiki|>The Roman Empire"
python sample.py "<|code|>def fibonacci"
```

## Quick start

```bash
cd model
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt

# Download corpora, train custom BPE, write train/val .bin files. ~30 min.
python prepare.py

# Smoke test: ~23M params, 100 iters, ~7 min on M4 MPS.
python train.py --size smoke --iters 100 --eval-interval 50

# Big run: ~42M params, 8000 iters, throttled to 8 hours overnight.
python train.py --size big --iters 8000 --hours 8

# Sample with register tags.
python sample.py "<|wiki|>Newton" --stream
```

## Architecture sizes

Three preset sizes in `model.SIZES`. The BPE vocab matrix dominates parameter count at this scale.

| Size  | Layers | Heads | Embd | Params (V=32000) |
|-------|--------|-------|------|------------------|
| smoke | 6      | 6     | 384  | ~23 M            |
| mid   | 6      | 8     | 448  | ~29 M            |
| big   | 8      | 8     | 512  | ~42 M            |

## Overnight throttling

Training overnight on a laptop, the worry is thermal stress not throughput. The `--hours H` flag adds a sleep between iterations so total wall-clock approaches H hours. Same gradient updates, same final model, just spread out.

```bash
python train.py --size big --iters 8000 --hours 8
```

The throttle is self-correcting: if early iters run hot and slow down, later iters sleep less. Remove `--hours` for flat-out training.

Recommended overnight setup:
- Power cable plugged in
- Hard flat surface, lid open
- Close other apps
- Wrap the launch in `caffeinate` so macOS does not sleep:

```bash
caffeinate -dimsu python train.py --size big --iters 8000 --hours 8
```

## Early stopping

Training halts if val loss has not improved for `--patience` consecutive evals (default 8). The best checkpoint by val loss is always saved at `out/best_{size}.pt`. Set `--patience 0` to disable.

## Pretrained checkpoints

Trained model weights are too large for direct git storage and are hosted as GitHub release assets:

- [v2.0 release](https://github.com/Rburra1/Apollo/releases/tag/v2.0) - `best_big_v2.pt` (161 MB, val loss 3.95, vocab=32000)
- [v1.0 release](https://github.com/Rburra1/Apollo/releases/tag/v1.0) - `best_big_v1.pt` (197 MB, val loss 3.67, vocab=50260)

After downloading, place under `out/` and run sample.py:

```bash
mkdir -p out
curl -L -o out/best_big.pt https://github.com/Rburra1/Apollo/releases/download/v2.0/best_big_v2.pt
python model/sample.py "<|wiki|>Newton" --stream
```

To sample with v1, download `best_big_v1.pt` and pass `--ckpt out/best_big_v1.pt`. Note that v1 used the GPT-2 tiktoken tokenizer rather than the v2 SentencePiece tokenizer, so loading v1 weights requires checking out the v1.0 git tag for the matching `tokenizer.py`.

## File map

```
model/
  model.py         Apollo architecture (~200 lines, identical to Klotho)
  tokenizer.py     SentencePiece BPE wrapper with 3 register specials
  prepare.py       Downloads Gutenberg + Wikipedia + Python code, trains BPE, writes .bin
  train.py         Training loop with size presets, --hours throttle, early stopping
  sample.py        Generate text from a checkpoint, supports tag prompts and streaming
  requirements.txt
data/              [gitignored] tokenizer.model, meta.json, train.bin, val.bin, raw/
out/               trained checkpoints (best_big.pt committed past gitignore)
```

## Roadmap

v3 directions worth exploring (see SAMPLES.md "Lessons for v3" for full reasoning):

- Single-variable ablations to attribute v2's improvements to specific levers
- Rebalance Wikipedia sample to oversample non-biography articles
- Push training tokens past 100M (current binding constraint at 36M)
- Try longer block_size (current 256) on the same corpus
