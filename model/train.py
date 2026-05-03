"""
Train Apollo on the BPE-tokenized 3-register corpus.

Quickstart
----------
    python prepare.py
    python train.py --size smoke --iters 500            # ~30-40 min smoke
    python train.py --size big --iters 8000 --hours 8   # overnight 50M run

Key config
----------
--size {smoke,mid,big}      Pick architecture preset from model.SIZES.
--iters N                   Total optimizer steps. Defaults match the size.
--hours H                   Optional wall-clock floor. The loop sleeps between
                            iters so total training time approaches H hours.
                            Useful overnight to keep the laptop cool.
--batch B                   Batch size in sequences. Defaults: 32 smoke, 24 big.
--block T                   Context length in tokens. Default 256.

Throttle
--------
With --hours set, after every iter the loop sleeps just long enough that the
fraction of wall-clock elapsed tracks the fraction of iters completed. So an
8000-iter run set to 8 hours gives ~3.6 sec per iter, with the GPU idling
between steps and the chassis cooling down. Self-correcting: an early hot iter
that runs slow shrinks subsequent sleeps automatically.

The throttle does not change gradient updates - same model, same training, just
slower wall-clock. If you remove --hours the loop runs flat-out.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from model import Apollo, ApolloConfig, SIZES
from tokenizer import ApolloTokenizer


# ---------- paths ----------

DATA_DIR = Path(__file__).parent.parent / "data"
CKPT_DIR = Path(__file__).parent.parent / "out"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- defaults per size ----------

SIZE_DEFAULTS = {
    "smoke": {"iters":  500, "batch": 32, "lr": 3e-4},
    "mid":   {"iters": 4000, "batch": 28, "lr": 3e-4},
    "big":   {"iters": 8000, "batch": 24, "lr": 2.5e-4},
}


# ---------- helpers ----------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(it: int, *, base_lr: float, min_lr: float, warmup: int, decay_to: int) -> float:
    if it < warmup:
        return base_lr * (it + 1) / warmup
    if it >= decay_to:
        return min_lr
    decay_ratio = (it - warmup) / (decay_to - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def get_batch(split: str, train_data: np.ndarray, val_data: np.ndarray,
              device: torch.device, batch_size: int, block_size: int):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, batch_size, block_size,
                  eval_iters: int) -> dict:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, device, batch_size, block_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES.keys()), default="smoke")
    ap.add_argument("--iters", type=int, default=None,
                    help="Total optimizer steps (default depends on --size)")
    ap.add_argument("--hours", type=float, default=None,
                    help="Wall-clock floor in hours. Throttles via sleeping.")
    ap.add_argument("--batch", type=int, default=None,
                    help="Batch size in sequences")
    ap.add_argument("--block", type=int, default=256,
                    help="Context length in tokens")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=25)
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=8,
                    help="Stop training if val loss has not improved for this many "
                         "consecutive evals. Set to 0 to disable.")
    ap.add_argument("--out", default=None,
                    help="Override checkpoint name (default best_{size}.pt)")
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from")
    args = ap.parse_args()

    defaults = SIZE_DEFAULTS[args.size]
    iters = args.iters or defaults["iters"]
    batch_size = args.batch or defaults["batch"]
    base_lr = args.lr or defaults["lr"]
    block_size = args.block

    out_name = args.out or f"best_{args.size}.pt"
    ckpt_path = CKPT_DIR / out_name

    device = get_device()
    print(f"== Apollo train ==")
    print(f"  device: {device}")
    print(f"  size: {args.size} | iters: {iters} | batch: {batch_size} | block: {block_size}")
    print(f"  lr: {base_lr:.2e} | dropout: {args.dropout}")
    if args.hours is not None:
        print(f"  target hours: {args.hours:.2f} (throttled)")
    print(f"  out: {ckpt_path}")

    # Load tokenizer + data.
    tok = ApolloTokenizer.load(DATA_DIR / "tokenizer.model")
    train_data = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint32)
    val_data = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint32)
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    # Build model.
    arch = SIZES[args.size]
    cfg = ApolloConfig(
        block_size=block_size,
        vocab_size=tok.vocab_size,
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        n_embd=arch["n_embd"],
        dropout=args.dropout,
        bias=False,
    )
    model = Apollo(cfg).to(device)
    print(f"  params: {model.num_params() / 1e6:.2f}M total | "
          f"{model.body_params() / 1e6:.2f}M body (no embed)")

    # Optimizer.
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=base_lr,
        betas=(0.9, 0.95),
    )

    # Resume?
    start_iter = 0
    best_val = float("inf")
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_iter = int(ckpt.get("iter", 0))
        best_val = float(ckpt.get("val_loss", float("inf")))
        print(f"  resumed from {args.resume} at iter {start_iter}, val={best_val:.4f}")

    # Schedule params.
    warmup = max(100, iters // 25)
    decay_to = iters
    min_lr = base_lr / 10.0

    target_seconds = args.hours * 3600 if args.hours is not None else None
    start_time = time.time()
    evals_since_improvement = 0
    stopped_early = False

    # Training loop.
    for it in range(start_iter, iters + 1):
        # LR.
        lr = get_lr(it, base_lr=base_lr, min_lr=min_lr, warmup=warmup, decay_to=decay_to)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # Eval.
        if it % args.eval_interval == 0 or it == iters:
            losses = estimate_loss(model, train_data, val_data, device,
                                   batch_size, block_size, args.eval_iters)
            elapsed = time.time() - start_time
            improved = losses["val"] < best_val
            marker = " *" if improved else ""
            print(
                f"iter {it:5d} | lr {lr:.2e} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"elapsed {fmt_secs(elapsed)}{marker}"
            )
            if improved:
                best_val = losses["val"]
                evals_since_improvement = 0
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "iter": it,
                    "val_loss": best_val,
                    "size": args.size,
                }, ckpt_path)
            else:
                evals_since_improvement += 1
                if args.patience > 0 and evals_since_improvement >= args.patience:
                    print(
                        f"\nearly stop: val loss has not improved for "
                        f"{args.patience} consecutive evals (best val={best_val:.4f}). "
                        f"halting at iter {it}."
                    )
                    stopped_early = True
                    break

        if it == iters:
            break

        # One training step.
        X, Y = get_batch("train", train_data, val_data, device, batch_size, block_size)
        _, loss = model(X, Y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if it % args.log_interval == 0 and it > 0:
            print(f"  step {it:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

        if it > 0 and it % args.save_interval == 0:
            snap = CKPT_DIR / f"{args.size}_iter_{it}.pt"
            torch.save({
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "iter": it,
                "size": args.size,
            }, snap)

        # ---- throttle: sleep to track target wall-clock ----
        if target_seconds is not None and iters > 0:
            elapsed = time.time() - start_time
            target_at_this_iter = (it + 1) / iters * target_seconds
            slack = target_at_this_iter - elapsed
            if slack > 0:
                # Sleep in small chunks so Ctrl-C is responsive.
                end_at = time.time() + slack
                while time.time() < end_at:
                    time.sleep(min(0.5, end_at - time.time()))

    total = time.time() - start_time
    if stopped_early:
        print(f"\nstopped early in {fmt_secs(total)} | best val: {best_val:.4f}")
    else:
        print(f"\ndone in {fmt_secs(total)} | best val: {best_val:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
