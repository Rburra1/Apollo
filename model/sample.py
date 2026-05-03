"""
Sample text from a trained Apollo checkpoint.

Examples
--------
    # Plain prompt, defaults
    python sample.py "ROMEO:"

    # Tag-prompt to bias the register
    python sample.py "<|literature|>The night was"
    python sample.py "<|wiki|>The Roman Empire"
    python sample.py "<|code|>def fibonacci"

    # Stream tokens as they generate
    python sample.py "<|wiki|>Newton" --stream

    # Pick a checkpoint other than best_smoke.pt
    python sample.py "<|code|>class" --ckpt out/best_big.pt

Notes on tag prompting: the special tokens <|literature|>, <|wiki|>, <|code|>
were inserted at the start of every document during training, so prompting
with one of them tells the model "produce text in this register."
"""

import argparse
import sys
from pathlib import Path

import torch

from model import Apollo, ApolloConfig
from tokenizer import ApolloTokenizer


CKPT_DIR = Path(__file__).parent.parent / "out"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="<|literature|>")
    ap.add_argument("--ckpt", default=None,
                    help="Path to checkpoint. Defaults to out/best_smoke.pt then best_big.pt.")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Locate checkpoint.
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        for cand in ("best_big.pt", "best_mid.pt", "best_smoke.pt"):
            if (CKPT_DIR / cand).exists():
                ckpt_path = CKPT_DIR / cand
                break
        else:
            print("ERROR: no checkpoint found in out/. Train first or pass --ckpt.", file=sys.stderr)
            sys.exit(1)
    print(f"# loading {ckpt_path}", file=sys.stderr)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ApolloConfig(**ckpt["config"])
    model = Apollo(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = ApolloTokenizer.load(DATA_DIR / "tokenizer.model")
    if cfg.vocab_size != tok.vocab_size:
        print(f"WARN: ckpt vocab_size {cfg.vocab_size} != tokenizer vocab_size {tok.vocab_size}",
              file=sys.stderr)

    print(f"# size: {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}D | params: "
          f"{model.num_params() / 1e6:.2f}M | val_loss: {ckpt.get('val_loss', '?')}",
          file=sys.stderr)

    # Encode prompt with specials enabled so <|tag|> markers tokenize as one id.
    encoded = tok.encode(args.prompt) or [tok.eot_id]
    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    # Print the prompt itself first so the user sees the seed.
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    if args.stream:
        # Token-by-token to mimic the API streaming UX.
        with torch.no_grad():
            for _ in range(args.max_tokens):
                idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / max(args.temperature, 1e-8)
                if args.top_k:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
                piece = tok.decode([int(next_id.item())])
                sys.stdout.write(piece)
                sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        new_ids = out[0].tolist()[len(encoded):]
        sys.stdout.write(tok.decode(new_ids))
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
