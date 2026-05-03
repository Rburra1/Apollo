"""
Apollo - GPT-style transformer, written from scratch.

Architecture is the same as Klotho - this is on purpose. The whole point of
this project is to scale data and tokenization while holding the architecture
constant, so we can attribute output quality changes to the corpus and the
tokenizer rather than to architecture changes.

  - Decoder-only transformer (causal language model)
  - Pre-LayerNorm blocks: LN -> attention -> residual, LN -> FFN -> residual
  - Multi-head causal self-attention with single fused QKV projection
  - 4x expansion feed-forward with GELU
  - Token + learned positional embeddings, summed at input
  - Weight tying: input embedding shares weights with output projection
  - GPT-2 init: N(0, 0.02), residual projections scaled by 1/sqrt(2L)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ApolloConfig:
    block_size: int = 256        # max context length in tokens
    vocab_size: int = 50260      # set to actual tokenizer vocab at runtime
    n_layer: int = 6             # number of transformer blocks
    n_head: int = 6              # number of attention heads
    n_embd: int = 384            # embedding / hidden dimension
    dropout: float = 0.1         # dropout probability
    bias: bool = False           # whether linear layers use bias terms


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection."""

    def __init__(self, cfg: ApolloConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask: lower-triangular ones, registered as a buffer.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward with 4x expansion and GELU activation."""

    def __init__(self, cfg: ApolloConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block: attention + FFN with pre-norm and residuals."""

    def __init__(self, cfg: ApolloConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Apollo(nn.Module):
    """GPT-style language model. Same architecture as Klotho, ready to scale."""

    def __init__(self, cfg: ApolloConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_final = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share embedding matrix with output projection.
        self.head.weight = self.token_emb.weight

        # GPT-2 weight init.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias) if module.bias is not None else None
            nn.init.ones_(module.weight)

    def num_params(self) -> int:
        """Total trainable parameters. parameters() already dedupes weight-tied
        tensors, so the embedding is counted exactly once."""
        return sum(p.numel() for p in self.parameters())

    def body_params(self) -> int:
        """Transformer body params excluding the token embedding. With a 50K
        BPE vocab the embedding dominates total count, so the body number is
        what to compare across configs to reason about model capacity."""
        return self.num_params() - self.token_emb.weight.numel()

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) long tensor of token ids. Returns (logits, loss)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} > block_size {self.cfg.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_e = self.token_emb(idx)
        pos_e = self.pos_emb(pos)
        x = self.drop(tok_e + pos_e)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation. idx: (B, T) seed token ids."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# Named size presets. CLI in train.py picks one of these.
SIZES = {
    "smoke": {"n_layer": 6,  "n_head": 6, "n_embd": 384},   # ~30M with 50K BPE vocab
    "mid":   {"n_layer": 6,  "n_head": 8, "n_embd": 448},   # ~37M
    "big":   {"n_layer": 8,  "n_head": 8, "n_embd": 512},   # ~51M
}
