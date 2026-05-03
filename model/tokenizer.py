"""
Custom SentencePiece BPE tokenizer for Apollo v2.

v1 used tiktoken's GPT-2 BPE, which was trained on web text. That works but
has two problems for this corpus:

  1. It wastes vocab on patterns we don't have (URLs, JS keywords, emoji)
  2. It under-segments patterns we do have (literary contractions, numpy
     type names, code identifiers)

v2 trains a SentencePiece BPE directly on our 3-register corpus. The vocab
is allocated to the actual data distribution, so the same number of tokens
encodes more meaning per token, and training tokens stretch further.

The tokenizer file format is sentencepiece native (.model). We wrap it in
the same ApolloTokenizer interface as v1 so train.py and sample.py do not
care which tokenizer is in use.

Special tokens are added via SentencePiece user_defined_symbols and stay
intact through encode/decode. <|endoftext|> is also a user-defined symbol so
it tokenizes to a single id reliably.
"""

import json
import os
from pathlib import Path

import sentencepiece as spm


SPECIAL_TOKENS = (
    "<|literature|>",
    "<|wiki|>",
    "<|code|>",
    "<|endoftext|>",
)
EOT_TOKEN = "<|endoftext|>"

# Final vocab size. 32k is a sweet spot for 150MB of text - bigger than
# necessary risks unused entries; smaller wastes capacity on common bigrams.
DEFAULT_VOCAB_SIZE = 32000


class ApolloTokenizer:
    """Wraps a SentencePiece BPE model with Apollo's special-token convention."""

    def __init__(self, sp: spm.SentencePieceProcessor):
        self._sp = sp
        self._eot_id = sp.piece_to_id(EOT_TOKEN)

    @classmethod
    def train(
        cls,
        corpus_files: list[str | Path],
        out_path: str | Path,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ) -> "ApolloTokenizer":
        """Train a fresh SentencePiece BPE on the provided text files."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip the .model suffix - SentencePiece appends it itself.
        prefix = str(out_path).removesuffix(".model")

        spm.SentencePieceTrainer.train(
            input=",".join(str(p) for p in corpus_files),
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            user_defined_symbols=list(SPECIAL_TOKENS),
            byte_fallback=True,             # any byte still encodes
            normalization_rule_name="identity",  # do not lowercase / nfkc
            num_threads=os.cpu_count() or 4,
            train_extremely_large_corpus=False,
        )

        sp = spm.SentencePieceProcessor()
        sp.load(prefix + ".model")
        return cls(sp)

    @classmethod
    def load(cls, path: str | Path) -> "ApolloTokenizer":
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(path))
        return cls(sp)

    def save(self, path: str | Path) -> None:
        # Trained models are written by SentencePiece during train().
        # We also write a tiny json sidecar describing the layout for
        # backward compatibility with v1's loading pattern.
        path = Path(path)
        config = {
            "kind": "apollo-sentencepiece-bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
        }
        sidecar = path.parent / "tokenizer.json"
        with open(sidecar, "w") as f:
            json.dump(config, f, indent=2)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    @property
    def eot_id(self) -> int:
        return self._eot_id

    def special_id(self, token: str) -> int:
        return self._sp.piece_to_id(token)

    def encode(self, text: str, allow_specials: bool = True) -> list[int]:
        # SentencePiece's user_defined_symbols handle the special tags
        # automatically - they tokenize to single ids regardless of context.
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)


if __name__ == "__main__":
    # Smoke test: requires the v2 corpus files in data/raw/.
    import sys
    DATA_DIR = Path(__file__).parent.parent / "data"
    corpus = [
        DATA_DIR / "raw" / "literature.txt",
        DATA_DIR / "raw" / "wiki.txt",
        DATA_DIR / "raw" / "code.txt",
    ]
    missing = [str(p) for p in corpus if not p.exists()]
    if missing:
        print(f"missing corpus files (run prepare.py first): {missing}", file=sys.stderr)
        sys.exit(1)

    out = DATA_DIR / "tokenizer.model"
    print(f"training BPE vocab=4000 (toy run) on {[str(p) for p in corpus]}")
    tok = ApolloTokenizer.train(corpus, out, vocab_size=4000)
    tok.save(out)
    print(f"vocab_size: {tok.vocab_size}")
    print(f"eot_id: {tok.eot_id}")
    for s in SPECIAL_TOKENS:
        print(f"  {s} -> {tok.special_id(s)}")
    sample = "<|literature|>The night was dark and stormy.<|endoftext|>"
    ids = tok.encode(sample)
    print(f"encoded {len(ids)} ids: {ids[:20]}...")
    print(f"decoded: {tok.decode(ids)!r}")
