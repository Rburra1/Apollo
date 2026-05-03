"""
Apollo v2 data prep.

Changes from v1
---------------
1. Filter non-English from Project Gutenberg literature corpus.
   v1's wiki register collapsed to ungrammatical French on OOD prompts because
   French translations (Hugo, Dumas) had been pulled into the literature
   corpus and entangled with wiki tokens. v2 detects French/non-English by
   checking common stopword frequency before accepting a book.

2. Swap the code corpus.
   v1 was numpy/pandas/django/scipy heavy, which means thousands of test
   files. The model learned 'code looks like this' (asserts and GH issue
   refs) but not function semantics. v2 swaps in production source from
   smaller, more idiomatic libraries: flask, requests, fastapi, httpx,
   click, rich, pydantic, jinja, etc - and explicitly skips test directories.

3. Train a custom SentencePiece BPE on the actual corpus.
   v1 used GPT-2's BPE, trained on web data. v2 trains its own BPE on
   the 150MB three-register mix so vocab is allocated where the data is.

Sources
-------
1. Project Gutenberg literature  ->  <|literature|>{text}<|endoftext|>   (English-only)
2. English Wikipedia (random)    ->  <|wiki|>{text}<|endoftext|>
3. Python production source      ->  <|code|>{text}<|endoftext|>           (no test dirs)

Resumability: each register is cached at `data/raw/{register}.txt`. If the
cache file exists at >=90% of target size, it is reused. Delete a cache file
to force a redownload.

Run:
    python prepare.py
"""

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from tqdm import tqdm

from tokenizer import ApolloTokenizer, SPECIAL_TOKENS, DEFAULT_VOCAB_SIZE


# ---------- config ----------

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BYTES_PER_REGISTER = 50 * 1024 * 1024  # 50 MB
VAL_FRACTION = 0.05
SEED = 1337

WIKI_DELAY = 0.05
GUTEN_DELAY = 0.10


# ---------- 1. Project Gutenberg, English-only ----------

# Curated list of public-domain English books. Removed the v1 entries that
# were French translations or French originals (Dumas Three Musketeers /
# Monte Cristo / Kreutzer Sonata, Hugo Les Mis, Tolstoy Anna Karenina if
# pulled as French translation, etc). Replaced with more English-language
# novels, essays, and plays.
GUTENBERG_IDS = [
    1342, 11, 84, 98, 1661, 174, 2701, 1080, 219, 16328,    # P&P, Alice, Frankenstein, Tale of Two Cities, Holmes, Dorian Gray, Moby Dick, Modest Proposal, Heart of Darkness, Beowulf
    345, 4300, 2542, 768, 158, 1184, 76, 2814,              # Dracula, Ulysses, Doll's House, Wuthering Heights, Emma, Count of Monte Cristo (English), Huck Finn, Dubliners
    1260, 5200, 36, 100, 408,                               # Jane Eyre, Metamorphosis, War of Worlds, Shakespeare Complete, Souls of Black Folk
    74, 120, 35, 244, 829, 209, 1064, 1322,                 # Tom Sawyer, Treasure Island, Time Machine, Study in Scarlet, Gulliver, Turn of Screw, Aspects of Novel, Leaves of Grass
    1400, 766, 580, 1023, 5097,                             # Great Expectations, David Copperfield, Pickwick Papers, Bleak House
    600, 2680, 730,                                         # Notes from Underground, Meditations, Oliver Twist
    23, 1228, 4217, 1933, 244,                              # Cranford, Common Sense, Portrait of Artist, Beyond Good and Evil
    2814, 1497,                                             # Dubliners, The Republic
    # Additional English literature to backfill capacity:
    158,    # Emma
    105,    # Persuasion (Austen)
    161,    # Sense and Sensibility (Austen)
    141,    # Mansfield Park (Austen)
    113,    # Northanger Abbey (Austen)
    768,    # Wuthering Heights
    1260,   # Jane Eyre
    969,    # Agnes Grey
    9176,   # Tenant of Wildfell Hall
    1400,   # Great Expectations
    580,    # Pickwick Papers
    98,     # A Tale of Two Cities
    1023,   # Bleak House
    883,    # Mystery of Edwin Drood
    967,    # Hard Times
    963,    # Little Dorrit
    700,    # Vanity Fair (Thackeray)
    2641,   # A Room With a View (Forster)
    2610,   # Howards End (Forster)
    2891,   # Where Angels Fear to Tread (Forster)
    160,    # The Awakening (Chopin)
    74,     # Tom Sawyer
    76,     # Huck Finn
    119,    # A Connecticut Yankee
    102,    # Pudd'nhead Wilson
    245,    # Innocents Abroad
    11,     # Alice in Wonderland
    12,     # Through the Looking Glass
    27,     # Wonderful Wizard of Oz
    33,     # Scarlet Letter
    2701,   # Moby Dick
    15,     # Bartleby the Scrivener
    9296,   # Walden (Thoreau)
    2554,   # Crime and Punishment (English translation - kept as it dominantly trained on English)
    1232,   # The Prince (Machiavelli, English)
    16328,  # Beowulf
    100,    # Shakespeare Complete
    1112,   # Romeo and Juliet
    1129,   # Macbeth
    1524,   # Hamlet
    1513,   # King Lear
]
GUTENBERG_IDS = list(dict.fromkeys(GUTENBERG_IDS))


def fetch_gutenberg(book_id: int) -> str | None:
    """Fetch one Gutenberg book as plain text. Returns None on failure."""
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in candidates:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except Exception:
            continue
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    """Strip the leading and trailing Gutenberg license/header text."""
    start_re = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
    end_re = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
    m1 = start_re.search(text)
    if m1:
        text = text[m1.end():]
    m2 = end_re.search(text)
    if m2:
        text = text[:m2.start()]
    return text.strip()


# Cheap English / non-English detector based on stopword frequency.
# This catches French and other Romance language books reliably without
# needing a heavyweight language detector dependency.
_ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "is", "it", "with",
    "for", "as", "was", "but", "had", "his", "her", "you", "be", "on",
    "not", "have", "this", "by", "at", "they", "from", "or", "an", "she",
}
_FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "et", "à", "ne", "pas",
    "que", "qui", "il", "elle", "vous", "nous", "est", "mais", "dans", "pour",
    "ce", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "avec", "sur",
    "monsieur", "madame", "tout", "tous", "comme", "bien", "très", "votre",
}


def looks_english(text: str, sample_words: int = 5000) -> bool:
    """Returns True if the text appears to be predominantly English.

    Books on Gutenberg are huge, so the 'too short to judge' threshold is
    deliberately low - if we have less than 30 words something is wrong with
    the download anyway and we should reject it.
    """
    words = re.findall(r"[a-zA-Zà-ÿÀ-Ÿ]+", text[:200_000].lower())[:sample_words]
    if len(words) < 30:
        return False
    en = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    fr = sum(1 for w in words if w in _FRENCH_STOPWORDS)
    en_rate = en / len(words)
    fr_rate = fr / len(words)
    # English text typically hits 15-25% stopword rate. French books on this
    # detector hit 8-15% French and <5% English. Threshold is generous to
    # avoid false negatives on Shakespeare-era English.
    return en_rate >= 0.05 and en_rate > fr_rate * 1.5


def collect_gutenberg(target_bytes: int, out_path: Path) -> int:
    if out_path.exists() and out_path.stat().st_size >= target_bytes * 0.9:
        print(f"  cache hit: {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path.stat().st_size

    total = 0
    n_books = 0
    n_skipped_lang = 0
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="literature")
    with open(out_path, "w", encoding="utf-8") as f:
        for bid in GUTENBERG_IDS:
            if total >= target_bytes:
                break
            raw = fetch_gutenberg(bid)
            if raw is None:
                continue
            cleaned = strip_gutenberg_boilerplate(raw)
            if len(cleaned) < 5000:
                continue
            if not looks_english(cleaned):
                n_skipped_lang += 1
                continue
            f.write(cleaned)
            f.write("\n\n")
            chunk_bytes = len(cleaned.encode("utf-8")) + 2
            total += chunk_bytes
            n_books += 1
            pbar.update(chunk_bytes)
            time.sleep(GUTEN_DELAY)
    pbar.close()
    print(f"  literature: {n_books} books, {total:,} bytes (skipped {n_skipped_lang} non-English)")
    return total


# ---------- 2. Wikipedia ----------

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "ApolloLM/0.2 (research; contact: rburra@syr.edu)",
    "Accept": "application/json",
}


def _wiki_get(params: dict, retries: int = 4) -> dict | None:
    backoff = 1.0
    for _ in range(retries):
        try:
            r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=30)
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/json"):
                return r.json()
        except Exception:
            pass
        time.sleep(backoff)
        backoff *= 2
    return None


def _wiki_random_titles(n: int = 20) -> list[str]:
    data = _wiki_get({
        "format": "json",
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(n, 50),
    })
    if not data:
        return []
    return [p["title"] for p in data.get("query", {}).get("random", [])]


def _wiki_extract(title: str) -> str:
    data = _wiki_get({
        "format": "json",
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    })
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return page.get("extract", "") or ""
    return ""


def collect_wikipedia(target_bytes: int, out_path: Path) -> int:
    if out_path.exists() and out_path.stat().st_size >= target_bytes * 0.9:
        print(f"  cache hit: {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path.stat().st_size

    total = 0
    n_articles = 0
    seen: set[str] = set()
    consecutive_failures = 0
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="wiki")
    with open(out_path, "w", encoding="utf-8") as f:
        while total < target_bytes:
            titles = _wiki_random_titles(20)
            if not titles:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    print("\n  WARN: too many consecutive Wikipedia API failures, giving up early")
                    break
                time.sleep(2.0)
                continue
            consecutive_failures = 0
            for title in titles:
                if total >= target_bytes:
                    break
                if title in seen:
                    continue
                seen.add(title)
                content = _wiki_extract(title)
                if not content or len(content) < 500:
                    time.sleep(WIKI_DELAY)
                    continue
                f.write(content)
                f.write("\n\n")
                chunk_bytes = len(content.encode("utf-8")) + 2
                total += chunk_bytes
                n_articles += 1
                pbar.update(chunk_bytes)
                time.sleep(WIKI_DELAY)
    pbar.close()
    print(f"  wiki: {n_articles} articles, {total:,} bytes")
    return total


# ---------- 3. Python production code (v2: less test-fixture-heavy) ----------

CODE_REPOS = [
    # Web frameworks and clients - production-grade idiomatic Python.
    ("https://github.com/pallets/flask.git", "flask"),
    ("https://github.com/psf/requests.git", "requests"),
    ("https://github.com/encode/httpx.git", "httpx"),
    ("https://github.com/encode/starlette.git", "starlette"),
    ("https://github.com/tiangolo/fastapi.git", "fastapi"),
    ("https://github.com/aio-libs/aiohttp.git", "aiohttp"),
    ("https://github.com/tornadoweb/tornado.git", "tornado"),
    ("https://github.com/bottlepy/bottle.git", "bottle"),

    # Application libs and dev tools.
    ("https://github.com/pallets/click.git", "click"),
    ("https://github.com/pallets/jinja.git", "jinja"),
    ("https://github.com/pallets/werkzeug.git", "werkzeug"),
    ("https://github.com/pallets/itsdangerous.git", "itsdangerous"),
    ("https://github.com/Textualize/rich.git", "rich"),
    ("https://github.com/Textualize/textual.git", "textual"),
    ("https://github.com/python-poetry/poetry.git", "poetry"),
    ("https://github.com/pypa/pip.git", "pip"),
    ("https://github.com/pypa/setuptools.git", "setuptools"),

    # Data validation / typing.
    ("https://github.com/pydantic/pydantic.git", "pydantic"),
    ("https://github.com/python-attrs/attrs.git", "attrs"),
    ("https://github.com/pytest-dev/pluggy.git", "pluggy"),
    ("https://github.com/python/mypy.git", "mypy"),
    ("https://github.com/psf/black.git", "black"),
    ("https://github.com/PyCQA/isort.git", "isort"),
    ("https://github.com/PyCQA/flake8.git", "flake8"),

    # Async and networking.
    ("https://github.com/python-trio/trio.git", "trio"),
    ("https://github.com/MagicStack/uvloop.git", "uvloop"),
    ("https://github.com/encode/uvicorn.git", "uvicorn"),
    ("https://github.com/encode/databases.git", "databases"),

    # Notebook / scientific tooling (smaller, less test-heavy than core scipy).
    ("https://github.com/jupyter/notebook.git", "notebook"),
    ("https://github.com/ipython/ipython.git", "ipython"),

    # Documentation / templating.
    ("https://github.com/sphinx-doc/sphinx.git", "sphinx"),
    ("https://github.com/mkdocs/mkdocs.git", "mkdocs"),

    # Distributed / messaging.
    ("https://github.com/celery/celery.git", "celery"),
    ("https://github.com/celery/kombu.git", "kombu"),

    # CLI / TUI / utility.
    ("https://github.com/dbcli/pgcli.git", "pgcli"),
    ("https://github.com/dbcli/mycli.git", "mycli"),
    ("https://github.com/prompt-toolkit/python-prompt-toolkit.git", "prompt_toolkit"),
]


def shallow_clone(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True,
            timeout=300,
        )
        return True
    except Exception as exc:
        print(f"  clone failed for {url}: {exc}")
        return False


# v2: explicitly skip test directories so we get production source, not
# test fixtures. This was the root cause of v1's empty code completions.
_SKIP_DIRS = (
    "/.git/", "/vendor/", "/build/", "/dist/", "/_build/",
    "/tests/", "/test/", "/testing/",
    "/conftest", "/fixtures/",
    "/examples/", "/docs/", "/doc/",
    "/site-packages/",
)


def iter_python_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.py"):
        s = str(p) + "/"
        if any(seg in s for seg in _SKIP_DIRS):
            continue
        # Also skip files starting with test_ or _test or named conftest.
        name = p.name
        if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
            continue
        yield p


def collect_code(target_bytes: int, out_path: Path) -> int:
    if out_path.exists() and out_path.stat().st_size >= target_bytes * 0.9:
        print(f"  cache hit: {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path.stat().st_size

    clones_dir = RAW_DIR / "_clones"
    clones_dir.mkdir(exist_ok=True)

    total = 0
    n_files = 0
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="code")
    with open(out_path, "w", encoding="utf-8") as f:
        for url, name in CODE_REPOS:
            if total >= target_bytes:
                break
            dest = clones_dir / name
            if not shallow_clone(url, dest):
                continue
            for py in iter_python_files(dest):
                if total >= target_bytes:
                    break
                try:
                    src = py.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if len(src) < 200:  # skip very small modules
                    continue
                f.write(f"# === {name}/{py.relative_to(dest)} ===\n")
                f.write(src)
                f.write("\n\n")
                chunk_bytes = len(src.encode("utf-8")) + 64
                total += chunk_bytes
                n_files += 1
                pbar.update(chunk_bytes)
    pbar.close()

    try:
        shutil.rmtree(clones_dir)
    except Exception:
        pass

    print(f"  code: {n_files} files, {total:,} bytes")
    return total


# ---------- tokenize + write .bin ----------

def stream_tagged_documents(text_path: Path, tag: str) -> Iterator[str]:
    """Yield documents from a raw register file, tagged with the register marker."""
    buf: list[str] = []
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    doc = "".join(buf).strip()
                    if doc:
                        yield f"{tag}{doc}<|endoftext|>"
                    buf = []
            else:
                buf.append(line)
    if buf:
        doc = "".join(buf).strip()
        if doc:
            yield f"{tag}{doc}<|endoftext|>"


def tokenize_all(tok: ApolloTokenizer) -> tuple[np.ndarray, dict]:
    """Tokenize all three registers. Returns (ids, stats)."""
    sources = [
        ("<|literature|>", RAW_DIR / "literature.txt"),
        ("<|wiki|>",       RAW_DIR / "wiki.txt"),
        ("<|code|>",       RAW_DIR / "code.txt"),
    ]
    all_ids: list[int] = []
    stats = {}
    for tag, path in sources:
        if not path.exists():
            print(f"WARN: missing {path}, skipping")
            continue
        before = len(all_ids)
        for doc in stream_tagged_documents(path, tag):
            ids = tok.encode(doc)
            all_ids.extend(ids)
        stats[tag] = len(all_ids) - before
        print(f"  {tag}: {stats[tag]:,} tokens")
    arr = np.array(all_ids, dtype=np.uint32)
    return arr, stats


def main():
    random.seed(SEED)

    print("== Apollo v2 data prep ==")
    print(f"target: {TARGET_BYTES_PER_REGISTER // (1024*1024)} MB per register\n")

    print("[1/3] Project Gutenberg literature (English-only)")
    collect_gutenberg(TARGET_BYTES_PER_REGISTER, RAW_DIR / "literature.txt")

    print("\n[2/3] Wikipedia")
    collect_wikipedia(TARGET_BYTES_PER_REGISTER, RAW_DIR / "wiki.txt")

    print("\n[3/3] Python production code")
    collect_code(TARGET_BYTES_PER_REGISTER, RAW_DIR / "code.txt")

    # Train custom SentencePiece BPE on the actual corpus.
    print("\n[tokenize] training custom SentencePiece BPE")
    print(f"  vocab_size: {DEFAULT_VOCAB_SIZE}")
    corpus_files = [
        RAW_DIR / "literature.txt",
        RAW_DIR / "wiki.txt",
        RAW_DIR / "code.txt",
    ]
    tok = ApolloTokenizer.train(
        corpus_files=[p for p in corpus_files if p.exists()],
        out_path=DATA_DIR / "tokenizer.model",
        vocab_size=DEFAULT_VOCAB_SIZE,
    )
    tok.save(DATA_DIR / "tokenizer.model")
    print(f"  vocab_size: {tok.vocab_size}")
    print(f"  eot_id: {tok.eot_id}")
    for s in SPECIAL_TOKENS:
        print(f"    {s} -> {tok.special_id(s)}")

    print("\n[tokenize] encoding all registers")
    ids, stats = tokenize_all(tok)
    print(f"  total tokens: {len(ids):,}")

    n = len(ids)
    n_val = int(n * VAL_FRACTION)
    n_train = n - n_val
    train_ids = ids[:n_train].astype(np.uint32)
    val_ids = ids[n_train:].astype(np.uint32)

    train_path = DATA_DIR / "train.bin"
    val_path = DATA_DIR / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "version": "v2",
        "vocab_size": tok.vocab_size,
        "dtype": "uint32",
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "register_tokens": stats,
        "specials": list(SPECIAL_TOKENS),
        "target_bytes_per_register": TARGET_BYTES_PER_REGISTER,
        "tokenizer_kind": "apollo-sentencepiece-bpe",
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n== done ==")
    print(f"  train.bin: {n_train:,} tokens ({train_path.stat().st_size:,} bytes)")
    print(f"  val.bin:   {n_val:,} tokens   ({val_path.stat().st_size:,} bytes)")
    print(f"  vocab_size: {tok.vocab_size}")


if __name__ == "__main__":
    main()
