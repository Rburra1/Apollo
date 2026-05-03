# Apollo Samples

Generated text from trained Apollo checkpoints. Each block records: model size, val_loss, prompt, decoding params, and the output.

---

## Apollo v2 (big, val 3.95)

### What changed from v1

v2 was a multi-variable change targeting v1's three documented failure modes (French collapse on wiki OOD, [Illustration] loops on literature OOD, semantically-empty code completions). All three changes shipped in one training run, so attribution to a single fix isn't possible without ablation.

| Lever | v1 | v2 |
|--|--|--|
| Tokenizer | tiktoken GPT-2 BPE (vocab=50260) | Custom SentencePiece BPE trained on the actual corpus (vocab=32000) |
| Literature corpus | Project Gutenberg, no language filter | Same corpus + English-only stopword-frequency filter on new books |
| Code corpus | numpy, pandas, django, scipy, sqlalchemy (test-fixture-heavy, 4,272 files) | flask, requests, fastapi, click, rich, pydantic, mypy, black, sphinx, celery, etc - 38 production-grade libs with test directories explicitly excluded |
| Train.py | Cosine LR keeps stepping past best val | Early stopping on val plateau (patience=8 evals) |

### Architecture and training

- **Architecture:** unchanged from v1 (8L / 8H / 512 embed, block_size=256, weight tying, GPT-2 init)
- **Params:** 41.70M total, 25.31M body, 10.24M token embedding (smaller than v1's 19.30M because vocab dropped 50260 → 32000)
- **Corpus:** 117MB raw (47.9MB literature + 52.4MB wiki + 22.2MB code), 36.2M tokens
- **Token mix:** 12.5M literature / 12.7M wiki / 11.0M code - much closer to balanced than v1's 12.9M / 12.3M / 24.0M
- **Training:** 8000 iters, batch=24, block=256, lr=2.5e-4 cosine to 2.5e-5
- **Wall clock:** 8.00 hours throttled
- **Best val:** 3.9537 (random baseline ln(32000) = 10.37)

### Loss reduction comparison vs v1

Absolute val losses are not comparable across different vocab sizes because the random-guess baseline differs. The right comparison is loss reduction from baseline:

| | v1 | v2 |
|--|--|--|
| Random baseline | 10.82 | 10.37 |
| Best val | 3.67 | 3.95 |
| **Reduction** | **7.15** | **6.42** |

v2 reduced loss less than v1. Plausible reason: v2 trained on 36.2M tokens vs v1's 49.2M tokens (custom BPE produces fewer tokens per byte than GPT-2 BPE). Same iteration count means each iter saw less unique content. The capacity lever (body params) was held constant, so this is a data-volume effect, not a capacity effect.

The interesting question is whether v2's qualitative outputs are better despite the higher val loss number. Stress-test results below.

### Stress-test results: did v1's failure modes get fixed?

**1. French collapse on wiki OOD: FIXED.**

v1 prompted with "Quantum entanglement" produced "et l'au du mêtre, et la suivait d'est fépé" - ungrammatical French. v2 produced an English sports-bio article (low quality content, but English structure throughout). Diagnosis confirmed: GPT-2's BPE had French character-bigram priors from web training data. A custom BPE trained on our predominantly-English corpus removed those priors entirely.

**2. [Illustration] loop on literature OOD: FIXED.**

v1 prompted with "The spaceship descended onto" collapsed into endless `[Illustration]` markers after one short OOD continuation. v2 produced "the wall, which they looked in a deep window, as it is a white fire" - incoherent imagery but stayed in literary voice for 200 tokens without falling into a fixed-string loop. Diagnosis: v1 learned `[Illustration]` as a high-probability fallback token when uncertain; v2's tokenizer doesn't allocate a single token to that string, so the model can't lock onto it as a fallback.

**3. Code register as test fixtures: FIXED, with caveats.**

v1's `def fibonacci(n):` produced `tm.assert_numpy_array_equal(result, expected)` followed by GH-issue test stubs. v2 produced production-style code: type hints (`list[str | None]`), proper `__init__` signatures with `*` keyword-only separators, docstrings in real-codebase format, `__slots__` declarations. Caveat: still semantically empty - `class NeuralNetwork:` doesn't define a network. Stylistic register learned, function semantics not.

### New failure modes that emerged in v2

**1. Wiki occasionally produces non-Latin garbage characters.**

`<|wiki|>The Roman Empire was` produced one sentence then degraded to `"Chinese: م Рери Гк. МояящКи Зииияа Иеуегя"`. Diagnosis: SentencePiece's `byte_fallback=True` lets the model emit raw bytes when no token matches well. When the model gets uncertain mid-generation, it falls into byte-fallback territory. Different failure than v1's French (which was at least valid French character sequences); v2's byte-fallback is obviously broken which is arguably preferable for downstream evaluation.

**2. Wiki content quality dropped overall.**

v1 produced government/military noun word-salad. v2 produces sports-biography templated nonsense ("South Korean club," "2018 International League," "born August 1976"). The model learned Wikipedia's dominant article type (sports / people biographies) and applies that template to any wiki prompt. Same overfitting failure as v1, different surface manifestation.

**3. Literature character names are now invented.**

v1 produced consistent character names from real novels (D'Artagnan, Athos, Don Quixote, Sancho). v2 produces invented names that follow play-script conventions: NORA, HORA, CORA, PRINA. Diagnosis: v1's GPT-2 BPE had memorized those proper-noun token sequences; v2's custom BPE doesn't. Result: more creative, less coherent. Tradeoff between memorization-fluency and generative diversity.

### Net qualitative assessment

v2 is **better at OOD prompts** and **better stylistically on code**. v2 is **worse at producing surface-coherent text** on familiar prompts. The val loss says v1 won; the failure-mode tests say v2 won.

Choosing between them depends on what you optimize for: a model that confidently regurgitates training patterns (v1), or one that gracefully degrades to noise instead of falling into specific bad attractors (v2). For a research artifact, v2 is the more honest baseline.

### Sample 1 - in-distribution literature

**Prompt:** `<|literature|>The night was`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200 | **seed:** 1

```
The night was gone out, and said, that a little or good-matter is a
doubt, who went on it - a woman of a great deal that's
whom in it.

NORA. I could not go on, and I must have to make it a good mind to be
dressable. What do you want to mean?

NORA. You know anything I't.
I am glad to give you to take you, and you.

[_Exeunt._]

HORA. What's the most words of you?

CORA. How do you see the world?

PRINA. We will not bear my mind and call them more.
And do you say your master, and you are not a good
That is in the same. What do you?

NORA. I have
```

Notice the play-format conventions: speaker tags in caps, stage direction `[_Exeunt._]`. The character names are invented (NORA, HORA, CORA, PRINA) but follow real conventions.

### Sample 2 - in-distribution wiki

**Prompt:** `<|wiki|>The Roman Empire was`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200 | **seed:** 1

```
The Roman Empire was established in the early 19th century.

== See also ==
Musonius de Bras
Néa de la Mau

== References ==

== External links ==
"The Mess of the Art of the F.S.S.
"The Little S. (in French)
"The D.S.S., a Roman Church of G."

The Church of America (Chinese: م Рери Гк. МояящКи Зииияа Иеуегя
Иее вдяри Ира Знооди вах МИезадуниедоаливестотодя. ани Шибиуака ве
```

Wikipedia structural conventions are learned (See also, References, External links sections, parenthetical foreign-language gloss). Content is nonsense ("19th century" Roman Empire) and the model degrades to byte-fallback Unicode noise after ~150 tokens.

### Sample 3 - in-distribution code

**Prompt:** `<|code|>def fibonacci(n):`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200 | **seed:** 1

```python
def fibonacci(n):
    """
    Returns a function that is a generic context to create
    a function for a object.

* method can be called by a function.
*args - the value in the class name.
"""
return t
return t
return t

def __init__(
    self,
    **kwargs: type[Any],
    *,
    default: bool = False,
) -> None:
    if isinstance(func, bytes):
        return default
    if not isinstance(ctx, bytes):
        for key, value in self.model_defaults():
            if key in {exc.info.info.name} or not in value
            raise AttributeError(v, value)
        if default is not None:
            # If you want it is not a custom value, then
            # the type is the other, otherwise a value is set.
            return _set_info(value, value)
    return value
```

Type hints are realistic. The `*,` keyword-only separator pattern is correct. `__init__` signature is plausible. The fibonacci function itself does not compute fibonacci.

### Sample 4 - OOD literature (v1 failure mode: pattern collapse)

**Prompt:** `<|literature|>The spaceship descended onto`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200 | **seed:** 1

```
The spaceship descended onto the wall, which they looked
in a deep window, as it is a white fire, and a
spine it seemed for the little, and the other man, as to a long
happiness, it seemed to say, that an important one had been so
fold to the rest of a few months, and so he had gone on some of
pictor.

Bes, who had no one another, in some time he had been a bad
happet, and it had been very curious to do, but it was nothing
took him, it was not, in the case of his family, had
wowsing a long time. What does he know, if he were not so
in the first thing, he could not see that he would have been
thankful of his name, he might, then he could not know not. What do
nothing else? Where's he? And why?
```

No `[Illustration]` loop. Stayed in literary register for 200 tokens. Content is incoherent but the failure mode from v1 did not appear.

### Sample 5 - OOD wiki (v1 failure mode: French collapse)

**Prompt:** `<|wiki|>Quantum entanglement`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200 | **seed:** 1

```
Quantum entanglement (born August 1976) is a professional football
player who played as an professional football for the South Korean
club. He is the first professional football player.

== Early life ==
Massic worked at the 2018 International League (in Italian football)
and football statistics and professional football in the University
of Florida. His second season was in the New York City League.
Otow was born in the New York City of Fame in England.

== Career ==
He played in the New York State Cup for the New England–Way ===
=== 2000–195 ===
=== football ===
She played for the World Cup in the 2012-196 season, the United
States in the 2024–70–48–08 season. In 2005, she finished the first
season of the 2008 World Cup, and a gold medal for the 2000-19 season.
```

No French. The model interprets "Quantum entanglement" as a person and produces a sports-biography. Failure mode is now over-confident category-misclassification rather than language collapse.

### Sample 6 - OOD code (v1 failure mode: empty completion)

**Prompt:** `<|code|>class NeuralNetwork:`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200 | **seed:** 1

```python
class NeuralNetwork:
    """
    from .core.core import json
    """

class GetPath(Exception):
    """
    A dict of a function instance.
    """

__slots__ = None

def __init__(
    self,
    *,
    *,
    *,
    description: list[str | None] = None,
    **kwargs: Any,
) -> bytes:
    """Add a string to be used to be named.
    A class (e. a *path* object)
    is a list of :class:`Response`.
    """

def __init__(
    self,
    name: str,
    type: str,
    method: Path,
    alias: str,
    default: str | None = None,
    *,
    mode: tuple[str, float] | None = None,
) -> None:
    """
    Returns an object or a tuple.

Args:
    name: A value to use (e.g
```

Production-style type signatures, docstring conventions including `:class:` Sphinx-style references, `Args:` block format. The `*,` separator appears multiple times in one signature which is invalid Python - the model learned the pattern but not its constraint.

### Lessons for v3

1. The wiki register's failure mode is now overfitting to dominant article type (sports biographies). v3 should rebalance the wiki sample to oversample non-biography articles (history, science, geography).
2. SentencePiece's byte_fallback creates an obvious tell when the model is confused. Could be a feature (downstream evaluation can flag it) or a bug (looks worse to humans). Worth deciding which.
3. The architecture/data ratio is approximately the right ballpark for a 50M body-param model on 36M tokens, but training tokens is the binding constraint. v3 should target 100M+ tokens.
4. Single-variable ablation experiments are now overdue. v2 changed three things at once and we cannot attribute fixes to specific levers. A small ablation grid would help.

---

## Apollo v1 (big, val 3.67)

- **Architecture:** 8 layers, 8 heads, 512 embed dim, block_size=256
- **Params:** 51.05M total (25.31M body, 19.30M token embedding due to BPE vocab)
- **Tokenizer:** tiktoken GPT-2 BPE with three special tokens added (vocab=50260)
- **Corpus:** 150MB across three registers, each tagged at document start
  - 47.9MB Project Gutenberg literature (60 books)
  - 52.4MB English Wikipedia (12,644 articles)
  - 52.5MB Python source from popular libraries (4,272 files)
- **Training:** 8000 iters, batch=24, block=256, lr=2.5e-4 cosine to 2.5e-5
- **Wall clock:** 9.67 hours throttled (target 8h)
- **Best val:** 3.6748 (random baseline ln(50260) = 10.82, so ~1100x better than random)

### What worked in v1

The register switching mechanism is real. Each register tag produces text in the right voice:

- `<|literature|>` produces archaic dialogue, character names, narrator framing
- `<|wiki|>` produces encyclopedic noun-phrase chains and date/place patterns
- `<|code|>` produces syntactically valid Python with realistic test-fixture patterns

This was the central research question - does the model learn three registers or average them into one. Apollo v1 learned three.

### What failed in v1

Stress-testing on out-of-distribution prompts revealed three failure modes:

1. **Literature register is heavily memorization.** OOD prompt "The spaceship descended onto" produced one short OOD continuation, then collapsed into Gutenberg patterns ("the Countess," "house of the house"), then into a `[Illustration]` loop - a high-probability fallback the model learned because Gutenberg books contain illustration markers.
2. **Wiki register collapses to fake-French on OOD prompts.** "Quantum entanglement" produced ungrammatical French-adjacent text ("et l'au du mêtre," "fépé"). Diagnosis: GPT-2's BPE has French character-bigram priors from web training, and the model fell back to those under prompt OOD-ness.
3. **Code register is syntactically valid but semantically empty.** `class NeuralNetwork:` ended the document immediately, then the model produced pandas/numpy GH-issue test stubs. Code corpus was too narrow - 12 popular libs heavy on test fixtures. Model learned "code looks like this" but not function semantics.

### Sample 1 - in-distribution literature (v1)

**Prompt:** `<|literature|>The night was`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200

```
The night was the old man, and the man that was the
good gentleman.

"My dear, I am not a little," said Morrel.

"Oh, I want you to speak with me," said Don Quixote, with
the face-man, taking an hour; and we had gone to see him.

"Good enough," replied the count, "but you were, and as I will get up
yourself, and not the best of this; for I am to-morrow."

"Now," replied Sancho, "you shall go up and end my way?"

"Ah, I am afraid, I am sure my dear fellow."

"What have you," said Athos.

"We had the first time," replied D'Artagnan, "but
```

Note Athos, D'Artagnan, Sancho, Don Quixote, Morrel - characters from across multiple novels in the same 200-token sample.

### Sample 2 - in-distribution wiki (v1)

**Prompt:** `<|wiki|>The Roman Empire was`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200

```
The Roman Empire was on the city of the South American Navy, and at the
University of Representatives and was moved to the President, but the U.S.
from the American Empire the National Minister were included in the National
Government under the New York City of the University of France. The National
Department of the National Institute of California was held on the National
National Football League of the United States.

The South Wales Army became part of the National Party of the National War
(R) in the National Academy of Washington Council, and the National
Assembly of Public Army to the West of Representatives, and the National
Assembly of the National Army.
```

The model loops on government/military noun phrases.

### Sample 3 - in-distribution code (v1)

**Prompt:** `<|code|>def fibonacci(n):`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200

```python
def fibonacci(n):
     # Test that the dtype of the values
      # should have the result in an array.
      result = getattr(n, 2), dtype)
     result = isinstance(result, dict)
    tm.assert_numpy_array_equal(result, expected)

def test_numpy_numeric_values(self):
    # GH 298
    df = DataFrame(
        [0, 1],
        [1, 2],
       columns=MultiIndex.from_tuples(["a", "
```

Function definition has nothing to do with fibonacci. Model immediately switches to a pandas test fixture pattern.

### Sample 4 - OOD literature failure (v1)

**Prompt:** `<|literature|>The spaceship descended onto`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200

```
The spaceship descended onto the whole country.

[Illustration]

The Countess was in the court of the morning, which was the
of the world.

(With course of the house of the house was in the
migrations in the world. It is in the world of the
time-a man!-a new. I have already seen a great thing to
the man of the same family?

[Illustration:
[Illustration:]
[Illustration:
[... continues to loop on Illustration ...]
```

### Sample 5 - OOD wiki failure (v1)

**Prompt:** `<|wiki|>Quantum entanglement`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200

```
Quantum entanglement et l'au du mêtre, et la suivait d'est fépé à l'éé
dit d'étant pas et, l'équ'ément d'enment a'une.

Je de sa cette pême mais avères que se fécous n'est les
l'un fait, vous nous vous, nous pas, mais n'il pas toutément fait le
dont décient, la sous un nous vous quelques.

«

-- The monsieur, je d'est bien.
```

The model switched to ungrammatical French.

### Sample 6 - OOD code failure (v1)

**Prompt:** `<|code|>class NeuralNetwork:`
**Temperature:** 0.8 | **top_k:** 40 | **max_tokens:** 200

```python
class NeuralNetwork:

class TestData(Benchmark):
   def test_invalid(self):
       # GH 1360
      # Test that the type
       # if the data is not None.
       # ValueError: GH#1959.

if using_string is not None:
          assert not self._engine is not None and not is not None
        else:
           assert not self._engine == "pyarrow"
          else:
```

Model ends the document on the OOD class definition, then reverts to test fixtures.
