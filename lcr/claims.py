"""Object-claim extraction: caption -> list of claims over the fixed vocabulary V."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import nltk

from . import config
from .vocab import KIND, SURFACE_TO_CLASS

_WORD = re.compile(r"[a-z]+")
_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
nltk.data.path.insert(0, str(config.DATA / "nltk"))
_LEM = nltk.WordNetLemmatizer()


@dataclass(frozen=True)
class Claim:
    cls: str          # canonical class in V
    kind: str         # person / thing / stuff
    word: str         # surface word as written
    index: int        # word index in the tokenised caption


def words(caption: str) -> list[str]:
    """Lower-cased alphabetic words; 't-shirt' is normalised to 'tshirt'."""
    return _WORD.findall(caption.lower().replace("t-shirt", "tshirt").replace("t shirt", "tshirt"))


@lru_cache(maxsize=100_000)
def _lemma(word: str) -> str:
    return _LEM.lemmatize(word, "n")


def word_class(word: str) -> str | None:
    """Vocabulary class of a single word, or None. Surface form first, then lemma."""
    if word in SURFACE_TO_CLASS:
        return SURFACE_TO_CLASS[word]
    return SURFACE_TO_CLASS.get(_lemma(word))


def extract_from_words(ws: list[str], dedupe: bool = True) -> list[Claim]:
    """Claims from an already tokenised caption.

    A word is a claim if (i) the POS tagger marks it as a noun, (ii) it maps to
    a class in V, and (iii) it is not a modifier of a following vocabulary noun
    ("water bottle" claims a bottle, not water). With `dedupe`, only the first
    mention of each class is kept, so each caption contributes one claim per class.
    """
    if not ws:
        return []
    tags = [t for _, t in nltk.pos_tag(ws)]
    out: list[Claim] = []
    seen: set[str] = set()
    for i, (w, t) in enumerate(zip(ws, tags)):
        if t not in _NOUN_TAGS:
            continue
        cls = word_class(w)
        if cls is None:
            continue
        # Noun-compound modifier: the next word is a noun that is itself in V
        # ("water bottle", "bike rider"). Requiring the head to be in V avoids
        # dropping claims when the tagger mislabels a participle as a noun.
        if (i + 1 < len(ws) and tags[i + 1] in _NOUN_TAGS
                and word_class(ws[i + 1]) is not None):
            continue
        if dedupe and cls in seen:
            continue
        seen.add(cls)
        out.append(Claim(cls=cls, kind=KIND[cls], word=w, index=i))
    return out


def extract(caption: str, dedupe: bool = True) -> list[Claim]:
    return extract_from_words(words(caption), dedupe=dedupe)


def reference_classes(refs: list[str]) -> set[str]:
    """Union of vocabulary classes mentioned in the reference captions.

    Uses a lexical match without the POS / compound filters, i.e. the most
    permissive reading of the references, so that 'unsupported by references'
    is not inflated by tagger errors.
    """
    out: set[str] = set()
    for r in refs:
        for w in words(r):
            c = word_class(w)
            if c is not None:
                out.add(c)
    return out


_NON_VISUAL = {"camera": __import__("re").compile(r"\b(at|to|toward|towards|from|for|into) the camera\b")}
# Compounds whose object word does not claim that object (docs/CHECK_S4_PROTOKOL.md, I5); used by Check S4 only.
_COMPOUNDS = {"plate": re.compile(r"\bhome plates?\b"), "dog": re.compile(r"\bhot ?dogs?\b"),
              "car": re.compile(r"\btoy cars?\b"), "bicycle": re.compile(r"\b(dirt|motor) ?bikes?\b")}


def non_visual(draft: str, cls: str, compounds: bool = False) -> bool:
    """Idioms that use an object word without claiming the object is visible ('smiles at the camera'); with
    `compounds`, also compounds that name another thing ('hot dog', 'home plate', 'toy car', 'dirt bike')."""
    d = draft.lower()
    pat = _NON_VISUAL.get(cls)
    if pat and pat.search(d):
        return True
    pat = _COMPOUNDS.get(cls) if compounds else None
    return bool(pat and pat.search(d))

