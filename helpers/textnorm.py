"""Token normalisation shared by the cut engine and the style learner.

Darija is written inconsistently: the same word appears with and without
diacritics, with any of the four alef forms, with taa marbuta or haa. Two
transcripts of the same sentence therefore differ as strings while being
identical as speech. Every diff in this project runs on normalised tokens so
that "the LLM kept this phrase" is decided by sound, not by spelling.

The normalised form is for MATCHING ONLY. Captions and reports always use the
original `display` text.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanwin, shadda, sukun, superscript alef, and the tatweel stretcher.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")

_ALEF = re.compile(r"[آأإاٱ]")          # آ أ إ ا ٱ -> ا
_YAA = re.compile(r"[ىيی]")                        # ى ي ی -> ي
_WAW = re.compile(r"[ؤو]")                              # ؤ و -> و
_TAA_MARBUTA = re.compile(r"ة")                              # ة -> ه
_HAMZA = re.compile(r"[ءئ]")                            # ء ئ -> (dropped)

_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_EXT_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}

# Anything that is not a letter, digit, or intra-word apostrophe/hyphen.
_PUNCT = re.compile(r"[^\w؀-ۿ'\-]+", re.UNICODE)

_ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_RANGE = re.compile(r"[A-Za-zÀ-ɏ]")
_DIGIT_RANGE = re.compile(r"[0-9]")

# Latin words that are French rather than English when they show up in Darija
# speech. Only used for the coarse phrase-level tag in the packed view.
_FRENCH_HINTS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "est", "c'est",
    "pour", "avec", "mais", "donc", "alors", "parce", "que", "qui", "quoi",
    "million", "milliard", "pourcent", "marché", "entreprise", "société",
    "euh", "bon", "genre", "voilà", "très", "beaucoup", "aussi", "bien",
}


def strip_diacritics(text: str) -> str:
    return _ARABIC_DIACRITICS.sub("", text)


def normalize_token(token: str) -> str:
    """Canonical matching form of a single token. May return "" (drop it)."""
    if not token:
        return ""
    t = unicodedata.normalize("NFKC", token)
    t = t.translate(_ARABIC_INDIC).translate(_EXT_ARABIC_INDIC)
    t = strip_diacritics(t)
    t = _ALEF.sub("ا", t)
    t = _YAA.sub("ي", t)
    t = _WAW.sub("و", t)
    t = _TAA_MARBUTA.sub("ه", t)
    t = _HAMZA.sub("", t)
    t = _PUNCT.sub("", t)
    t = t.strip("-'")
    return t.casefold()


def tokenize(text: str) -> list[str]:
    """Split text into raw (un-normalised) tokens."""
    if not text:
        return []
    return [t for t in re.split(r"\s+", text.strip()) if t]


def normalize_tokens(text: str) -> list[str]:
    """Tokenize then normalise, dropping tokens that normalise to nothing."""
    return [n for t in tokenize(text) if (n := normalize_token(t))]


def token_lang(token: str) -> str:
    """Coarse per-token language tag: ary | lat | num | other."""
    if _ARABIC_RANGE.search(token):
        return "ary"
    if _LATIN_RANGE.search(token):
        return "lat"
    if _DIGIT_RANGE.search(token):
        return "num"
    return "other"


def is_latin(token: str) -> bool:
    return token_lang(token) == "lat"


def guess_latin_lang(tokens: list[str]) -> str:
    """fr or en for a run of Latin tokens. Coarse; used for display tags only."""
    lowered = {normalize_token(t) for t in tokens}
    return "fr" if lowered & _FRENCH_HINTS else "en"


def phrase_lang(tokens: list[str]) -> str:
    """Language tag for a phrase: ary | fr | en | mixed."""
    kinds = {token_lang(t) for t in tokens if token_lang(t) != "other"}
    kinds.discard("num")
    if not kinds:
        return "ary"
    if kinds == {"ary"}:
        return "ary"
    if kinds == {"lat"}:
        return guess_latin_lang(tokens)
    return "mixed"
