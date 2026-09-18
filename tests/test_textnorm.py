"""Normalisation is the floor the whole cut engine stands on.

If two spellings of the same Darija word stop matching, the diff sees a deleted
word that was never deleted and the editor cuts speech Ali kept.
"""

from __future__ import annotations

from helpers import textnorm as T


def test_diacritics_and_taa_marbuta_collapse():
    assert T.normalize_token("الشَّرِكَة") == T.normalize_token("الشركه") == "الشركه"


def test_all_alef_forms_are_one():
    forms = ["اقتصاد", "أقتصاد", "إقتصاد", "آقتصاد", "ٱقتصاد"]
    assert len({T.normalize_token(f) for f in forms}) == 1


def test_arabic_indic_digits_become_ascii():
    assert T.normalize_token("٢٠٢٦") == "2026"
    assert T.normalize_token("۲۰۲۶") == "2026"


def test_latin_is_casefolded_and_punctuation_stripped():
    assert T.normalize_token("SpaceX,") == "spacex"
    assert T.normalize_token("«OpenAI»") == "openai"


def test_tatweel_is_not_a_difference():
    assert T.normalize_token("صـــافي") == T.normalize_token("صافي")


def test_tokens_that_are_pure_punctuation_are_dropped():
    assert T.normalize_tokens("... salam -- khoya ?") == ["salam", "khoya"]


def test_phrase_language_tags():
    assert T.phrase_lang(["الشركة", "ديال"]) == "ary"
    assert T.phrase_lang(["donc", "le", "marché"]) == "fr"
    assert T.phrase_lang(["the", "market", "opened"]) == "en"
    assert T.phrase_lang(["الشركة", "OpenAI"]) == "mixed"
    # A number alone does not decide the language of the phrase.
    assert T.phrase_lang(["الشركة", "2026"]) == "ary"
