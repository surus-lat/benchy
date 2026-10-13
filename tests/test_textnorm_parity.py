# -*- coding: utf-8 -*-
"""textnorm parity: the vendored copy against fixed vectors.

The SAME vector table is locked into the reference engine and the other vendored
copy: editing a vector here without editing it there breaks those suites -- that
is the drift alarm, by design.
"""

from __future__ import annotations

import pytest

from benchy import textnorm
from benchy.textnorm import fold, tokens

# (name, input, fold(input), sorted(tokens(input)))
TEXTNORM_VECTORS = [
    ("basic", "Hello World", "hello world", ["hello", "world"]),
    ("accents_es", "  José   Ángel\tGarcía  ", "jose angel garcia",
     ["angel", "garcia", "jose"]),
    ("casefold_german", "Straße", "strasse", ["strasse"]),
    ("turkish_dotted_I", "İSTANBUL", "istanbul", ["istanbul"]),
    ("ligature_fi", "ﬁle", "file", ["file"]),
    ("fullwidth_digits", "１２３ ABC", "123 abc", ["123", "abc"]),
    ("punctuation", "factura #1.234,56!", "factura #1.234,56!",
     ["1", "234", "56", "factura"]),
    ("underscore", "snake_case", "snake_case", ["case", "snake"]),
    ("empty", "", "", []),
    ("whitespace_only", " \t\n ", "", []),
    ("none_value", None, "none", ["none"]),
    ("int_value", 123, "123", ["123"]),
    ("emoji_cjk", "Factura 📄 日本語", "factura 📄 日本語", ["factura"]),
    ("combining_e_acute", "é", "e", ["e"]),
    ("alnum_mixed", "ABC123def", "abc123def", ["abc123def"]),
    ("superscript", "x²", "x2", ["x2"]),
]


@pytest.mark.parametrize("name,value,expected_fold,_", TEXTNORM_VECTORS)
def test_fold_fixed_vectors(name, value, expected_fold, _):
    assert fold(value) == expected_fold


@pytest.mark.parametrize("name,value,_,expected_tokens", TEXTNORM_VECTORS)
def test_tokens_fixed_vectors(name, value, _, expected_tokens):
    assert sorted(tokens(value)) == expected_tokens


def test_tokens_are_an_order_free_frozenset():
    assert isinstance(tokens("a b"), frozenset)


def test_textnorm_imports_nothing_from_the_package():
    import inspect
    for line in inspect.getsource(textnorm).splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "benchy" not in stripped
