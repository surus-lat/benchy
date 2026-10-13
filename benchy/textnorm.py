# VENDORED from programpipeline 158119be4b50a196b8a132339cd551496d54bc62 (program_pipeline/textnorm.py) — no editar
"""Text normalization: one fold, one tokenizer.

NFKD + casefold + strip combining marks + collapse whitespace; tokens are the
``[a-z0-9]+`` runs of the folded text, as a frozenset. VENDORED copy of the
reference implementation; parity is locked by fixed vectors in
``tests/test_textnorm_parity.py`` -- any drift breaks that test.

Stdlib-only, imports nothing from the package.
"""
import re
import unicodedata


def fold(text):
    """casefold + strip accents + collapse whitespace -- the shared fold."""
    folded = unicodedata.normalize("NFKD", str(text).strip().casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(folded.split())


def tokens(text):
    """Alphanumeric tokens of the folded text, order-free."""
    return frozenset(re.findall(r"[a-z0-9]+", fold(text)))
