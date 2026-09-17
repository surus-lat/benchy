"""Shared fixtures: the canonical benchmark from paper §9."""

from __future__ import annotations

import textwrap

import pytest
import yaml

CANONICAL = textwrap.dedent("""
    version: "1.0"
    ontology_version: "1.0"

    benchmark:
      task: extract
      domain: finance
      language: es

    program:
      input:
        image: image
      output:
        invoice_number: string
        date: date
        supplier: string
        subtotal: float
        total: float

    scoring:
      weights:
        invoice_number: 1
        date: 1
        supplier: 1
        subtotal: 1
        total: 5
      aggregator: weighted_mean

    data:
      path: ./data/invoices.jsonl

    ai-system:
      type: external
      id: invoice-extractor-v7
""").strip()


def edit(**changes) -> str:
    """The canonical document with top-level sections replaced or removed.

    A value of `None` deletes the key, so unknown/missing-key cases stay readable.
    """
    doc = yaml.safe_load(CANONICAL)
    for key, value in changes.items():
        key = key.replace("ai_system", "ai-system")
        if value is None:
            doc.pop(key, None)
        else:
            doc[key] = value
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


@pytest.fixture
def canonical() -> str:
    return CANONICAL
