"""A deterministic stand-in AI-system, so the example runs offline.

This is an adapter in the full sense of paper A.10: it takes the named-field input
object and returns a named-field output object. Everything about *how* it produces
that — here, a regex; elsewhere, an HTTP call to a provider, a local model, or a
whole agent — is its own business and invisible to the engine.

It is deliberately imperfect: it misreads the third invoice's total, so the example
produces a score that is neither 0 nor 1 and the weights visibly do something. That
third example scores 4/9 — five of six fields right, but the one it missed carries
weight 5 of the 9 on offer.

`supplier.tax_id` has weight 0, so the benchmark score is identical whether this
function extracts it correctly or not. That is what a zero weight means: the field is
still required, still type-checked, still reported in `field_scores` — it simply does
not move the number.
"""

import re


def extractor(input_object):
    text = input_object["text"]

    def find(pattern, default=""):
        match = re.search(pattern, text)
        return match.group(1) if match else default

    total = float(find(r"Total ([\d.]+)", "0"))
    if find(r"Factura (\S+)") == "B-777":
        total = 60.0  # an off-by-a-cent misread, on the heaviest-weighted field

    return {
        "invoice_number": find(r"Factura (\S+)"),
        "date": find(r"del (\d{4}-\d{2}-\d{2})"),
        "supplier": {
            "name": find(r"\. ([^,]+), CUIT"),
            "tax_id": find(r"CUIT ([\d-]+)"),
        },
        "subtotal": float(find(r"Subtotal ([\d.]+)", "0")),
        "total": total,
    }
