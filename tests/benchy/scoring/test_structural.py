"""field_wise, field_wise_weighted, list_wise.

field_wise is the highest-value salvage in this package (see
`src/tasks/common/utils/structured_metrics_calculator.py`). The design here
is deliberately narrower than that god-object: field_wise applies ONE
per-field scorer uniformly across a set of dotted field paths, and gets its
partial-credit behavior entirely from whatever `per_field` scorer is passed
in (usually `exact_match()`, sometimes `numeric_close()` restricted to
numeric fields via the `restrict` transform).

Three decisions this test file locks in, per a reference-benchmark review of
`src/tasks/structured/.data/chat_extract_data.jsonl` (nested Spanish lead
extraction: `{"nombre": str|null, "tiene_negocio": bool|null, "negocio":
{"descripcion_negocio": str|null, "meses_en_negocio": int|null,
"cantidad_empleados": int|null}}`):

1. Dotted paths work: `fields=("nombre", "negocio.descripcion_negocio")`.
2. `fields=None` means "every leaf path of the expected object" (dicts are
   descended into; lists are treated as opaque leaves — use `list_wise` for
   those). This is recomputed per sample since `expected` shapes may vary.
3. Null handling distinguishes "field absent from the prediction entirely"
   (tracked separately, in `breakdown["missing"]`) from "field present with
   value None" (scored normally — and `exact_match`/`numeric_close` treat
   `(None, None)` as correct, so a correctly-predicted null scores 1.0).
"""

from __future__ import annotations

import pytest

from benchy.core import Scorer
from benchy.scoring.primitives import exact_match, numeric_close
from benchy.scoring.registry import parse_scorer
from benchy.scoring.structural import field_wise, field_wise_weighted, list_wise

FIELDS = ("vendor", "total", "due_date")


class TestFieldWiseBasics:
    def test_is_a_scorer(self):
        assert isinstance(field_wise(fields=FIELDS, per_field=exact_match()), Scorer)

    def test_perfect_prediction_scores_exactly_one(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        pred = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        assert s.fitness(pred, exp, None) == 1.0

    def test_uniformly_wrong_prediction_scores_exactly_zero(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        pred = {"vendor": "Wrong", "total": "0", "due_date": "1999-01-01"}
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        assert s.fitness(pred, exp, None) == 0.0

    def test_partial_credit_averages_per_field_scores(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        pred = {"vendor": "Acme", "total": "0", "due_date": "1999-01-01"}
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        assert s.fitness(pred, exp, None) == pytest.approx(1 / 3)

    def test_end_to_end_through_aggregate_is_exact_no_smoothing(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        perfect_pred = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        wrong_pred = {"vendor": "X", "total": "0", "due_date": "1999-01-01"}
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        scores = [s.evaluate(perfect_pred, exp, None), s.evaluate(perfect_pred, exp, None)]
        assert s.aggregate(scores)["fitness"] == 1.0
        scores = [s.evaluate(wrong_pred, exp, None), s.evaluate(wrong_pred, exp, None)]
        assert s.aggregate(scores)["fitness"] == 0.0

    def test_breakdown_carries_a_value_per_field(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        pred = {"vendor": "Acme", "total": "0", "due_date": "2024-01-01"}
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        score = s.evaluate(pred, exp, None)
        assert score.breakdown["fields"]["vendor"]["value"] == 1.0
        assert score.breakdown["fields"]["total"]["value"] == 0.0
        assert score.breakdown["fields"]["due_date"]["value"] == 1.0

    def test_field_absent_from_prediction_is_tracked_as_missing(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        pred = {"vendor": "Acme"}  # total, due_date entirely absent
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        score = s.evaluate(pred, exp, None)
        assert set(score.breakdown["missing"]) == {"total", "due_date"}
        # missing fields still count against the score (scored as None vs value).
        assert score.value == pytest.approx(1 / 3)

    def test_field_not_requested_is_ignored_even_if_present(self):
        s = field_wise(fields=("vendor",), per_field=exact_match())
        pred = {"vendor": "Acme", "total": "999"}
        exp = {"vendor": "Acme", "total": "100"}
        assert s.fitness(pred, exp, None) == 1.0

    def test_repr_shows_fields_and_per_field(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        assert repr(s) == "field_wise(fields=('vendor', 'total', 'due_date'), per_field=exact_match(case_insensitive=True, strip=True))"

    def test_repr_round_trips(self):
        s = field_wise(fields=FIELDS, per_field=exact_match())
        assert parse_scorer(repr(s)) == s

    def test_accepts_a_list_of_fields_and_normalizes_to_a_tuple(self):
        s = field_wise(fields=["vendor", "total"], per_field=exact_match())
        assert s.fields == ("vendor", "total")


class TestFieldWiseDottedPaths:
    def test_nested_field_is_scored_via_dotted_path(self):
        s = field_wise(fields=("nombre", "negocio.descripcion_negocio"), per_field=exact_match())
        pred = {"nombre": "Maria", "negocio": {"descripcion_negocio": "Estudio contable"}}
        exp = {"nombre": "Maria", "negocio": {"descripcion_negocio": "Estudio contable"}}
        assert s.fitness(pred, exp, None) == 1.0

    def test_nested_field_mismatch_is_detected(self):
        s = field_wise(fields=("negocio.descripcion_negocio",), per_field=exact_match())
        pred = {"negocio": {"descripcion_negocio": "Restaurante"}}
        exp = {"negocio": {"descripcion_negocio": "Estudio contable"}}
        assert s.fitness(pred, exp, None) == 0.0

    def test_missing_parent_object_resolves_to_missing_not_a_crash(self):
        s = field_wise(fields=("negocio.descripcion_negocio",), per_field=exact_match())
        pred = {}  # no "negocio" key at all
        exp = {"negocio": {"descripcion_negocio": "Estudio contable"}}
        score = s.evaluate(pred, exp, None)
        assert score.value == 0.0
        assert "negocio.descripcion_negocio" in score.breakdown["missing"]


class TestFieldWiseNullHandling:
    """The chat_extract benchmark's null-as-correct-answer case."""

    def test_predicting_null_when_null_is_expected_scores_one(self):
        s = field_wise(fields=("negocio.meses_en_negocio",), per_field=exact_match())
        pred = {"negocio": {"meses_en_negocio": None}}
        exp = {"negocio": {"meses_en_negocio": None}}
        assert s.fitness(pred, exp, None) == 1.0

    def test_predicting_a_value_when_null_is_expected_scores_zero(self):
        s = field_wise(fields=("negocio.meses_en_negocio",), per_field=exact_match())
        pred = {"negocio": {"meses_en_negocio": 48}}
        exp = {"negocio": {"meses_en_negocio": None}}
        assert s.fitness(pred, exp, None) == 0.0

    def test_predicting_null_when_a_value_is_expected_scores_zero(self):
        s = field_wise(fields=("negocio.meses_en_negocio",), per_field=exact_match())
        pred = {"negocio": {"meses_en_negocio": None}}
        exp = {"negocio": {"meses_en_negocio": 48}}
        assert s.fitness(pred, exp, None) == 0.0

    def test_null_present_is_distinguished_from_field_absent(self):
        s = field_wise(fields=("negocio.meses_en_negocio",), per_field=exact_match())
        # Present-with-null (correct) must NOT show up in "missing".
        pred = {"negocio": {"meses_en_negocio": None}}
        exp = {"negocio": {"meses_en_negocio": None}}
        score = s.evaluate(pred, exp, None)
        assert score.breakdown["missing"] == []
        assert score.value == 1.0

    def test_type_coercion_via_exact_match_stringification(self):
        # "48" vs 48, and "true" vs True: exact_match stringifies both sides.
        s = field_wise(fields=("meses", "activo"), per_field=exact_match())
        pred = {"meses": "48", "activo": "true"}
        exp = {"meses": 48, "activo": True}
        assert s.fitness(pred, exp, None) == 1.0


class TestFieldWiseAutoDiscoverFields:
    def test_fields_none_discovers_every_leaf_path_of_expected(self):
        s = field_wise(fields=None, per_field=exact_match())
        pred = {
            "nombre": "Maria",
            "tiene_negocio": True,
            "negocio": {"descripcion_negocio": "Estudio contable", "meses_en_negocio": 48, "cantidad_empleados": 8},
        }
        exp = {
            "nombre": "Maria",
            "tiene_negocio": True,
            "negocio": {"descripcion_negocio": "Estudio contable", "meses_en_negocio": 48, "cantidad_empleados": 8},
        }
        assert s.fitness(pred, exp, None) == 1.0

    def test_fields_none_still_catches_a_wrong_nested_leaf(self):
        s = field_wise(fields=None, per_field=exact_match())
        pred = {"nombre": "Maria", "negocio": {"descripcion_negocio": "Restaurante", "meses_en_negocio": 48}}
        exp = {"nombre": "Maria", "negocio": {"descripcion_negocio": "Estudio contable", "meses_en_negocio": 48}}
        assert s.fitness(pred, exp, None) < 1.0

    def test_fields_none_repr_round_trips(self):
        s = field_wise(fields=None, per_field=exact_match())
        assert parse_scorer(repr(s)) == s
        assert repr(s) == "field_wise(fields=None, per_field=exact_match(case_insensitive=True, strip=True))"


class TestFieldWiseWeighted:
    def test_is_a_scorer(self):
        assert isinstance(field_wise_weighted(fields=FIELDS, per_field=exact_match(), weights={"total": 3}), Scorer)

    def test_weights_bias_the_average(self):
        s = field_wise_weighted(
            fields=FIELDS, per_field=exact_match(), weights={"vendor": 1, "total": 3, "due_date": 2}
        )
        # only "total" (weight 3) is correct; vendor(1) and due_date(2) wrong.
        pred = {"vendor": "Wrong", "total": "100", "due_date": "wrong"}
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        assert s.fitness(pred, exp, None) == pytest.approx(3 / 6)

    def test_missing_weight_defaults_to_one(self):
        s = field_wise_weighted(fields=("vendor", "total"), per_field=exact_match(), weights={"total": 3})
        pred = {"vendor": "Wrong", "total": "100"}
        exp = {"vendor": "Acme", "total": "100"}
        assert s.fitness(pred, exp, None) == pytest.approx(3 / 4)

    def test_repr_preserves_the_partial_weights_dict_unmerged(self):
        s = field_wise_weighted(fields=("a", "b"), per_field=exact_match(), weights={"a": 2})
        assert "weights={'a': 2}" in repr(s)

    def test_repr_round_trips(self):
        s = field_wise_weighted(
            fields=("vendor", "total"), per_field=exact_match(), weights={"vendor": 1, "total": 3}
        )
        assert parse_scorer(repr(s)) == s

    def test_perfect_prediction_scores_exactly_one_end_to_end(self):
        s = field_wise_weighted(fields=FIELDS, per_field=exact_match(), weights={"total": 5})
        exp = {"vendor": "Acme", "total": "100", "due_date": "2024-01-01"}
        scores = [s.evaluate(exp, exp, None)]
        assert s.aggregate(scores)["fitness"] == 1.0


class TestBinaryRoundTripsThroughFieldWiseWeighted:
    """The exact acceptance check from the module spec."""

    def test_nested_binary_field_wise_weighted_round_trips(self):
        from benchy.scoring.transforms import binary

        s = binary(field_wise_weighted(fields=("a", "b"), per_field=exact_match(), weights={"a": 2}))
        assert parse_scorer(repr(s)) == s


class TestListWiseOrderSensitive:
    def test_is_a_scorer(self):
        assert isinstance(list_wise(item_scorer=exact_match()), Scorer)

    def test_identical_lists_score_one(self):
        s = list_wise(item_scorer=exact_match())
        assert s.fitness(["a", "b", "c"], ["a", "b", "c"], None) == 1.0

    def test_pairs_index_wise_by_default(self):
        s = list_wise(item_scorer=exact_match())
        assert s.fitness(["a", "x", "c"], ["a", "b", "c"], None) == pytest.approx(2 / 3)

    def test_length_mismatch_penalizes_via_none_padding(self):
        s = list_wise(item_scorer=exact_match())
        value = s.fitness(["a", "b"], ["a", "b", "c"], None)
        assert value == pytest.approx(2 / 3)

    def test_both_empty_scores_one(self):
        s = list_wise(item_scorer=exact_match())
        assert s.fitness([], [], None) == 1.0

    def test_none_prediction_treated_as_empty_list(self):
        s = list_wise(item_scorer=exact_match())
        value = s.fitness(None, ["a", "b"], None)
        assert value == 0.0

    def test_breakdown_reports_counts_and_per_item_scores(self):
        s = list_wise(item_scorer=exact_match())
        score = s.evaluate(["a", "x"], ["a", "b"], None)
        assert score.breakdown["n_predicted"] == 2
        assert score.breakdown["n_expected"] == 2
        assert len(score.breakdown["items"]) == 2

    def test_repr_round_trips(self):
        s = list_wise(item_scorer=exact_match(), order_sensitive=True)
        assert parse_scorer(repr(s)) == s


class TestListWiseOrderInsensitive:
    def test_reordered_lists_score_one(self):
        s = list_wise(item_scorer=exact_match(), order_sensitive=False)
        assert s.fitness(["c", "a", "b"], ["a", "b", "c"], None) == 1.0

    def test_finds_the_best_alignment_not_just_any(self):
        s = list_wise(item_scorer=exact_match(), order_sensitive=False)
        # best alignment: pred[1]->exp[0] ("a"), pred[0]->exp[1] ("x" vs "b" wrong),
        # pred[2]->exp[2] ("c"): 2 out of 3 correct is achievable.
        value = s.fitness(["x", "a", "c"], ["a", "b", "c"], None)
        assert value == pytest.approx(2 / 3)

    def test_repr_round_trips(self):
        s = list_wise(item_scorer=exact_match(), order_sensitive=False)
        assert parse_scorer(repr(s)) == s
