"""Definition-of-done checks: the exact assertions the module spec and the
cross-module integration seam suite exercise against `benchy.scoring`."""

from __future__ import annotations

import benchy.core
import benchy.scoring as scoring
from benchy.scoring import (
    BaseScorer,
    binary,
    cer,
    clamp,
    contains,
    exact_match,
    f1_token,
    field_wise,
    field_wise_weighted,
    invert,
    iou,
    levenshtein_ratio,
    mean,
    mse,
    multiple_choice_accuracy,
    numeric_close,
    parse_scorer,
    pearson,
    regex_match,
    register_scorer,
    registry,
    restrict,
    threshold,
    weighted_sum,
    wer,
)

REQUIRED_NAMES = [
    "BaseScorer",
    "exact_match", "contains", "regex_match", "f1_token", "numeric_close",
    "wer", "cer", "multiple_choice_accuracy", "pearson", "mse", "iou",
    "levenshtein_ratio",
    "field_wise", "field_wise_weighted", "list_wise",
    "binary", "threshold", "restrict", "mean", "weighted_sum", "invert", "clamp",
    "parse_scorer", "register_scorer", "registry",
]


class TestRequiredApiSurface:
    def test_every_required_name_is_exported(self):
        for name in REQUIRED_NAMES:
            assert hasattr(scoring, name), f"missing export: {name}"

    def test_all_matches_the_required_names_exactly(self):
        assert set(scoring.__all__) == set(REQUIRED_NAMES)

    def test_isinstance_of_core_scorer(self):
        assert isinstance(exact_match(), benchy.core.Scorer)

    def test_every_primitive_is_a_scorer(self):
        instances = [
            exact_match(), contains(), regex_match(pattern=r"x"), f1_token(),
            numeric_close(), wer(), cer(), multiple_choice_accuracy(), pearson(),
            mse(), iou(), levenshtein_ratio(),
        ]
        for instance in instances:
            assert isinstance(instance, benchy.core.Scorer)

    def test_structural_and_transforms_are_scorers(self):
        instances = [
            field_wise(fields=("a",), per_field=exact_match()),
            field_wise_weighted(fields=("a",), per_field=exact_match(), weights={"a": 1}),
            scoring.list_wise(item_scorer=exact_match()),
            binary(exact_match()),
            threshold(exact_match(), cutoff=0.5),
            restrict(exact_match(), fields=("a",)),
            mean(exact_match()),
            weighted_sum({"a": exact_match()}, weights={"a": 1.0}),
            invert(exact_match()),
            clamp(exact_match()),
        ]
        for instance in instances:
            assert isinstance(instance, benchy.core.Scorer)


class TestSpecAcceptanceRoundTrip:
    def test_the_exact_round_trip_from_the_module_spec(self):
        s = binary(field_wise_weighted(fields=("a", "b"), per_field=exact_match(), weights={"a": 2}))
        assert parse_scorer(repr(s)) == s


class TestIntegrationSeamShapes:
    """Verbatim call shapes from the cross-module seam suite
    (tests/benchy/integration/test_seams.py) so a regression here is caught
    before the merge gate runs."""

    def test_scoring_exact_match(self):
        assert isinstance(scoring.exact_match(), benchy.core.Scorer)

    def test_scoring_field_wise(self):
        s = scoring.field_wise(fields=("name",), per_field=scoring.exact_match())
        assert isinstance(s, benchy.core.Scorer)

    def test_scoring_binary_of_field_wise(self):
        s = scoring.binary(scoring.field_wise(fields=("a", "b"), per_field=scoring.exact_match()))
        assert isinstance(s, benchy.core.Scorer)

    def test_scoring_wer(self):
        assert isinstance(scoring.wer(), benchy.core.Scorer)

    def test_parse_scorer_repr_round_trip_is_stable(self):
        s = scoring.field_wise(fields=("name",), per_field=scoring.exact_match())
        rebuilt = scoring.parse_scorer(repr(s))
        assert repr(parse_scorer(repr(rebuilt))) == repr(s)

    def test_evaluate_three_positional_args(self):
        sample = benchy.core.Sample(id="s1", input={})
        s = scoring.exact_match()
        score = s.evaluate("hi", "hi", sample)
        assert score.value == 1.0

    def test_aggregate_on_a_list_of_scores_reports_fitness(self):
        s = scoring.exact_match()
        sample = benchy.core.Sample(id="s1", input={})
        score_a = s.evaluate("hi", "hi", sample)
        score_b = s.evaluate("hi", "bye", sample)
        agg = s.aggregate([score_a, score_b])
        assert agg["fitness"] == 0.5

    def test_wer_inverts_so_a_better_transcript_scores_higher(self):
        sample = benchy.core.Sample(id="s1", input={})
        s = scoring.wer()
        good = s.fitness("hola mundo", "hola mundo", sample)
        bad = s.fitness("chau planeta", "hola mundo", sample)
        assert good > bad

    def test_cer_inverts_so_a_better_transcript_scores_higher(self):
        sample = benchy.core.Sample(id="s1", input={})
        s = scoring.cer()
        good = s.fitness("hola mundo", "hola mundo", sample)
        bad = s.fitness("chau planeta", "hola mundo", sample)
        assert good > bad

    def test_mse_inverts_so_a_closer_number_scores_higher(self):
        s = scoring.mse()
        close = s.fitness(100.0, 101.0, None)
        far = s.fitness(100.0, 999.0, None)
        assert close > far

    def test_field_wise_perfect_and_uniformly_wrong_are_exact(self):
        s = scoring.field_wise(fields=("name",), per_field=scoring.exact_match())
        exp = {"name": "Acme"}
        perfect = s.evaluate({"name": "Acme"}, exp, None)
        wrong = s.evaluate({"name": "Nope"}, exp, None)
        assert s.aggregate([perfect])["fitness"] == 1.0
        assert s.aggregate([wrong])["fitness"] == 0.0


class TestImportIsInstant:
    def test_importing_benchy_scoring_does_not_pull_in_heavy_deps(self):
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import benchy.scoring\n"
            "heavy = {'jiwer', 'Levenshtein', 'numpy', 'scipy', 'rapidfuzz', 'jsonschema'}\n"
            "loaded = heavy & set(sys.modules)\n"
            "assert not loaded, f'heavy deps eagerly imported: {loaded}'\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
