"""iou — generic intersection-over-union.

Supports two shapes without any imaging dependency: a length-4 numeric
sequence is treated as a bounding box (x1, y1, x2, y2); anything else
iterable is treated as a set of items (pixel coordinates, label ids, tokens)
and scored via Jaccard similarity, which literally *is* IoU for masks
represented as coordinate sets.
"""

from __future__ import annotations

import pytest

from benchy.core import Scorer
from benchy.scoring.primitives import iou
from benchy.scoring.registry import parse_scorer


class TestIouBoundingBoxes:
    def test_is_a_scorer(self):
        assert isinstance(iou(), Scorer)

    def test_identical_boxes_score_one(self):
        s = iou()
        assert s.fitness((0, 0, 10, 10), (0, 0, 10, 10), None) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        s = iou()
        assert s.fitness((0, 0, 1, 1), (5, 5, 6, 6), None) == 0.0

    def test_partial_overlap_scores_between_zero_and_one(self):
        s = iou()
        value = s.fitness((0, 0, 10, 10), (5, 5, 15, 15), None)
        assert 0.0 < value < 1.0
        # intersection = 5x5=25, union = 100+100-25=175
        assert value == pytest.approx(25 / 175)


class TestIouSets:
    def test_identical_sets_score_one(self):
        s = iou()
        assert s.fitness({1, 2, 3}, {1, 2, 3}, None) == 1.0

    def test_disjoint_sets_score_zero(self):
        s = iou()
        assert s.fitness({1, 2}, {3, 4}, None) == 0.0

    def test_partial_overlap_sets(self):
        s = iou()
        value = s.fitness({1, 2, 3}, {2, 3, 4}, None)
        assert value == pytest.approx(2 / 4)

    def test_both_empty_scores_one(self):
        s = iou()
        assert s.fitness([], [], None) == 1.0

    def test_prediction_empty_expected_nonempty_scores_zero(self):
        s = iou()
        assert s.fitness([], [1, 2], None) == 0.0


class TestIouEdgeCases:
    def test_none_prediction_scores_zero(self):
        s = iou()
        assert s.fitness(None, (0, 0, 1, 1), None) == 0.0

    def test_none_expected_scores_zero(self):
        s = iou()
        assert s.fitness((0, 0, 1, 1), None, None) == 0.0

    def test_repr_round_trips(self):
        s = iou()
        assert parse_scorer(repr(s)) == s
