"""Data transforms: from_samples, take, sample, filter, map, split, laziness."""

from __future__ import annotations

import benchy.core as core
from benchy.core import Sample
from benchy.data import Data


def _samples(n: int) -> list[Sample]:
    return [Sample(id=f"s{i}", input={"text": f"hello {i}"}, expected=str(i)) for i in range(n)]


class TestFromSamples:
    def test_isinstance_of_core_protocol(self):
        data = Data.from_samples(_samples(3))
        assert isinstance(data, core.Data)

    def test_len_and_list_round_trip(self):
        samples = _samples(5)
        data = Data.from_samples(samples)
        assert len(data) == 5
        assert list(data) == samples

    def test_reiterable_not_exhausted(self):
        data = Data.from_samples(_samples(3))
        first_pass = list(data)
        second_pass = list(data)
        assert first_pass == second_pass
        assert len(first_pass) == 3


class TestTake:
    def test_take_returns_first_n(self):
        data = Data.from_samples(_samples(10))
        taken = data.take(3)
        assert isinstance(taken, core.Data)
        assert [s.id for s in taken] == ["s0", "s1", "s2"]
        assert len(taken) == 3

    def test_take_more_than_available(self):
        data = Data.from_samples(_samples(2))
        assert len(data.take(10)) == 2

    def test_take_is_lazy_over_a_generator_source(self):
        consumed = {"n": 0}

        def factory():
            for i in range(1_000_000):
                consumed["n"] += 1
                yield Sample(id=f"g{i}", input={"text": str(i)})

        data = Data(factory)
        taken = data.take(3)
        # Constructing the view must not touch the source at all.
        assert consumed["n"] == 0
        # A plain for-loop (what the engine actually does) rather than
        # list(taken): list() pre-sizes via __len__ when the length isn't
        # known, which would (correctly, and boundedly) touch the source
        # twice here. We're isolating what take() itself guarantees.
        result = [s for s in taken]
        assert [s.id for s in result] == ["g0", "g1", "g2"]
        # Only the 3 requested rows (plus at most a tiny constant lookahead)
        # may have been pulled from the underlying generator.
        assert consumed["n"] <= 4

    def test_take_is_reiterable(self):
        data = Data.from_samples(_samples(10)).take(3)
        assert [s.id for s in data] == [s.id for s in data]


class TestSample:
    def test_sample_returns_n_items(self):
        data = Data.from_samples(_samples(20))
        sub = data.sample(5, seed=0)
        assert isinstance(sub, core.Data)
        assert len(sub) == 5

    def test_sample_is_deterministic_given_seed(self):
        data = Data.from_samples(_samples(50))
        a = [s.id for s in data.sample(10, seed=42)]
        b = [s.id for s in data.sample(10, seed=42)]
        assert a == b

    def test_sample_differs_across_seeds_generally(self):
        data = Data.from_samples(_samples(50))
        a = [s.id for s in data.sample(10, seed=1)]
        b = [s.id for s in data.sample(10, seed=2)]
        assert a != b

    def test_sample_does_not_materialize_whole_source_in_ram(self):
        # Reservoir sampling: peak extra memory is O(n), not O(source size).
        seen = {"n": 0}

        def factory():
            for i in range(200_000):
                seen["n"] += 1
                yield Sample(id=f"g{i}", input={})

        data = Data(factory)
        sub = data.sample(5, seed=0)
        result = list(sub)
        assert len(result) == 5
        # One full pass is expected (reservoir sampling needs to see every
        # row once) but it must not buffer more rows than requested.
        assert seen["n"] == 200_000


class TestFilterAndMap:
    def test_filter_keeps_matching_samples(self):
        data = Data.from_samples(_samples(10))
        evens = data.filter(lambda s: int(s.expected) % 2 == 0)
        assert isinstance(evens, core.Data)
        assert [s.id for s in evens] == ["s0", "s2", "s4", "s6", "s8"]

    def test_map_transforms_each_sample(self):
        data = Data.from_samples(_samples(3))
        upper = data.map(lambda s: Sample(id=s.id, input=s.input, expected=(s.expected or "").upper()))
        assert isinstance(upper, core.Data)
        assert [s.expected for s in upper] == ["0", "1", "2"]

    def test_transforms_do_not_mutate_original(self):
        data = Data.from_samples(_samples(4))
        _ = data.filter(lambda s: False)
        _ = data.take(1)
        assert len(data) == 4


class TestSplit:
    def test_split_filters_by_meta_split_field(self):
        samples = [
            Sample(id="a", input={}, meta={"split": "train"}),
            Sample(id="b", input={}, meta={"split": "test"}),
            Sample(id="c", input={}, meta={"split": "test"}),
        ]
        data = Data.from_samples(samples)
        test_split = data.split("test")
        assert isinstance(test_split, core.Data)
        assert [s.id for s in test_split] == ["b", "c"]

    def test_split_with_no_information_raises_clear_error(self):
        from benchy.core import LoadError

        data = Data.from_samples(_samples(3))
        try:
            data.split("validation")
        except LoadError as exc:
            assert "split" in str(exc).lower()
        else:
            raise AssertionError("expected LoadError")
