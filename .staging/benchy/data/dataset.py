"""``Data`` — the evidence, as a lazily re-iterable stream of ``Sample``.

The core contract (``benchy.core.Data``) only asks for ``__iter__``,
``__len__``, ``take`` and ``split``. This module's ``Data`` class satisfies
that protocol and adds everything a benchmark author needs day to day:
``sample``, ``filter``, ``map``, ``validate`` and ``cached``.

Laziness is the whole point. ``Data`` never wraps a generator instance
directly — it wraps a zero-argument *factory* that produces a fresh
iterator each time it's called. That's what makes ``list(data)`` safe to
call twice and what lets ``take(3)`` on a million-row source touch only the
first three rows.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

from benchy.core import LoadError, Sample

if TYPE_CHECKING:
    from benchy.data.validation import ValidationReport

#: A factory that produces a fresh, independent iterator each time it is
#: called. Source loaders and every ``Data`` transform must preserve this
#: shape — never hand back an already-partially-consumed iterator.
SampleFactory = Callable[[], Iterator[Sample]]


class Data:
    """A stream of :class:`benchy.core.Sample`.

    Construct directly from a factory for lazy/streaming sources, or via
    :meth:`from_samples` for in-memory data (tests, synthetic benchmarks).
    Every transform (``take``, ``sample``, ``filter``, ``map``, ``split``)
    returns a *new* ``Data``; nothing here ever mutates ``self``.
    """

    __slots__ = ("_factory", "_len_hint", "_cache_key", "_reload", "_meta")

    def __init__(
        self,
        factory: SampleFactory,
        *,
        len_hint: int | None = None,
        cache_key: str | None = None,
        reload: Callable[[str], "Data"] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Wrap ``factory`` — a callable returning a fresh ``Iterator[Sample]``.

        ``cache_key`` is the deterministic identity of this Data's source
        (set by :func:`benchy.data.load`); it is what :meth:`cached` keys the
        on-disk materialisation on, and it is cleared by any transform whose
        result is not reproducibly derivable from the key alone (``filter``,
        ``map``). ``reload`` is an optional hook used by :meth:`split` to
        re-resolve the *same* source with a different split name (e.g.
        re-invoking an ``hf:`` loader) when no ``meta["split"]`` field is
        available to filter on instead.
        """
        self._factory = factory
        self._len_hint = len_hint
        self._cache_key = cache_key
        self._reload = reload
        self._meta = dict(meta) if meta else {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_samples(cls, samples: Iterable[Sample]) -> "Data":
        """Wrap an in-memory collection of samples.

        The iterable is consumed once, eagerly, into a tuple — so this is
        for tests and synthetic/small data, not for lazily streaming a
        source. Re-iterating the result yields the same samples every time.
        """
        materialized = tuple(samples)
        return cls(lambda: iter(materialized), len_hint=len(materialized))

    # ------------------------------------------------------------------
    # Core protocol: __iter__, __len__, take, split
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._factory())

    def __len__(self) -> int:
        if self._len_hint is None:
            self._len_hint = sum(1 for _ in self)
        return self._len_hint

    def take(self, n: int) -> "Data":
        """The first ``n`` samples, without consuming more than ``n`` when iterated."""
        if n < 0:
            raise ValueError(f"take(n) requires n >= 0, got {n}")
        parent_factory = self._factory
        len_hint = None if self._len_hint is None else min(self._len_hint, n)
        return Data(
            lambda: itertools.islice(parent_factory(), n),
            len_hint=len_hint,
            cache_key=_derive_key(self._cache_key, f"take{n}"),
        )

    def split(self, name: str) -> "Data":
        """Select the named split.

        Two strategies, tried in order:

        1. If any sample carries ``meta["split"]``, filter to samples whose
           split matches ``name`` (case-sensitive, exact match).
        2. Otherwise, if this ``Data`` was produced by :func:`benchy.data.load`
           from a source that supports re-resolving with a different split
           (currently: ``hf:``), reload the source with ``split=name``.

        Raises :class:`benchy.core.LoadError` naming what's missing when
        neither strategy applies — that's a config/authoring error, not a
        malformed-row situation, so it raises rather than warning.
        """
        has_split_meta = any(s.meta.get("split") is not None for s in itertools.islice(self, 64))
        if has_split_meta:
            return self.filter(lambda s: s.meta.get("split") == name)
        if self._reload is not None:
            return self._reload(name)
        raise LoadError(
            f"cannot resolve split {name!r}: no sample carries meta['split'] and this "
            "Data was not loaded from a source that supports re-resolving splits "
            "(e.g. hf:). Pass split= to load() instead, or tag samples' meta['split']."
        )

    # ------------------------------------------------------------------
    # Extra transforms
    # ------------------------------------------------------------------

    def sample(self, n: int, *, seed: int = 0) -> "Data":
        """A deterministic random subset of size ``n`` (reservoir sampling).

        Uses Algorithm R seeded by ``random.Random(seed)`` so the same seed
        over the same source order always yields the same subset, while only
        ever holding ``n`` samples in memory regardless of source size.
        """
        if n < 0:
            raise ValueError(f"sample(n) requires n >= 0, got {n}")
        parent_factory = self._factory

        def factory() -> Iterator[Sample]:
            rng = random.Random(seed)
            reservoir: list[Sample] = []
            for i, item in enumerate(parent_factory()):
                if i < n:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, i)
                    if j < n:
                        reservoir[j] = item
            return iter(reservoir)

        return Data(factory, len_hint=None, cache_key=_derive_key(self._cache_key, f"sample{n}s{seed}"))

    def filter(self, predicate: Callable[[Sample], bool]) -> "Data":
        """Keep only samples for which ``predicate(sample)`` is truthy."""
        parent_factory = self._factory

        def factory() -> Iterator[Sample]:
            return (s for s in parent_factory() if predicate(s))

        return Data(factory)

    def map(self, fn: Callable[[Sample], Sample]) -> "Data":
        """Apply ``fn`` to every sample. ``fn`` must return a ``Sample``."""
        parent_factory = self._factory
        len_hint = self._len_hint

        def factory() -> Iterator[Sample]:
            return (fn(s) for s in parent_factory())

        return Data(factory, len_hint=len_hint)

    def validate(self, schema: dict[str, Any]) -> "ValidationReport":
        """Validate every sample's ``input`` against a JSON Schema.

        Reports; does not raise. See :mod:`benchy.data.validation`.
        """
        from benchy.data.validation import validate_data

        return validate_data(self, schema)

    def cached(self) -> "Data":
        """Materialise this ``Data`` to the local on-disk cache.

        If this ``Data`` carries a stable ``cache_key`` (i.e. it — or an
        ancestor reached only through ``take``/``sample``/``split`` — came
        from :func:`benchy.data.load`), the materialisation is written under
        that key and is reused across processes/runs. Otherwise (after a
        ``filter``/``map`` with an arbitrary Python callable, which cannot be
        assumed stable across runs) it is written to a fresh directory for
        the lifetime of this call — still avoiding recomputation for the
        ``Data`` object returned, just not deduplicated across runs.
        """
        from benchy.data.cache import materialize

        return materialize(self)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def cache_key(self) -> str | None:
        return self._cache_key


def _derive_key(parent: str | None, token: str) -> str | None:
    if parent is None:
        return None
    return f"{parent}/{token}"
