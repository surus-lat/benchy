"""Academic domains in ontology 1.0 — added for university benchmarks.

Additive identifiers: they change nothing for a benchmark that does not use
them, and a field like anthropology no longer has to hide under `general`.
"""

import pytest

from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError

BENCHMARK = """
version: "1.0"
ontology_version: "1.0"
benchmark: {{ task: classify, domain: {domain}, language: es }}
program:
  input: {{ question: string }}
  output: {{ answer: {{ enum: [a, b, c] }} }}
scoring:
  weights: {{ answer: 1 }}
  aggregator: weighted_mean
data: {{ path: ./exam.jsonl }}
ai-system: {{ type: model, provider: bedrock, model: us.moonshotai.kimi-k3 }}
"""


@pytest.mark.parametrize("domain", ["anthropology", "sociology", "history", "linguistics", "biology", "public_health"])
def test_academic_domains_compile(domain):
    assert compile_benchmark(BENCHMARK.format(domain=domain))["benchmark"]["domain"] == domain


def test_unknown_domains_are_still_refused():
    with pytest.raises(BenchyError) as err:
        compile_benchmark(BENCHMARK.format(domain="astrology"))
    assert err.value.code == "unknown_domain"
