"""Audit AI benchmark module.

Benchmarks for the 2-stage medical transfer audit pipeline:
- Algorithm 1: Policy Evaluation (7 policies, batched LLM call)
- Algorithm 2: Recommendation Synthesis (final audit decision)

Tasks are discovered automatically from the .py files in this directory.
"""

from .policy_eval import PolicyEval
from .recommendation import Recommendation

__all__ = ["PolicyEval", "Recommendation"]