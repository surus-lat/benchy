"""The normative documents and the engine declare the same task vocabulary.

This guards the defect that `paper/v10-transcribe-removal-brief.md` was written to
fix: v9 described exact match as *current* and named normalized matching among future
evaluators; v10 hardened that into an absolute prohibition while simultaneously
promoting `transcribe` to a validated task. The two edits were never reconciled, and
nothing caught it — the paper and the registry disagreed in silence for a version.

Appendix B and A.4 are the two places a reader checks, so a mismatch between them (or
between either and the shipped registry) is that same defect in miniature.

Skipped when `paper/` is absent, so an installed distribution does not depend on it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchy import ontology

PAPER = Path(__file__).resolve().parents[1] / "paper"
pytestmark = pytest.mark.skipif(not PAPER.is_dir(), reason="paper/ is not distributed")


def latest(stem: str) -> Path:
    """The highest-versioned document matching `stem`."""
    def version(path: Path) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", path.stem.rsplit("v", 1)[-1]))

    candidates = sorted(PAPER.glob(f"{stem}*.md"), key=version)
    assert candidates, f"no {stem}* document in {PAPER}"
    return candidates[-1]


def engine_tasks() -> set[str]:
    return set(ontology.load("1.0")["tasks"])


def test_paper_appendix_b_registry_matches_the_shipped_registry():
    text = latest("technical-paper-v").read_text()
    appendix_b = text[text.index("# Appendix B"):]
    declared = set(re.findall(r"^  (\w+):\n    description:", appendix_b, re.M))
    assert declared == engine_tasks()


def test_paper_a4_validators_match_the_shipped_registry():
    text = latest("technical-paper-v").read_text()
    block = text[text.index("For ontology version `1.0`:"):text.index("If an ontology task is not supported")]
    assert set(re.findall(r"^(\w+)$", block, re.M)) == engine_tasks()


def test_spec_task_rules_match_the_shipped_registry():
    text = latest("benchy-engine-spec-v").read_text()
    section = text[text.index("## 5. Task validation"):text.index("## 6. Scoring")]
    assert set(re.findall(r"^- `(\w+)`:", section, re.M)) == engine_tasks()


def test_handoff_validator_table_matches_the_shipped_registry():
    text = latest("benchy-engine-agent-handoff-v").read_text()
    assert set(re.findall(r'^    "(\w+)": validate_', text, re.M)) == engine_tasks()


def test_paper_mentions_a_withdrawn_task_only_where_it_explains_the_withdrawal():
    """The brief's own consistency check: nothing before §10 may mention it.

    §10 states the limitation and Appendix E describes the extension that would
    readmit it — including a fenced block showing the constraints it *would* carry.
    Both are explanations. Anywhere earlier would be a declaration.
    """
    text = latest("technical-paper-v").read_text()
    normative = text[: text.index("## 10. Scope and future extensions")]
    assert "transcrib" not in normative.lower()


@pytest.mark.parametrize("stem", ["benchy-engine-spec-v", "benchy-engine-agent-handoff-v"])
def test_spec_and_handoff_never_declare_a_withdrawn_task(stem):
    """`transcribe` may be explained in prose, never declared as an entry."""
    for line in latest(stem).read_text().splitlines():
        if "transcrib" not in line.lower():
            continue
        # A declaration is a registry entry, a validator-table row, or a rule bullet.
        assert not re.match(r'^\s*(transcribe:|"transcribe":|- `transcribe`)', line), line


# ---------------------------------------------------------------------------
# provider documentation matches the shipped table
# ---------------------------------------------------------------------------

def test_every_documented_provider_exists_and_vice_versa():
    """A provider added to the code and forgotten in the docs, or the reverse.

    Not string-matching the snippets — that breaks on reformatting. This pins the one
    thing that actually matters: the set of provider names a reader is told about.
    """
    from benchy import providers

    root = PAPER.parent
    readme = (root / "README.md").read_text()
    skill = root / ".agent" / "skills" / "add-provider" / "SKILL.md"

    shipped = set(providers._ENDPOINTS)
    documented = {name for name in shipped if f"`{name}`" in readme}
    assert documented == shipped, f"README omits: {sorted(shipped - documented)}"

    if skill.is_file():
        text = skill.read_text()
        missing = {name for name in shipped if f'"{name}"' not in text}
        assert not missing, f"add-provider skill omits: {sorted(missing)}"


def test_each_provider_names_a_credential_the_docs_mention():
    from benchy import providers

    readme = (PAPER.parent / "README.md").read_text()
    for name, (_endpoint, credential) in providers._ENDPOINTS.items():
        assert credential in readme, f"{name}'s credential {credential} is undocumented"
