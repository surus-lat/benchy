"""nb CLI — take an exam: python -m nb /sentiment good -o out.json"""
import argparse
import json
import sys
from pathlib import Path

from . import Exam, locate


def main():
    ap = argparse.ArgumentParser(prog="nb", description=__doc__)
    ap.add_argument("exam", help="ontology path, e.g. /sentiment")
    ap.add_argument("system", help="system name under <exam>/systems/ or path to a spec json")
    ap.add_argument("-o", "--out", required=True, help="artifact json; same path re-run = resume")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tries", type=int, default=3)
    a = ap.parse_args()
    d = Path(a.exam) if Path(a.exam).is_dir() else locate(a.exam)
    spec_path = Path(a.system) if a.system.endswith(".json") else d / "systems" / f"{a.system}.json"
    art = Exam(d).run(json.loads(spec_path.read_text()), out=Path(a.out),
                      workers=a.workers, tries=a.tries)
    # errors is a projection over the records — the CLI is its one reader
    errors = sum(c["status"] == "error" for c in art["cases"])
    print(f"{a.exam} score={art['score']:.3f} errors={errors} "
          f"done={len(art['cases'])}/{art['total']} -> {a.out}")
    # errors-projection gate: the exit code reports operability (did every case
    # complete?), not quality (how good were the answers?) — c13: a
    # score<1 gate MUTATED in and survived the whole suite until the
    # dumb-stub judge (score 0.5, zero errors must exit 0).
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()