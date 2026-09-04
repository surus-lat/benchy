"""benchy.cli — `benchy new | list | run | show | export-loss`.

    benchy new <ontology>              scaffold benchmarks/<t>/<d>/<l>/
    benchy list                        walk benchmarks/, print the ontology tree
    benchy run <ref> --system URL      run a benchmark, print/save a Report
    benchy show <ref>                  print the benchmark's task/scoring/data summary
    benchy export-loss <ref> --to X    export the benchmark as a loss/metric

`list` deliberately never constructs a `Benchmark` -- it only parses each
`benchmark.yaml` with plain YAML, so it (and `--help`) work even before
`benchy.task` / `benchy.data` / `benchy.scoring` / `benchy.system` exist.
Every other verb needs at least one of those sibling modules and imports it
lazily, right where it's used, with a clean error if it's missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from benchy.core import BenchyError, OntologyPath

__all__ = ["main", "build_parser"]


_SCAFFOLD_TEMPLATE = """\
ontology: {ontology}
name: TODO name this benchmark
task:
  builtin: freeform
  instructions: TODO describe what a good answer looks like
data:
  source: "jsonl:./data/samples.jsonl"
  expected: expected
scoring: exact_match()
baselines: ["echo:"]
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchy", description="Create and run AI-system benchmarks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold benchmarks/<task>/<domain>/<language>/")
    p_new.add_argument("ontology", help="e.g. classification/sentiment/en")
    p_new.add_argument("--root", default="benchmarks", help="benchmarks root (default: benchmarks)")

    p_list = sub.add_parser("list", help="walk benchmarks/, print the ontology tree")
    p_list.add_argument("--root", default="benchmarks", help="benchmarks root (default: benchmarks)")

    p_run = sub.add_parser("run", help="run a benchmark against a system")
    p_run.add_argument("ref", help="ontology path or benchmark.yaml path/dir")
    p_run.add_argument("--system", required=True, help="system URL, e.g. openai:gpt-5-mini or echo:")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--concurrency", type=int, default=None)
    p_run.add_argument("--json", dest="json_out", default=None, help="also save the Report as JSON here")
    p_run.add_argument("--format", choices=["text", "md", "json"], default="text")

    p_show = sub.add_parser("show", help="print the benchmark's task/scoring/data summary")
    p_show.add_argument("ref", help="ontology path or benchmark.yaml path/dir")

    p_export = sub.add_parser("export-loss", help="export the benchmark as a loss function")
    p_export.add_argument("ref", help="ontology path or benchmark.yaml path/dir")
    p_export.add_argument("--to", choices=["dspy", "textgrad", "python"], required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (BenchyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    handler = {
        "new": _cmd_new,
        "list": _cmd_list,
        "run": _cmd_run,
        "show": _cmd_show,
        "export-loss": _cmd_export_loss,
    }[args.command]
    return handler(args)


# --------------------------------------------------------------------------
# new
# --------------------------------------------------------------------------


def _cmd_new(args: argparse.Namespace) -> int:
    onto = OntologyPath.parse(args.ontology)
    target_dir = Path(args.root, *onto.segments)
    yaml_path = target_dir / "benchmark.yaml"
    if yaml_path.exists():
        print(f"error: {yaml_path} already exists", file=sys.stderr)
        return 1
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "data").mkdir(exist_ok=True)
    yaml_path.write_text(_SCAFFOLD_TEMPLATE.format(ontology=str(onto)), encoding="utf-8")
    print(f"scaffolded {yaml_path}")
    return 0


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.root)
    entries = _walk_benchmarks(root)
    if not entries:
        print(f"no benchmarks found under {root}/")
        return 0
    for ontology_str, name, _path in entries:
        label = ontology_str + (f"  — {name}" if name else "")
        print(f"  {label}")
    return 0


def _walk_benchmarks(root: Path) -> list[tuple[str, str | None, Path]]:
    if not root.exists():
        return []
    entries: list[tuple[str, str | None, Path]] = []
    for yaml_path in sorted(root.rglob("benchmark.yaml")):
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict):
            continue
        ontology_str = raw.get("ontology") or str(yaml_path.parent.relative_to(root))
        entries.append((ontology_str, raw.get("name"), yaml_path))
    entries.sort(key=lambda row: row[0])
    return entries


# --------------------------------------------------------------------------
# run / show / export-loss -- these need a real Benchmark
# --------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from benchy.benchmark import load_benchmark

    bench = load_benchmark(args.ref)

    try:
        from benchy.system import load as load_system
    except ImportError as exc:
        raise ImportError(
            "'benchy.system' is not installed; cannot resolve --system "
            "(it lands from a sibling worktree and is wired in at merge)."
        ) from exc

    system = load_system(args.system)
    report = bench.run_sync(system, limit=args.limit, concurrency=args.concurrency)
    _emit_report(report, args)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from benchy.benchmark import load_benchmark

    bench = load_benchmark(args.ref)
    print(f"Benchmark: {bench.label}")
    print(f"Ontology:  {bench.ontology}")
    print(f"Task:      {bench.task!r}")
    print(f"Data:      {len(bench.data)} samples")
    print(f"Scoring:   {bench.scoring!r}")
    if bench.baselines:
        print(f"Baselines: {', '.join(bench.baselines)}")
    return 0


def _cmd_export_loss(args: argparse.Namespace) -> int:
    from benchy.benchmark import load_benchmark

    bench = load_benchmark(args.ref)

    if args.to == "python":
        print(_python_loss_snippet(bench, args.ref))
        return 0
    if args.to == "dspy":
        from benchy.loss import as_dspy_metric

        as_dspy_metric(bench)  # raises a clean, actionable error if unavailable
        print(
            "# from benchy.loss import as_dspy_metric\n"
            f"# metric = as_dspy_metric(load_benchmark({args.ref!r}))"
        )
        return 0
    if args.to == "textgrad":
        from benchy.loss import as_textgrad_loss

        as_textgrad_loss(bench)  # raises a clean, actionable error if unavailable
        print(
            "# from benchy.loss import as_textgrad_loss\n"
            f"# loss = as_textgrad_loss(load_benchmark({args.ref!r}))"
        )
        return 0
    raise AssertionError(f"unreachable: --to {args.to!r}")  # argparse `choices` guards this


def _python_loss_snippet(bench, ref: str) -> str:
    return (
        "from benchy.benchmark import load_benchmark\n\n"
        f"benchmark = load_benchmark({ref!r})\n"
        "loss = benchmark.as_loss()   # (System) -> Awaitable[float]\n"
    )


def _emit_report(report, args: argparse.Namespace) -> None:
    import benchy.report as report_mod

    if args.json_out:
        report_mod.save(report, args.json_out)
        print(f"wrote {args.json_out}")

    if args.format == "text":
        print(report_mod.render_text(report))
    elif args.format == "md":
        print(report_mod.render_markdown(report))
    elif args.format == "json" and not args.json_out:
        print(json.dumps(report_mod.to_json(report), indent=2))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
