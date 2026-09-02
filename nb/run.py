"""CLI: python3 -m nb.run <bench_dir> <system_name>"""
import json
import sys

from . import bench


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print("usage: python3 -m nb.run <bench_dir> <system>", file=sys.stderr)
        return 2
    bench_dir, system_name = argv[0], argv[1]
    system = bench.load_system(bench_dir, system_name)
    result = bench.run(bench_dir, system)
    bench.save(result, bench_dir, system_name)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())