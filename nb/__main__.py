"""`benchy` is `python -m nb` until packaging earns a console script."""
import sys

from . import cli
from .cli import run, new

# the verb table (cycle 8): at two verbs, if/elif IS the table — a registry
# dict would be indirection wearing framework clothing.  `main` died as a
# concept: the dispatch is module code, argv is read once, and the tests
# drive the real process (subprocess), so the argv=None testability seam
# was dead weight.
argv = sys.argv[1:]
try:
    if argv[:1] == ["run"]:
        flags = dict(zip(argv[3::2], argv[4::2]))
        run(*argv[1:3], **{k.lstrip("-"): int(v) for k, v in flags.items()})
    elif argv[:1] == ["new"]:
        new(*argv[1:2])
    else:
        raise SystemExit(cli.__doc__.strip())
except (LookupError, ValueError, FileNotFoundError, TypeError) as e:
    raise SystemExit(f"benchy: {e}")