#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assert the deployable path never imports the training framework.

WHY AN EXECUTABLE CHECK
-----------------------
"Self-contained" is the kind of claim that decays.  A vendored module gets a
convenience import added, a probe copies a helper out of the framework "just for
now", and nothing fails on the machine where the framework happens to be
installed on `sys.path`.  The breakage only surfaces later, on a clone, as an
ImportError in the middle of a conversion.

So this script runs each deployable entry point in a subprocess with
`gluefactory` made unimportable at the interpreter level, and reports which ones
demand it.  Two are allowed to:

    pruning/finetune.py          launches `python -m gluefactory.train`
    scripts/verify_vendored_matcher.py  compares the port against the original

Everything else must pass.  `--help` is the probe: it exercises the whole import
graph of the file without doing any work, which is exactly the part that would
carry a stray dependency.

    python scripts/check_selfcontained.py
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The two documented exceptions, with the reason each one is allowed.
ALLOWED = {
    "pruning/finetune.py": "launches `python -m gluefactory.train`",
    "scripts/verify_vendored_matcher.py": "compares the port against the original",
}

BLOCKER = '''\
"""Make `gluefactory` unimportable, to prove nothing on the deploy path needs it."""
import importlib.abc
import sys


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "gluefactory" or name.startswith("gluefactory."):
            raise ModuleNotFoundError("blocked by check_selfcontained.py: " + name)
        return None


sys.meta_path.insert(0, _Block())
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    targets = sorted([p for p in (REPO / "scripts").glob("*.py")
                      if p.name != Path(__file__).name]
                     + list((REPO / "pruning").glob("*.py"))
                     + list((REPO / "eval").glob("*.py")))

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "sitecustomize.py").write_text(BLOCKER)
        env = {"PYTHONPATH": tmp}

        blocked, ok, exceptions = [], [], []
        for t in targets:
            rel = str(t.relative_to(REPO))
            r = subprocess.run([args.python, str(t), "--help"],
                               cwd=str(REPO), env={**env}, capture_output=True, text=True)
            if rel in ALLOWED:
                # Listed regardless of exit code, and NOT counted as clean: both
                # reach for the framework at RUN time, so `--help` (which only
                # walks the import graph) cannot observe the dependency.  Calling
                # these "independent" because their import succeeded would be the
                # reassuring answer rather than the true one.
                exceptions.append(rel)
                print(f"  runtime  {rel}\n             -> {ALLOWED[rel]}")
                continue
            if r.returncode == 0:
                ok.append(rel)
                print(f"  ok       {rel}")
            else:
                blocked.append(rel)
                tail = [l for l in r.stderr.strip().splitlines() if l.strip()][-1:]
                print(f"  BLOCKED  {rel}   {tail[0] if tail else ''}")

    print(f"\n{len(ok)} independent at import time, "
          f"{len(exceptions)} known run-time exceptions, {len(blocked)} blocked")
    print("Scope: `--help` exercises each file's whole import graph, so it catches a "
          "vendored\nmodule that reaches back for the framework.  It says nothing "
          "about run-time needs,\nwhich is why the two exceptions above are listed "
          "rather than counted.")
    if blocked:
        print("\nThe deployable path is NOT self-contained. Offenders:")
        for b in blocked:
            print(f"  - {b}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
