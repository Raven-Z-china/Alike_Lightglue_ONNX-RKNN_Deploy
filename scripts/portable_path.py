#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""How committed reports refer to files.

Reports are written to a directory outside the repository, and they are
shared, so every path in them is public.  An
absolute path publishes the filesystem layout of whatever machine produced the
report - which is both a needless leak and actively unhelpful, because it cannot
be resolved by anyone reading it afterwards.

The rule is therefore: **store a path relative to the repository whenever the
file is inside it, and a descriptive name when it is not.**  A checkpoint that
lives in the training checkout becomes `<checkpoint:name>` rather than the
absolute path it happened to have, and the reader is expected to supply
`GLUEFACTORY_ROOT` themselves.

The function is total - it never fails on an odd input, because a helper that
raises inside report-building would lose the whole run's results.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Where a non-repository file is expected to come from, by the root it is under.
#: Used to name the placeholder, so a report says *what* is missing and how to
#: supply it rather than just flagging it as external.
EXTERNAL_ROOTS = (
    ("GLUEFACTORY_ROOT", "checkpoint"),
    ("HPATCHES_ROOT", "hpatches"),
)


def _external_label(p: Path) -> str:
    """Name an out-of-repo file by its role, not by where it sits on disk."""
    for env_var, label in EXTERNAL_ROOTS:
        root = os.environ.get(env_var)
        if root:
            try:
                rel = p.relative_to(Path(root))
                return f"<{label}:{rel}>"
            except ValueError:
                pass
    return f"<external:{p.name}>"


def portable(p, role: str = "") -> str:
    """Render `p` for a committed report: repo-relative, or a role label.

    `role` names what the file is ("checkpoint") and forces the label form even
    when the path happens to resolve inside the repository.  Without it, the same
    checkpoint written as `weights/...` in one report and as a label in another
    would depend on which directory the caller happened to run from - a report
    should not vary that way.
    """
    if p is None:
        return ""
    try:
        path = Path(p)
    except TypeError:
        return str(p)
    if role in ("checkpoint", "onnx"):
        text = str(p)
        if text.startswith(f"<{role}:") and text.endswith(">"):
            # Already labelled.  Re-labelling would nest the label (`<onnx:<onnx:x>>`)
            # and there is no way to tell that from a real filename afterwards.
            return text
        return f"<{role}:{path.name}>"
    # Resolve so that a relative argument and an absolute one agree, but do not
    # require the file to exist - a report may be written about a path that was
    # deleted afterwards, and failing here would be worse than a stale label.
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for base in (REPO,):
        for candidate in (resolved, path):
            try:
                return str(candidate.relative_to(base))
            except ValueError:
                continue
    return _external_label(resolved)
