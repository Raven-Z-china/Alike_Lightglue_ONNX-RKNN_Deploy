"""Vendored model implementations, kept in the upstream directory shape.

The submodule path (`model/matchers/lightglue.py`) deliberately mirrors the
training framework's (`gluefactory/models/matchers/lightglue.py`) so that a diff
against upstream is a one-line command and the provenance of any hunk is obvious.

See `model/matchers/lightglue.py` for what was trimmed and why it cannot change a
result; `scripts/verify_vendored_matcher.py` is what checks that claim.
"""
from .lightglue import LightGlue

__all__ = ["LightGlue"]
