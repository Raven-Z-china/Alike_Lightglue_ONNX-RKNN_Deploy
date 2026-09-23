#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build one graph in a fresh process and count its NPU fallbacks correctly.

WHY A SUBPROCESS, AND WHY THE OBVIOUS WAY IS WRONG
-------------------------------------------------
The RKNN build log is emitted by native code and is split across at least two
sinks with different flush behaviour.  Every in-process capture method tried here
missed part of it:

    method                                  'No lowering'  'will fallback'
    contextlib.redirect_stdout                        0               0
    os.dup2 fd 1/2 -> temp file                       0              36
    the toolkit's own `verbose_file=`                 0               0
    ... and the TERMINAL, read after the build       76              41

Each of the first three looks like a working capture, and each of them reports a
CLEAN graph for a graph with dozens of host nodes.  The `No lowering found` lines
reach the terminal through a Python-level buffer that flushes after the redirect
window has closed, so no amount of fd juggling inside the build's duration sees
them.

Capturing at the PROCESS level has none of these problems: run the build as a
child, read the child's stdout and stderr, and everything the build writes to
either stream is in hand - including buffers flushed at exit.  That is what this
script does, and it is why the numbers it prints are the ones quoted in the
README.

A SECOND MEASUREMENT, SO THE PARSE CANNOT LIE QUIETLY
----------------------------------------------------
Each count is produced twice: once by `count_fallbacks`, and once by a plain
`str.count` on the same text.  A parser bug makes the two disagree, which is
reported as a failure - rather than as a number that looks like a result.

Usage
    python scripts/build_log_probe.py --onnx weights/optimized/lightglue_stage_d7.onnx \
        --stage lightglue --dump /tmp/mm.log
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CHILD = r"""
import os, sys
sys.path.insert(0, {scripts!r})
from rknn.api import RKNN
stage, onnx, target = sys.argv[1], sys.argv[2], sys.argv[3]
# verbose=True AT CONSTRUCTION: the warnings do not exist on any other path.
rk = RKNN(verbose=True)
cfg = {{"target_platform": target, "float_dtype": "float16"}}
if stage == "alike":
    cfg["mean_values"] = [[0, 0, 0]]
    cfg["std_values"] = [[255, 255, 255]]
if rk.config(**cfg) != 0:
    raise SystemExit("config failed")
if rk.load_onnx(model=onnx) != 0:
    raise SystemExit("load_onnx failed")
rc = rk.build(do_quantization=False)
rk.release()
# The toolkit's ONNX optimiser dumps check<N>_<stage>.onnx (35-75 MB each) into the
# CWD.  Nothing reads them and four builds leave ~200 MB behind, so they go.  Kept
# inline rather than imported from `convert_to_rknn` so this probe stays torch-free.
import glob
for _p in glob.glob("check*_*.onnx"):
    try:
        os.unlink(_p)
    except OSError:
        pass
sys.stdout.flush()
sys.stderr.flush()
raise SystemExit(0 if rc == 0 else 3)
"""


def run_child(onnx, stage, target):
    """Run the build in a child process and return its combined output."""
    code = CHILD.format(scripts=str(REPO / "scripts"))
    p = subprocess.run([sys.executable, "-c", code, stage, onnx, target],
                       capture_output=True, text=True, cwd=str(REPO))
    return p.stdout + p.stderr, p.returncode


def count(text):
    from convert_to_rknn import count_fallbacks
    lines, ops, fb = count_fallbacks(text)
    return lines, ops, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--stage", required=True, choices=["alike", "lightglue"])
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--dump", default=None, help="write the raw combined log here")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    text, rc = run_child(args.onnx, args.stage, args.target)
    if args.dump:
        Path(args.dump).write_text(text, errors="replace")
    if rc != 0:
        print(f"[probe] child exit {rc}; log tail:")
        print("\n".join(text.splitlines()[-15:]))
        return rc

    lines, ops, fb = count(text)
    # Second, independent count on the same text.
    raw_no_lower = text.count("No lowering found")
    raw_will = text.count("will fallback to CPU")
    consistent = (raw_no_lower == fb["n_no_lowering"]
                  and raw_will == fb["n_will_fallback"])
    out = {"onnx": args.onnx, "stage": args.stage,
           "log_bytes": len(text),
           "fallback_nodes": fb["real"], "fallback_ops": ops,
           "n_no_lowering": fb["n_no_lowering"],
           "n_will_fallback": fb["n_will_fallback"],
           "unassigned_target_lines": fb["ambiguous"],
           "raw_no_lowering": raw_no_lower, "raw_will_fallback": raw_will,
           "parser_consistent": consistent,
           "fallback_messages": sorted(set(
               l.split("] ", 1)[-1][:120] for l in lines
               if "fallback" in l.lower()))[:6]}
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    if not consistent:
        print("[probe] PARSER INCONSISTENT - the two counts disagree, so "
              "`fallback_nodes` is not trustworthy for this build", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
