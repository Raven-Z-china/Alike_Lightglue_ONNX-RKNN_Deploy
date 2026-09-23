#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark the converted stages: what actually runs on the NPU, and how much.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
On-board latency CANNOT be measured here.  There is no RK3588 attached, and the
RKNN PC "simulator" runs the NPU's op sequence on the x86 host - a number from it
is the speed of a host CPU emulating an NPU, which is neither the right magnitude
nor the right ordering across op types.  No absolute timing from this script is a
deployment claim, and the report says so.

What CAN be measured here, and is the thing that decides whether a change was
worth making:

1. **How many nodes fall off the NPU.**  Each one is a host round-trip in the
   middle of the graph: the NPU stalls, a tensor is copied out, the host computes
   it, and the result is copied back.  That is a real, structural cost and it is
   the dominant one for a graph like this matcher.  This is a COUNT, not a
   timing, so it is exact.
2. **The op-type breakdown of those nodes**, which says which rewrite to do next.
3. **Model size**, which is a deployment constraint in its own right.
4. **Relative simulator time.**  Same host, same simulator, same op mix except
   for the change under test - so the RATIO between two variants carries
   information even though neither absolute value does.  Reported as a ratio,
   never as a latency.

THE AUDIT DETAIL THAT MAKES (1) TRUSTWORTHY
-------------------------------------------
`RKNN(verbose=False)` suppresses the warnings that report fallbacks: on this graph
it emits 480 kB with zero `will fallback to CPU` lines, against 1.2 MB with 36 of
them when verbose.  A benchmark that builds quietly and then greps the log
therefore reports a clean graph for a graph with 36 host nodes - which is exactly
what happened here.  `capture_build_log` forces verbose for the build and restores
the caller's setting afterwards, and a per-stage log-size floor fails the run if
the capture regresses to the quiet path.

Usage
    python scripts/bench_stages.py --stage1 weights/original/alike_stage_bil.onnx \
        --stage2 weights/original/lightglue_stage_einsum.onnx \
                 weights/optimized/lightglue_stage_d7.onnx \
        --out <reports>/bench_stages.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from convert_to_rknn import (capture_build_log, count_fallbacks,  # noqa: E402
                             drop_optimizer_dumps)


def make_rknn(onnx, norm, target):
    """Construct an RKNN with the configuration this benchmark requires.

    `verbose=True` IS PART OF THE MEASUREMENT, NOT A LOGGING PREFERENCE.  The
    toolkit only emits the `will fallback to CPU` warnings on its verbose path:

        RKNN(verbose=False)                        480,129 bytes    0 warnings
        RKNN(verbose=False) + later set_log_level  866,686 bytes    0 warnings
        RKNN(verbose=True)                       1,216,961 bytes   36 warnings

    A benchmark built quietly therefore reports zero fallbacks for a graph with 36
    of them.  The post-hoc setter is the trap - it grows the log while still
    omitting the warnings - so `verbose` must reach the CONSTRUCTOR.  Nothing is
    printed by this: `capture_build_log` diverts the build's fd 1/2 into a file.
    """
    from rknn.api import RKNN

    rk = RKNN(verbose=True)
    cfg = {"target_platform": target, "float_dtype": "float16"}
    if norm:
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    if rk.config(**cfg) != 0:
        raise RuntimeError(f"config failed for {onnx}")
    if rk.load_onnx(model=str(onnx)) != 0:
        raise RuntimeError(f"load_onnx failed for {onnx}")
    return rk


def log_floor(stage):
    """A per-stage floor for the captured log, used as a regression guard.

    The two graphs differ by ~7x in verbose log size, so one number cannot serve
    both: 600 kB would reject every valid stage-1 build (it emits ~170 kB), while
    a low floor would fail to catch a quiet matcher build at 480 kB.
    """
    return 600_000 if stage == "lightglue" else 80_000


def build_and_audit(onnx, norm, target, stage, min_log_bytes=None):
    """Build one ONNX graph and report its NPU residency, exactly."""
    rk = make_rknn(onnx, norm, target)
    cap = capture_build_log()
    try:
        with cap:
            t0 = time.time()
            rc = rk.build(do_quantization=False)
            build_s = time.time() - t0
        log = cap.read()
    finally:
        cap.cleanup()
        drop_optimizer_dumps()
    if rc != 0:
        raise RuntimeError(f"build failed for {onnx}")
    floor = min_log_bytes or log_floor(stage)
    if len(log) < floor:
        raise RuntimeError(
            f"{onnx}: build log is {len(log)} bytes, below the {floor} floor for "
            f"a verbose {stage} build - the capture or the construction verbosity "
            f"regressed and the fallback count would be a false clean")
    lines, ops, fb = count_fallbacks(log)
    rk.release()
    return {"log_bytes": len(log), "build_s": round(build_s, 1),
            "fallback_nodes": fb["real"], "fallback_ops": ops,
            "unassigned_target_lines": fb["ambiguous"],
            "n_will_fallback": fb["n_will_fallback"],
            "fallback_messages": sorted(
                set(l.split("] ", 1)[-1][:110] for l in lines
                    if "fallback" in l.lower()))[:6]}


def export_and_size(onnx, out_dir, norm, target, stage, min_log_bytes=None):
    """Build, export, and return the .rknn size.  Separate from the timing run so
    the size is measured on a graph that was actually written out."""
    rk = make_rknn(onnx, norm, target)
    cap = capture_build_log()
    try:
        with cap:
            rc = rk.build(do_quantization=False)
        log = cap.read()
    finally:
        cap.cleanup()
        drop_optimizer_dumps()
    if rc != 0:
        raise RuntimeError(f"build failed for {onnx}")
    floor = min_log_bytes or log_floor(stage)
    if len(log) < floor:
        raise RuntimeError(f"{onnx}: build log {len(log)} bytes < floor {floor}")
    out = Path(out_dir) / (Path(onnx).stem + "_fp.rknn")
    if rk.export_rknn(str(out)) != 0:
        raise RuntimeError(f"export failed for {out}")
    rk.release()
    return out, os.path.getsize(out) / 1e6


def time_simulator(onnx, norm, target, stage, iters, warmup=1):
    """Time the PC simulator.  Returns ms/inference.

    Stage 1 is PER-IMAGE (`(1,512,512,3)`), so its number is the cost of ONE
    extraction and a pair costs two of them; stage 2 consumes the pair's four
    tensors in a single call.  Both are reported as given, and the pair-level
    figure is derived rather than measured - a per-image model cannot be timed as a
    pair without also paying the dispatch overhead twice, which is exactly the
    trade-off the per-image design makes.

    Explicitly NOT a latency number - see the module docstring.  Kept because the
    RATIO between two variants on the same host is meaningful when the only
    difference is the op mix under test.
    """
    rk = make_rknn(onnx, norm, target)
    cap = capture_build_log()
    try:
        with cap:
            rk.build(do_quantization=False)
    finally:
        cap.cleanup()
        drop_optimizer_dumps()
    rk.init_runtime(target=None)

    if stage == "alike":
        inp = np.zeros((1, 512, 512, 3), dtype=np.uint8)
        call = lambda: rk.inference(inputs=[inp], data_format="nhwc")  # noqa: E731
    else:
        k = np.zeros((1, 512, 2), np.float32)
        d = np.zeros((1, 512, 128), np.float32)
        call = lambda: rk.inference(inputs=[k, k.copy(), d, d.copy()])  # noqa: E731

    for _ in range(warmup):
        call()
    t0 = time.time()
    for _ in range(iters):
        call()
    dt = (time.time() - t0) / iters * 1000.0
    rk.release()
    return dt


def graph_stats(onnx_path):
    """Node and op counts straight from the ONNX, which is exact.

    The parameter is deliberately NOT called `onnx`: doing so shadows the module
    of the same name inside this function, and the failure is a
    `FileNotFoundError` naming a Python file as the model - which reads like a
    missing graph rather than a shadowing bug.
    """
    import collections

    import onnx as onnx_mod
    m = onnx_mod.load(str(onnx_path))
    c = collections.Counter(n.op_type for n in m.graph.node)
    return {"nodes": len(m.graph.node),
            "einsum": c.get("Einsum", 0), "matmul": c.get("MatMul", 0),
            "transpose": c.get("Transpose", 0),
            "scatter": c.get("ScatterElements", 0) + c.get("ScatterND", 0),
            "softmax": c.get("Softmax", 0) + c.get("LogSoftmax", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(REPO / "weights/original/alike_stage_bil.onnx"))
    ap.add_argument("--stage2", nargs="*", default=[
        str(REPO / "weights/original/lightglue_stage_einsum.onnx"),
        str(REPO / "weights/optimized/lightglue_stage_d7.onnx")])
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--min-log-bytes", type=int, default=None,
                    help="override the per-stage regression guard on the captured "
                         "build log size.  The default is scaled by stage because a "
                         "verbose matcher build is ~1.2 MB while stage 1 is ~170 kB, "
                         "so one number cannot guard both")
    ap.add_argument("--out-dir", default=str(REPO / "models"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("NPU RESIDENCY + SIZE  (exact counts; no latency claims)")
    print("=" * 78)
    report = {"target": args.target, "sim_iters": args.iters,
              "stage1": None, "stage2": []}

    # ---- stage 1 ---------------------------------------------------------- #
    if args.stage1 and Path(args.stage1).is_file():
        print(f"\n--- stage 1: {Path(args.stage1).name} (ALIKE backbone + DKD)")
        a = build_and_audit(args.stage1, True, args.target, "alike",
                            args.min_log_bytes)
        g = graph_stats(args.stage1)
        a["graph"] = g
        size_mb = None
        if not args.skip_export:
            p, size_mb = export_and_size(args.stage1, args.out_dir, True,
                                         args.target, "alike",
                                         args.min_log_bytes)
            a["rknn"] = str(p.relative_to(REPO))
            a["rknn_MB"] = round(size_mb, 2)
        if not args.skip_timing:
            a["sim_ms"] = round(time_simulator(args.stage1, True, args.target,
                                               "alike", args.iters), 1)
            a["sim_ms_per_pair"] = round(2 * a["sim_ms"], 1)
        print(f"    log {a['log_bytes']:,} bytes | fallback nodes "
              f"{a['fallback_nodes']} {a['fallback_ops'] or ''}")
        print(f"    graph {g['nodes']} nodes | size "
              f"{a.get('rknn_MB', float('nan'))} MB"
              + (f" | sim {a['sim_ms']} ms per IMAGE "
                 f"({a['sim_ms_per_pair']} ms per pair, two calls)"
                 if "sim_ms" in a else ""))
        report["stage1"] = a

    # ---- stage 2 ---------------------------------------------------------- #
    for onnx in args.stage2:
        if not Path(onnx).is_file():
            continue
        print(f"\n--- stage 2: {Path(onnx).name} (LightGlue matcher)")
        a = build_and_audit(onnx, False, args.target, "lightglue",
                            args.min_log_bytes)
        g = graph_stats(onnx)
        a["graph"] = g
        a["onnx"] = str(Path(onnx).relative_to(REPO))
        if not args.skip_export:
            p, size_mb = export_and_size(onnx, args.out_dir, False,
                                         args.target, "lightglue",
                                         args.min_log_bytes)
            a["rknn"] = str(p.relative_to(REPO))
            a["rknn_MB"] = round(size_mb, 2)
        if not args.skip_timing:
            a["sim_ms"] = round(time_simulator(onnx, False, args.target,
                                               "lightglue", args.iters), 1)
        print(f"    log {a['log_bytes']:,} bytes | fallback nodes "
              f"{a['fallback_nodes']} {a['fallback_ops'] or ''}")
        if a["n_will_fallback"]:
            print(f"    !! {a['n_will_fallback']} 'will fallback to CPU' warning(s):")
            for msg in a["fallback_messages"][:3]:
                print(f"       {msg}")
        print(f"    graph {g['nodes']} nodes: Einsum {g['einsum']}, "
              f"MatMul {g['matmul']}, Transpose {g['transpose']}")
        print(f"    size {a.get('rknn_MB', float('nan'))} MB"
              + (f" | sim {a['sim_ms']} ms" if "sim_ms" in a else ""))
        report["stage2"].append(a)

    # ---- the comparison the whole exercise is for ------------------------- #
    if len(report["stage2"]) >= 2:
        base, new = report["stage2"][0], report["stage2"][-1]
        print("\n" + "=" * 78)
        print(f"{Path(base['onnx']).name}  ->  {Path(new['onnx']).name}")
        print("=" * 78)
        d_fb = base["fallback_nodes"] - new["fallback_nodes"]
        print(f"  NPU-fallback nodes : {base['fallback_nodes']} -> "
              f"{new['fallback_nodes']}   ({d_fb:+d})")
        print(f"  Einsum nodes       : {base['graph']['einsum']} -> "
              f"{new['graph']['einsum']}")
        print(f"  Transpose nodes    : {base['graph']['transpose']} -> "
              f"{new['graph']['transpose']}")
        print(f"  model size         : {base.get('rknn_MB')} -> "
              f"{new.get('rknn_MB')} MB")
        if "sim_ms" in base and "sim_ms" in new and base["sim_ms"]:
            r = new["sim_ms"] / base["sim_ms"]
            print(f"  simulator time     : {base['sim_ms']} -> {new['sim_ms']} ms "
                  f"({r:.2f}x)  [host-emulated, ratio only - not a latency claim]")
        report["comparison"] = {"from": base["onnx"], "to": new["onnx"],
                                "fallback_delta": d_fb,
                                "einsum_delta": new["graph"]["einsum"] - base["graph"]["einsum"],
                                "transpose_delta": new["graph"]["transpose"] - base["graph"]["transpose"]}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n[bench] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
