#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONNX -> RKNN conversion with a three-layer operator audit.

Run in the `rknn` environment (rknn-toolkit2 on an x86 host; the target board is
not needed for conversion or for numeric validation).

Why three checks and not one
----------------------------
A single "did it convert" signal is not trustworthy:

  1. LOAD  - `rknn.load_onnx()` parses the graph and hard-fails on operators it
             cannot map at all.  Necessary, not sufficient.
  2. BUILD - operators the NPU cannot execute are reported as "not support" and
             silently scheduled on the CPU.  The model still builds and still
             runs; it is just slow, and on a board without the CPU fallback
             runtime it fails at inference.  Every such line is captured and
             surfaced, because this is the failure mode that matters here -
             the entire point of the round/clip/gather rewrite was to avoid it.
  3. NUMERIC - the PC simulator is run against the torch reference.  An operator
             that is "supported" but lowered incorrectly shows up here.

Static shapes are mandatory
---------------------------
RKNN requires fixed dimensions.  Both stages are exported static: stage 1 as
`(1,3,S,S)` - one image per call - and stage 2 as four `(1,K,...)` tensors, so
`--size` here must match the export and is only used to shape the dummy input for
the normalisation config.

Input conventions
-----------------
stage 1  uint8 NHWC, normalised on the NPU (`mean=0, std=255`), so the model sees
         exactly `image_u8 / 255` - bit-identical to the float input the torch
         reference used.  No float preprocessing is needed in C++.
stage 2  float32 inputs, no normalisation (`mean=0, std=1`); the descriptors are
         already L2-normalised by stage 1.
`--dtype i8` additionally quantises (int8).  That path is reported separately and
is expected to lose accuracy; it is never the default.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Lines the RKNN build log uses to announce an operator it will not run on the NPU.
# "No lowering found ... use CustomOperatorLower instead" is the message that
# actually matters, and an earlier revision of this script missed it entirely: a
# graph with boolean `Or` nodes converted "successfully" while every one of those
# nodes was handed to a custom operator, and the simulator then disagreed with
# the reference by 484 px.  A fallback detector that does not recognise this line
# is worse than none, because it reports success.
FALLBACK_PATTERNS = re.compile(
    r"(?im)^.*(not support|unsupport|fallback|fall back|using cpu|"
    r"cpu op|cpu_?op|no lowering found|custom ?operator|unknown op target|"
    r"unkown op target).*$")


class capture_build_log:
    """Capture the RKNN build log, which does NOT go through Python's stdout.

    THE VERBOSITY IS NOT JUST COSMETIC - IT DECIDES WHAT THE AUDIT CAN SEE
    ---------------------------------------------------------------------
    Two independent things have to be right for this audit to mean anything, and
    both were wrong at some point in this project.

    1. THE STREAM.  The toolkit writes its log from C++ to file descriptor 1:

       * `redirect_stdout` replaces `sys.stdout`, a Python object.  It cannot
         intercept a write made by native code.  Measured: 0 bytes captured while
         the build emitted 1.2 MB - so the audit was reading an empty string and
         reporting a clean graph.
       * `contextlib.redirect_stderr` fails the same way.
       * The fd itself must be redirected, to a temporary file, for the duration
         of the call.  The file is read back afterwards and the original fd
         restored in a `finally` so an exception cannot leave stdout pointing at
         a deleted file.
       * fd 2 is folded into the same file as well: the toolkit splits severity
         levels across both streams, and reading only one is the same bug wearing
         a different costume.

    2. THE LOG LEVEL, AND WHEN IT IS SET.  Measured on this exact graph, same
       toolkit, three ways of asking for the log:

           RKNN(verbose=False)                        480,129 bytes    0 warnings
           RKNN(verbose=False) + later set_log_level  866,686 bytes    0 warnings
           RKNN(verbose=True)                       1,216,961 bytes   36 warnings

       Only the third reports the 36 `Transpose will fallback to CPU` nodes.  The
       other two are not merely quieter - they OMIT the warnings, so an audit
       reading them sees a graph with no fallbacks.  The earlier version of this
       script built quietly and therefore reported a clean graph that had 36 nodes
       on the host.

       The post-hoc setter is the trap: it visibly raises the level (the log grows
       from 480 kB to 867 kB) while still not emitting the warnings.  `verbose` has
       to reach the CONSTRUCTOR, because `RKNN.__init__` is where
       `set_log_level_and_file_path` is called.  So the callers here construct with
       `verbose=True`, and `--show-all` no longer exists - there is no reason to
       build quietly, since fd 1/2 are diverted to a file and the 1.2 MB of
       progress bars never reaches a terminal.

    Note the log is mostly progress bars; only the interesting lines are extracted
    from it, and only those are printed unless `--verbose`.
    """

    def __init__(self):
        self.path = None
        self._saved = None

    def __enter__(self):
        fd, self.path = tempfile.mkstemp(prefix="rknn_build_", suffix=".log")
        os.close(fd)
        sys.stdout.flush()
        self._saved = (os.dup(1), os.dup(2))
        target = os.open(self.path, os.O_WRONLY | os.O_TRUNC)
        os.dup2(target, 1)
        os.dup2(target, 2)
        os.close(target)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._saved[0])
        os.close(self._saved[1])
        self._saved = None
        return False

    def read(self):
        try:
            # `errors="replace"` because the log interleaves progress-bar bytes
            # with text; a strict decode would raise partway through and hand back
            # a TRUNCATED log, which reads as "no fallbacks found".
            return Path(self.path).read_text(errors="replace")
        except OSError:
            return ""

    def cleanup(self):
        if self.path and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass


def drop_optimizer_dumps():
    """Delete the toolkit's graph-optimiser scratch files from the CWD.

    `rknn.build()` writes `check<N>_<stage>.onnx` (base_optimize, fold_constant,
    correct_ops, fuse_ops) into the process CWD through its ONNX optimiser.  They
    are 35-75 MB each and nothing ever reads them back, so four conversions leave
    ~200 MB of dead weights next to the shipped models - which is exactly the kind
    of artefact that later gets mistaken for an intermediate of the pipeline.

    The names are hard-coded in a compiled `.so`, so there is no config to turn the
    dump off; removing the files is the only lever.  Called after every build.
    """
    removed = []
    for path in sorted(Path.cwd().glob("check*_*.onnx")):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    if removed:
        print(f"[rknn] removed {len(removed)} optimizer dump(s): {', '.join(removed)}")
    return removed


def parse_norm(spec):
    mean, std = spec.split(":")
    return ([float(v) for v in mean.split(",")],
            [float(v) for v in std.split(",")])


def build_and_check(args):
    from rknn.api import RKNN

    print(f"[rknn] toolchain target={args.target} dtype={args.dtype} "
          f"norm={args.norm}")
    # `verbose=True` IS MANDATORY, NOT A PREFERENCE.  The warnings this script
    # audits are emitted by the toolkit's verbose path; constructing quietly
    # suppresses them and the audit then reports a clean graph for one with 36
    # nodes on the host (measured: 480 kB / 0 warnings quiet against
    # 1,217 kB / 36 warnings verbose).  Calling a log-level setter after
    # construction does NOT substitute - it grows the log to 867 kB and still
    # omits the warnings.
    #
    # This does not make the console noisy: the build's fd 1/2 are diverted into a
    # temp file by `capture_build_log`, so the 1.2 MB of progress bars is never
    # printed.  `--verbose` decides only whether the captured log is echoed.
    rknn = RKNN(verbose=True)

    # `mean_values`/`std_values` are applied PER INPUT CHANNEL, so they only make
    # sense for the image stage.  Passing them for the matcher fails outright
    # (`The len of mean_values ([0,0,0]) for input 0 is wrong, expect 512!`,
    # because RKNN reads input 0 as a 512-"channel" tensor) - the matcher's inputs
    # are keypoints and already-normalised descriptors and must be left alone.
    # The toolkit rejects `core_mask` (this version's valid keys are listed in the
    # error it prints, and `single_core_mode` has superseded it), so the core
    # selection is expressed that way.  `single_core_mode=True` pins the graph to
    # one NPU core: broadcaster costs a per-op sync, so for SMALL tensors one core
    # can be faster than three even though it has a third of the throughput.  The
    # matcher runs 512x512 attention - small enough that this is worth measuring
    # rather than assuming, which is why it is a flag.
    cfg = {"target_platform": args.target, "float_dtype": args.float_dtype}
    if args.single_core:
        cfg["single_core_mode"] = True
        print("[rknn] single_core_mode=True (one NPU core, no broadcast sync)")
    if args.flash_attention:
        cfg["enable_flash_attention"] = True
        print("[rknn] enable_flash_attention=True")
    if args.optimization_level is not None:
        cfg["optimization_level"] = args.optimization_level
        print(f"[rknn] optimization_level={args.optimization_level}")
    if args.norm != "none":
        mean, std = parse_norm(args.norm)
        cfg["mean_values"] = [mean]
        cfg["std_values"] = [std]
    ret = rknn.config(**cfg)
    if ret != 0:
        raise RuntimeError("rknn.config failed")

    print(f"[rknn] [1/3] load_onnx {args.onnx}")
    if rknn.load_onnx(model=args.onnx) != 0:
        raise RuntimeError(
            "load_onnx FAILED - the graph contains an operator the toolkit "
            "cannot parse or map at all.  Check the log above for the op name.")

    print(f"[rknn] [2/3] build(do_quantization={args.dtype == 'i8'})")
    # fd-level capture - see `capture_build_log`.  The object was constructed with
    # `verbose=True` above, which is the OTHER half of the same requirement:
    # `redirect_stdout` cannot see a native write (0 bytes captured while the
    # build emitted 1.2 MB), and a quiet construction suppresses the warnings.
    # Either mistake yields a confident "no operator falls back".
    cap = capture_build_log()
    try:
        with cap:
            ret = rknn.build(do_quantization=(args.dtype == "i8"),
                             dataset=args.dataset)
        log = cap.read()
    finally:
        cap.cleanup()
    drop_optimizer_dumps()
    # The floor is a regression guard for exactly the two failures above, both of
    # which have happened: a quiet build is far smaller than a verbose one, and a
    # broken capture is ~0 bytes.  The threshold is scaled per stage because the
    # two graphs are very different sizes - the matcher is ~1.2 MB verbose against
    # ~480 kB quiet, while stage 1 is ~170 kB verbose.  A single number would
    # either miss the quiet path on stage 1 or reject every valid stage-1 build.
    floor = args.min_log_bytes
    if floor is None:
        floor = 600_000 if args.stage == "lightglue" else 80_000
    if len(log) < floor:
        raise RuntimeError(
            f"the build log is only {len(log)} bytes, below the {floor} floor for "
            f"a verbose {args.stage} build - the capture or the construction "
            f"verbosity is wrong, and the fallback audit would report a false "
            f"clean result.  Do not trust any 'no operator falls back' line from "
            f"this run.")
    # The build log is ~800 kB of progress bars, so only the interesting lines are
    # summarised; `--verbose` echoes the lot, which is now safe because stdout is
    # unrestricted again by the time this runs.
    if args.verbose:
        sys.stdout.write(log)
    lines, ops, fb = count_fallbacks(log)
    print(f"[rknn] audit: {len(log)} bytes of build log examined; "
          f"no_lowering={fb['n_no_lowering']} "
          f"to_cpu={fb['n_to_cpu']} will_fallback={fb['n_will_fallback']} "
          f"unassigned_target={fb['ambiguous']}")
    if ops:
        print("[rknn] !! CPU fallback - these node(s) are NOT on the NPU:")
        for op, n in sorted(ops.items(), key=lambda kv: -kv[1]):
            print(f"            {n:4d}  {op}")
    elif fb["real"]:
        print(f"[rknn] !! CPU fallback - {fb['real']} node(s) reported "
              f"({fb['n_no_lowering']} 'no lowering', {fb['n_to_cpu']} 'to CPU', "
              f"{fb['n_will_fallback']} 'will fallback to CPU')")
    if fb["ambiguous"]:
        # Reported, but explicitly labelled as not-a-failure.  See the docstring:
        # target 0 is NPU_CORE_AUTO, so this fires for healthy NPU nodes too.
        print(f"[rknn]    note: {fb['ambiguous']} 'Unkown op target' line(s) - "
              f"target 0 is NPU_CORE_AUTO, so these are NOT fallbacks")
    if lines and args.verbose:
        print(f"[rknn]    distinct matched log lines ({len(lines)}):")
        for ln in lines:
            print("            " + ln)
    # The warnings are printed here rather than only counted, because the whole
    # point of this audit is that a single summary number can be wrong while every
    # individual line is available to be read.  A count without the message is
    # how "36 Transpose nodes moved to the CPU" stayed invisible for a whole pass.
    if fb["real"] and not args.verbose:
        shown = [l for l in lines if "fallback" in l.lower()][:5]
        for ln in shown:
            print("            " + ln)
        if len(shown) < len(lines):
            print(f"            ... and {len(lines) - len(shown)} more "
                  f"(use --verbose to see all)")
    else:
        print("[rknn] no operator falls back off the NPU in this graph")
    if ret != 0:
        raise RuntimeError("rknn.build FAILED (see log above)")

    out = args.out or os.path.splitext(args.onnx)[0] + f"_{args.dtype}.rknn"
    if rknn.export_rknn(out) != 0:
        raise RuntimeError("export_rknn failed")
    print(f"[rknn] wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
    return rknn, out, {"lines": lines, "ops": ops, **fb}


#: The messages that PROVE a node left the NPU.  Everything else is suspicious but
#: not conclusive, and the distinction matters: a check that cannot tell "this node
#: runs on the CPU" from "the compiler mentioned the CPU" is a check that cries
#: wolf, and a check nobody trusts gets ignored.
#:
#: `will fallback to CPU` is the one that fires on THIS graph and it is a REAL
#: fallback - `Transpose will fallback to CPU, because input shape has exceeded
#: the max limit, height(512) * width(64) = 32768, required product no larger than
#: 8192`.  The matcher transposes (B,H,N0,N1) tensors, and 512*512 = 262144 is 32x
#: over the NPU's transpose limit, so every one of those moves to the host.  It is
#: a warning rather than an error and it does not stop the build, which is exactly
#: why a detector that only looks for hard failures misses it.
REAL_FALLBACK = re.compile(
    r"(?i)(no lowering found|use CustomOperatorLower|"
    r"will fallback to CPU|fallback to CPU|"
    r"turn to Target:CPU|not support.*NPU|LayerNorm: Shape not support)")

#: Messages that correlate with a fallback but do not state one.  `Unkown op
#: target: 0` is the trap: `RKNN_NPU_CORE_AUTO` is 0, so the message fires for
#: every node whose core mask is AUTO - which is every NPU node.  It is logged on
#: the `E` channel with the word "Unkown" in it and reads exactly like a problem.
#: It is counted and reported separately, and it is NOT a failure.
AMBIGUOUS = re.compile(r"(?i)unkown op target")


def count_fallbacks(log):
    """Classify the build log into proven fallbacks and log noise.

    WHY THIS IS NOT A ONE-LINE GREP
    -------------------------------
    Three separate ways this went wrong in this project, in the order they were
    discovered:

    1. The pattern list had no `node type = <Op>` alternative.  The RKNN loader
       refused a graph with a Python traceback and the audit printed nothing -
       a false clean result.
    2. The unique-LINE set has one entry per op TYPE, so it bounded the report at
       "2" for a graph with 29 offending nodes.  Counting NODES is what decides
       whether a rewrite is worth doing.
    3. `Unkown op target: 0` was treated as a fallback.  It is not: target 0 is
       `RKNN_NPU_CORE_AUTO`, so the line is emitted for every AUTO-masked node
       including the ones that are happily on the NPU.  Treating it as a failure
       reported 39 fallbacks for a stage-1 graph that has none, which would have
       sent the whole optimisation pass chasing a phantom.

    So the return value separates them, and `real` is the only number that may
    drive a decision.
    """
    lines = sorted(set(m.group(0).strip()
                       for m in FALLBACK_PATTERNS.finditer(log)))
    # Proven: the compiler said a node has no lowering, or named an op it is
    # redirecting to the CPU.  This is the decision-making number.
    ops = {}
    for m in re.finditer(
            r"node type = ([A-Za-z_][A-Za-z_0-9]*), use CustomOperatorLower", log):
        ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    n_no_lowering = len(re.findall(r"(?i)no lowering found", log))
    n_to_cpu = len(re.findall(r"(?i)turn to Target:CPU", log))
    # `Transpose will fallback to CPU` is per-NODE.  Its op name is not on the
    # line, so it is counted rather than attributed.
    n_will_fallback = len(re.findall(r"(?i)will fallback to CPU", log))
    # THE TWO CATEGORIES ARE DISJOINT, SO THEY SUM - BUT `ops` IS ONE OF THEM.
    #
    # `ops` is parsed out of the SAME `No lowering found` lines that
    # `n_no_lowering` counts, so it is a breakdown of that count, not a third
    # category: adding both double-counts (measured, 29 became 58).
    #
    # The categories really are disjoint - the `No lowering found` lines name
    # Einsum and ScatterElements nodes, while the `will fallback to CPU` lines are
    # all Transpose - so the total is their SUM, not their max.  An earlier
    # version took the max on the reasoning that they might be two views of the
    # same nodes; on the pre-rewrite matcher that reported 36 where the truth is
    # 29 + 36 = 65, which is the difference between "36 nodes to fix" and "65".
    if sum(ops.values()) > n_no_lowering:
        # Cannot happen with the current patterns, but if it ever does, the
        # breakdown is describing something the total does not, and a silent
        # mismatch here is how this audit went wrong before.
        raise AssertionError(
            f"op breakdown ({sum(ops.values())} nodes) exceeds the no-lowering "
            f"count ({n_no_lowering}); the patterns disagree")
    real = n_no_lowering + n_to_cpu + n_will_fallback
    # Not proven: mention the CPU / an unknown target without stating a fallback.
    noisy = len(AMBIGUOUS.findall(log))
    return lines, ops, {"real": real, "ambiguous": noisy,
                        "n_no_lowering": n_no_lowering, "n_to_cpu": n_to_cpu,
                        "n_will_fallback": n_will_fallback}


def _assign_by_shape(outs, z, names):
    """Map simulator outputs to names by shape.

    The simulator does not guarantee the ONNX output order, so identifying by
    shape is the only safe route.  Ambiguity is an error rather than a guess.
    """
    picked, remaining = [], list(outs)
    for n in names:
        want = z[n].shape
        cand = [o for o in remaining if o.shape == want]
        if not cand:
            # tolerate a simulator that drops/keeps a singleton dim
            cand = [o for o in remaining
                    if o.ndim == len(want) and o.shape[-1] == want[-1]]
        if not cand:
            picked.append(None)
            continue
        picked.append(cand[0])
        remaining.remove(cand[0])
    return picked


def _match_order(sim_kpts, ref_kpts):
    """For each simulated keypoint, the index of the nearest reference one.

    MANDATORY, not a nicety.  `topk` returns its selection in score order, and
    the ORDER is backend-specific: the simulated model and the torch reference
    hold the same keypoints in a different sequence.  Comparing element by
    element reports a 484 px error on outputs whose value ranges agree to three
    decimals - a completely false alarm that cost real time here.  Every
    downstream tensor is permuted with this index before being compared.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_kpts)
    dist, idx = tree.query(sim_kpts, k=1)
    return idx, dist


def numeric_check(rknn, args):
    """Compare the PC simulator against the torch reference dump."""
    if not args.ref or not os.path.isfile(args.ref):
        print("[rknn] [3/3] skipped: no --ref dump")
        return {}

    z = np.load(args.ref)
    rknn.init_runtime(target=None)          # None = PC simulator

    if args.stage == "alike":
        u8 = z["image_u8"]                                     # (B,3,S,S)
        inp = np.transpose(u8, (0, 2, 3, 1)).astype(np.uint8)   # -> NHWC
        outs = rknn.inference(inputs=[inp], data_format="nhwc")
        names = ["keypoints", "descriptors", "scores"]
    else:
        ins = [z["keypoints0"], z["keypoints1"], z["descriptors0"],
               z["descriptors1"]]
        outs = rknn.inference(inputs=[i.astype(np.float32) for i in ins])
        names = ["matches0", "mscores0"]

    outs = _assign_by_shape(outs, z, names)
    report = {}
    print(f"[rknn] [3/3] simulator vs torch reference ({args.ref})")

    if args.stage == "alike":
        k_sim = outs[0].astype(np.float64)
        k_ref = z["keypoints"].astype(np.float64)
        for b in range(k_ref.shape[0]):
            idx, dist = _match_order(k_sim[b], k_ref[b])
            print(f"    view{b} keypoints  max NN dist {dist.max():.4f} px   "
                  f"mean {dist.mean():.4f}   >1px: {int((dist > 1).sum())}/"
                  f"{k_ref.shape[1]}")
            report.setdefault("keypoints", {})[f"view{b}"] = {
                "max_nn_px": float(dist.max()), "mean_nn_px": float(dist.mean()),
                "n_over_1px": int((dist > 1).sum())}
        # Descriptors and scores are compared at the ORDER-MATCHED keypoints.
        # This is what makes the number meaningful: `topk` returns its selection
        # in score order and that order is backend-specific, so `d_sim[i]` and
        # `d_ref[i]` are descriptors of two DIFFERENT points.  An unmatched
        # comparison reported a median cosine of 0.88 here for a model whose
        # sampling operator is in fact exact; the mistake is cheap to make and
        # looks like a conversion failure.
        d_sim, s_sim = outs[1].astype(np.float64), outs[2].astype(np.float64)
        d_ref, s_ref = z["descriptors"].astype(np.float64), z["scores"].astype(np.float64)
        per = {"max_nn_px": [], "cos_med": [], "cos_p1": [], "cos_min": [],
               "score_max": [], "score_mean": []}
        for b in range(k_ref.shape[0]):
            idx, dist = _match_order(k_sim[b], k_ref[b])
            d_r, s_r = d_ref[b][idx], s_ref[b][idx]
            cos = ((d_sim[b] * d_r).sum(-1) /
                   (np.linalg.norm(d_sim[b], axis=-1) *
                    np.linalg.norm(d_r, axis=-1) + 1e-12))
            ds = np.abs(s_sim[b] - s_r)
            k1 = max(1, int(0.01 * cos.size))
            per["max_nn_px"].append(float(dist.max()))
            per["cos_med"].append(float(np.median(cos)))
            per["cos_p1"].append(float(np.sort(cos)[k1 - 1]))
            per["cos_min"].append(float(cos.min()))
            per["score_max"].append(float(ds.max()))
            per["score_mean"].append(float(ds.mean()))
            print(f"    view{b} descriptors median cos {np.median(cos):.7f}  "
                  f"p1 {np.sort(cos)[k1-1]:.6f}  min {cos.min():.6f}  "
                  f"order-matched")
            print(f"    view{b} scores      max|d| {ds.max():.3e}  "
                  f"mean {ds.mean():.3e}")
        report["descriptors"] = {
            "median_cos": float(np.mean(per["cos_med"])),
            "p1_cos": float(np.mean(per["cos_p1"])),
            "min_cos": float(min(per["cos_min"])),
            "note": "order-matched by keypoint nearest neighbour"}
        report["scores"] = {"max_abs": float(max(per["score_max"])),
                            "mean_abs": float(np.mean(per["score_mean"]))}
        return report

    # stage 2: matches are an index array; compare exactly
    m_sim, ms_sim = outs[0], outs[1].astype(np.float64)
    m_ref, ms_ref = z["matches0"].astype(np.int64), z["mscores0"].astype(np.float64)
    m_sim_i = m_sim.astype(np.int64)
    valid_sim, valid_ref = int((m_sim_i >= 0).sum()), int((m_ref >= 0).sum())
    identical = int((m_sim_i == m_ref).sum())
    print(f"    matches     sim {valid_sim} valid, ref {valid_ref} valid, "
          f"identical {identical}/{m_ref.size}")
    print(f"    mscores     max|d| {np.abs(ms_sim - ms_ref).max():.3e}")
    report["matches0"] = {"valid_sim": valid_sim, "valid_ref": valid_ref,
                          "identical": identical, "total": int(m_ref.size)}
    report["mscores0"] = {"max_abs": float(np.abs(ms_sim - ms_ref).max())}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--stage", required=True, choices=["alike", "lightglue"])
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--dtype", default="fp", choices=["fp", "i8"],
                    help="fp = float16 (the lossless path); i8 = int8 quantised")
    ap.add_argument("--dataset", default=None,
                    help="calibration list for i8 (one image path per line)")
    ap.add_argument("--norm", default="0,0,0:255,255,255",
                    help="mean:std applied by the NPU; stage 2 uses 0:1")
    ap.add_argument("--float-dtype", default="float16")
    ap.add_argument("--ref", default=None, help="torch reference .npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    ap.add_argument("--verbose", action="store_true",
                    help="echo the captured build log and every matched line.  "
                         "NOTE the build is ALWAYS captured verbosely - the "
                         "fallback warnings only exist on that path, so auditing "
                         "a quiet build reports a false clean.  This flag only "
                         "controls whether the log is printed.")
    ap.add_argument("--min-log-bytes", type=int, default=None,
                    help="override the regression guard on the captured log size "
                         "(default: scaled to the stage, since stage 1 emits far "
                         "less than the matcher)")
    ap.add_argument("--single-core", action="store_true",
                    help="build for ONE NPU core instead of all three.  "
                         "Broadcasting to the cores costs a per-op sync, so small "
                         "tensors can be faster on a single core despite having a "
                         "third of the throughput - worth measuring, not assuming.")
    ap.add_argument("--flash-attention", action="store_true",
                    help="hand attention to the toolkit's flash kernel.  The "
                         "rewritten MatMul form is what makes this applicable; "
                         "verify accuracy before using it, since it is a different "
                         "kernel and not bit-identical.")
    ap.add_argument("--optimization-level", type=int, default=None,
                    choices=[0, 1, 2, 3],
                    help="toolkit graph optimisation level; 3 is the default most "
                         "builds use, 0 disables it for debugging")
    args = ap.parse_args()

    if args.dtype == "i8" and not args.dataset:
        ap.error("--dtype i8 needs --dataset (a calibration image list)")

    rknn, out, fallbacks = build_and_check(args)
    numeric = numeric_check(rknn, args)
    rknn.release()

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({"onnx": args.onnx, "rknn": out, "stage": args.stage,
                       "target": args.target, "dtype": args.dtype,
                       "cpu_fallback_lines": fallbacks["lines"],
                       "cpu_fallback_ops": fallbacks["ops"],
                       "cpu_fallback_nodes": fallbacks["real"],
                       "cpu_fallback_ambiguous": fallbacks["ambiguous"],
                       "numeric": numeric}, f, indent=2)
        print(f"[rknn] report -> {args.report}")

    if fallbacks["real"]:
        print(f"\n[rknn] RESULT: converted WITH {fallbacks['real']} node(s) falling "
              f"back off the NPU - this model would not stay fully on the NPU")
        return 2
    print("\n[rknn] RESULT: converted, no operator falls back off the NPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
