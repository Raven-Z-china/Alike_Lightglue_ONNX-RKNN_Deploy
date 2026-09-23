#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert an exported ONNX graph to a TensorRT engine, and check it numerically.

Same three layers as `convert_to_rknn.py` - load, audit, compare against the torch
reference dump - because a conversion that is not verified against the traceable
reference is not evidence.  The comparisons reuse that module's helpers on purpose
(`capture_build_log`, `_match_order`), so the TRT report carries the SAME field
names as the RKNN one and the two can be read side by side.

What is different about TensorRT, and why it is not just "RKNN with another flag":

* NO CPU FALLBACK.  RKNN can hand an unsupported operator to the host, which is a
  residency problem you can count (see `scripts/build_log_probe.py`).  TRT has no
  such mode: an operator it cannot build either fails the build or is absorbed into
  a plugin, and both are visible only in the builder log.  So layer 2 here is "did
  it build, and what did it say", and the log is captured from fd 1/2 - the toolkit
  writes it from C++, so Python-level handlers see nothing (the lesson that made
  `capture_build_log` divert file descriptors in the first place).
* AN ENGINE IS NOT PORTABLE.  It is compiled for one GPU architecture, one TRT
  version and one CUDA version; a `.engine` file is not a distributable model the
  way an `.onnx` is, and one built here will refuse to load on a different card.
  The report therefore records the provenance (device, capability, TRT and CUDA
  version) next to the file, and the tool prints it after every build.
* FP16 IS NOT `--precision fp16` ALONE.  TRT keeps fp32 where a layer's precision
  cannot be reduced safely, so the honest question is not "did we ask for fp16" but
  "which layers actually ended up half" - which is why the report carries the
  per-layer precision of the built engine rather than the requested flag.

Usage
    conda activate alike
    python scripts/convert_to_trt.py weights/optimized/alike_stage_gath.onnx \
        --stage alike --precision fp16 --ref refs/s1_a.npz
    python scripts/convert_to_trt.py weights/optimized/lightglue_stage_d7.onnx \
        --stage lightglue --precision fp16 --ref refs/s2_d7.npz

    # fp32 for comparison, and the baselines under weights/original/ likewise:
    ... --precision fp32 --out weights/optimized/alike_stage_gath_fp32.engine
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import paths                                                    # noqa: E402
from convert_to_rknn import capture_build_log, _match_order     # noqa: E402
from portable_path import portable                              # noqa: E402

# Lines worth counting in a builder log.  `Unsupported` and `Plugin` are the ones
# that change what the engine is; `[E]`/`[W]` are the toolkit's own severity tags.
LOG_PATTERNS = {
    "errors": "[E]",
    "warnings": "[W]",
    "unsupported": "unsupported",
    "plugins": "Plugin",
}


_LOGGER = None


def logger(verbose=False):
    """One TRT logger for the whole process.

    TRT registers a logger globally and IGNORES any later, different one for an
    existing builder/runtime ("the current new logger is ignored"), so creating a
    second one to quiet the runtime is not just noise - it is a warning that the
    level you asked for is not the level you get.
    """
    global _LOGGER
    import tensorrt as trt
    if _LOGGER is None:
        _LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    return _LOGGER


def parse_precision(tag):
    tag = tag.lower()
    if tag not in ("fp16", "fp32"):
        raise SystemExit(f"--precision must be fp16 or fp32, got {tag!r}")
    return tag


def default_out(onnx_path, precision):
    return Path(onnx_path).with_suffix("").parent / \
        f"{Path(onnx_path).stem}_{precision}.engine"


def build(args):
    """Parse, build and serialise.  Returns (engine_bytes, log, seconds).

    Everything TRT does - including its CUDA init and the parser - runs inside the
    fd capture.  The toolkit logs from C++ (the reason `capture_build_log` diverts
    descriptors in the first place) and its INFO chatter buries the tool's own
    progress, so the only things printed outside the capture are the two progress
    lines and whatever the log itself says about failures.
    """
    import tensorrt as trt

    trt_log = logger(args.verbose)
    precision_note = ("fp16 tactics, fp32 where a layer cannot be reduced"
                      if args.precision == "fp16" else "fp32 (TRT default tactics)")
    print(f"[trt] [1/3] parse {args.onnx}")
    print(f"[trt] [2/3] build ({precision_note}, workspace "
          f"{args.workspace_mb} MB) - the tactic search takes minutes on the matcher")

    cap = capture_build_log()
    t0 = time.perf_counter()
    parsed = False
    failure = None
    serialized = None
    try:
        with cap:
            builder = trt.Builder(trt_log)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_log)
            parsed = bool(parser.parse_from_file(str(args.onnx)))
            if not parsed:
                failure = "parse"
            else:
                # Static shapes are a requirement of the whole deployment, not a
                # preference: TRT would accept a dynamic axis and then need an
                # optimisation profile, which no caller here builds.
                for i in range(network.num_inputs):
                    dims = tuple(network.get_input(i).shape)
                    if any(d < 0 for d in dims):
                        raise ValueError(
                            f"input {network.get_input(i).name} is dynamic {dims}; "
                            f"RKNN could not take it either, and this tool builds "
                            f"no optimisation profile")

                # The method was renamed between TRT 8 and 10 and both spellings
                # ship in different builds of the bindings, so depend on neither.
                make_config = getattr(builder, "create_config",
                                      None) or builder.create_builder_config
                config = make_config()
                if args.precision == "fp16":
                    config.set_flag(trt.BuilderFlag.FP16)
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                             args.workspace_mb << 20)
                # DETAILED is what puts layer names in the engine, for the audit.
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    failure = "build"
    finally:
        seconds = time.perf_counter() - t0
        log = cap.read()
        cap.cleanup()

    def excerpt():
        keep = [l for l in log.splitlines()
                if "[E]" in l or "error" in l.lower() or "unsupported" in l.lower()]
        return "\n".join(keep[-40:])[-4000:] or "(the log holds no error lines)"

    if failure == "parse":
        raise SystemExit(f"ONNX parse failed - the graph is not TRT-parseable "
                         f"as-is:\n{excerpt()}")
    if failure == "build":
        raise SystemExit(f"build failed after {seconds:.1f}s; the captured builder "
                         f"log is the only diagnostic TRT gives:\n{excerpt()}")
    return bytes(serialized), log, seconds


def audit(engine, log, seconds, engine_bytes):
    """What the engine is made of, and what the builder complained about."""
    import tensorrt as trt

    report = {"build_seconds": round(seconds, 1),
              "engine_MB": round(len(engine_bytes) / 1e6, 2)}
    counts = {k: log.lower().count(v.lower()) if k != "errors" else log.count(v)
              for k, v in LOG_PATTERNS.items()}
    report["log"] = {"bytes": len(log), **counts}
    if counts["errors"]:
        lines = [l.strip() for l in log.splitlines() if "[E]" in l][:5]
        report["log"]["error_lines"] = lines
        print(f"    !! builder reported {counts['errors']} error line(s):")
        for l in lines:
            print(f"       {l[:160]}")

    # What precision the engine ACTUALLY runs in.  TRT has no per-layer "precision"
    # field to read - a "layer" is a tactic, and the precision lives on each tensor
    # edge (`Format/Datatype`) and on the weights (`Type`) - so the honest audit is
    # a histogram of those, not a restatement of the flag that was requested.
    layers = {"total": 0, "tensors_by_dtype": {}, "weights_by_dtype": {},
              "by_type": {}, "plugins": 0}

    def bump(bucket, key):
        layers[bucket][key] = layers[bucket].get(key, 0) + 1

    try:
        insp = engine.create_engine_inspector()
        info = json.loads(insp.get_engine_information(
            trt.LayerInformationFormat.JSON))
        for layer in info.get("Layers", []):
            layers["total"] += 1
            kind = str(layer.get("LayerType", "?"))
            bump("by_type", kind)
            if "Plugin" in kind:
                layers["plugins"] += 1
            for edge in ("Inputs", "Outputs"):
                for t in layer.get(edge) or []:
                    bump("tensors_by_dtype", str(t.get("Format/Datatype", "?")))
            w = layer.get("Weights")
            if isinstance(w, dict) and "Type" in w:
                bump("weights_by_dtype", str(w["Type"]))
        # Fraction of the FLOATING-POINT edges that are half.  Index tensors are
        # int and can never be Half, so counting them would make a fully-fp16
        # arithmetic path look 28 % fp32; the raw histogram above is kept too, so
        # the claim can always be re-derived from the report.
        half = layers["tensors_by_dtype"].get("Half", 0)
        single = layers["tensors_by_dtype"].get("Float", 0)
        fp_edges = half + single
        layers["half_fraction_of_fp_edges"] = (round(half / fp_edges, 4)
                                               if fp_edges else None)
    except Exception as exc:                       # pragma: no cover - API drift
        layers["inspection_error"] = f"{type(exc).__name__}: {exc}"
    report["layers"] = layers
    return report


def deserialize(engine_bytes):
    """Deserialise with the runtime's chatter suppressed.

    Serialising the audit and the numeric check through one helper keeps the tool's
    output to its own lines: TRT reports "Loaded engine size: 19 MiB" at INFO, which
    is noise when the tool already prints the size it wrote.
    """
    import tensorrt as trt

    lg = logger()
    prev = getattr(lg, "min_severity", None)
    try:
        lg.min_severity = trt.Logger.ERROR
    except Exception:                              # pragma: no cover
        pass
    try:
        engine = trt.Runtime(lg).deserialize_cuda_engine(engine_bytes)
    finally:
        if prev is not None:
            try:
                lg.min_severity = prev
            except Exception:                      # pragma: no cover
                pass
    if engine is None:
        raise SystemExit("engine failed to deserialise")
    return engine


def run_engine(engine_bytes, feeds):
    """Execute the engine on `feeds` (name -> ndarray) and return its outputs.

    Tensors are allocated with torch rather than pycuda: torch is already a
    dependency here and its caching allocator and stream are what the rest of the
    project uses.  Only `data_ptr()` crosses into TRT.
    """
    import torch
    import tensorrt as trt

    engine = deserialize(engine_bytes)
    ctx = engine.create_execution_context()

    io = [(engine.get_tensor_name(i), engine.get_tensor_mode(
        engine.get_tensor_name(i))) for i in range(engine.num_io_tensors)]
    dev, outs = {}, {}
    for name, mode in io:
        if mode == trt.TensorIOMode.INPUT:
            if name not in feeds:
                raise SystemExit(f"engine wants input {name!r}, the reference dump "
                                 f"has {sorted(feeds)}")
            np_dtype = trt.nptype(engine.get_tensor_dtype(name))
            t = torch.from_numpy(
                np.ascontiguousarray(feeds[name].astype(np_dtype))).cuda()
            ctx.set_tensor_address(name, t.data_ptr())
            dev[name] = t
        else:
            shape = tuple(engine.get_tensor_shape(name))
            np_dtype = trt.nptype(engine.get_tensor_dtype(name))
            t = torch.empty(shape, dtype=torch.from_numpy(
                np.empty(0, dtype=np_dtype)).dtype, device="cuda")
            ctx.set_tensor_address(name, t.data_ptr())
            outs[name] = t

    stream = torch.cuda.Stream()
    if not ctx.execute_async_v3(stream.cuda_stream):
        raise SystemExit("execute_async_v3 returned false")
    stream.synchronize()
    return {k: v.cpu().numpy() for k, v in outs.items()}


def numeric_check(args, engine_bytes):
    """Compare the engine against the torch reference dump.

    Field names match `convert_to_rknn.numeric_check` on purpose: the same graph on
    two backends should produce two reports that can be diffed line by line.
    """
    if not args.ref or not os.path.isfile(args.ref):
        print("[trt] [3/3] skipped: no --ref dump")
        return {}

    z = np.load(args.ref)
    if args.stage == "alike":
        # `image` is the fp32 tensor the graph was TRACED with (0..1); `image_u8`
        # is its 0..255 uint8 twin, kept in the dump for the RKNN path, which
        # normalises inside the graph.  Feeding the u8 copy - or dividing this one
        # again - produces a near-black image and a verify result that looks like a
        # broken engine (37 px NN distance) rather than a wrong input.
        feeds = {"image": z["image"].astype(np.float32)}
        names = ["keypoints", "descriptors", "scores"]
    else:
        feeds = {n: z[n].astype(np.float32) for n in
                 ("keypoints0", "keypoints1", "descriptors0", "descriptors1")}
        names = ["matches0", "mscores0"]

    raw = run_engine(engine_bytes, feeds)
    print(f"[trt] [3/3] engine vs torch reference ({args.ref})")
    report = {"output_shapes": {k: list(v.shape) for k, v in raw.items()}}

    # Engine outputs are unordered by name; identify by shape, and refuse to guess.
    picked, remaining = [], list(raw.items())
    for n in names:
        want = z[n].shape
        cand = [kv for kv in remaining if kv[1].shape == want]
        if not cand:
            raise SystemExit(f"engine has no output shaped like {n} {want}; "
                             f"got {[(k, v.shape) for k, v in remaining]}")
        picked.append(cand[0])
        remaining.remove(cand[0])
    outs = [v for _k, v in picked]

    if args.stage == "alike":
        k_eng = outs[0].astype(np.float64)
        k_ref = z["keypoints"].astype(np.float64)
        for b in range(k_ref.shape[0]):
            idx, dist = _match_order(k_eng[b], k_ref[b])
            print(f"    view{b} keypoints  max NN dist {dist.max():.4f} px   "
                  f"mean {dist.mean():.4f}   >1px: {int((dist > 1).sum())}/"
                  f"{k_ref.shape[1]}")
            report.setdefault("keypoints", {})[f"view{b}"] = {
                "max_nn_px": float(dist.max()), "mean_nn_px": float(dist.mean()),
                "n_over_1px": int((dist > 1).sum())}
        d_eng, s_eng = outs[1].astype(np.float64), outs[2].astype(np.float64)
        d_ref, s_ref = (z["descriptors"].astype(np.float64),
                        z["scores"].astype(np.float64))
        per = {"cos_med": [], "cos_p1": [], "cos_min": [], "score_max": [],
               "score_mean": []}
        for b in range(k_ref.shape[0]):
            # Order-matched, because `topk` returns its selection in score order
            # and that order is backend-specific - the mistake that once reported
            # a median cosine of 0.88 for an exact operator.
            idx, _dist = _match_order(k_eng[b], k_ref[b])
            d_r, s_r = d_ref[b][idx], s_ref[b][idx]
            cos = ((d_eng[b] * d_r).sum(-1) /
                   (np.linalg.norm(d_eng[b], axis=-1) *
                    np.linalg.norm(d_r, axis=-1) + 1e-12))
            ds = np.abs(s_eng[b] - s_r)
            k1 = max(1, int(0.01 * cos.size))
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

    m_eng, ms_eng = outs[0].astype(np.int64), outs[1].astype(np.float64)
    m_ref, ms_ref = z["matches0"].astype(np.int64), z["mscores0"].astype(np.float64)
    valid_eng, valid_ref = int((m_eng >= 0).sum()), int((m_ref >= 0).sum())
    identical = int((m_eng == m_ref).sum())
    print(f"    matches     engine {valid_eng} valid, ref {valid_ref} valid, "
          f"identical {identical}/{m_ref.size}")
    print(f"    mscores     max|d| {np.abs(ms_eng - ms_ref).max():.3e}")
    report["matches0"] = {"valid_sim": valid_eng, "valid_ref": valid_ref,
                          "identical": identical, "total": int(m_ref.size)}
    report["mscores0"] = {"max_abs": float(np.abs(ms_eng - ms_ref).max())}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", help="the exported ONNX to convert")
    ap.add_argument("--stage", choices=["alike", "lightglue"], required=True)
    ap.add_argument("--precision", default="fp16",
                    help="fp16 (recommended) or fp32; see the docstring for why "
                         "the engine's own layer precision is what gets reported")
    ap.add_argument("--ref", default=None,
                    help="torch reference dump (`--ref` of export_onnx.py); "
                         "without it the numeric check is skipped")
    ap.add_argument("--out", default=None,
                    help="engine path; default is <onnx>_<precision>.engine")
    ap.add_argument("--report", default=None,
                    help="JSON report; default is "
                         "paths.outputs()/trt_<stage>_<name>_<precision>.json")
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--no-verify", action="store_true",
                    help="build only; skip the numeric check against --ref")
    ap.add_argument("--verbose", action="store_true",
                    help="TRT VERBOSE logging into the captured build log")
    args = ap.parse_args()

    # Imported late so `--help` works in an environment without TensorRT, and
    # reported the way the two-environment split is reported everywhere else: name
    # the missing piece and the file that installs it.
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit(
            f"TensorRT is required for conversion ({exc}).\n"
            f"  conda activate alike && pip install -r requirements-alike.txt\n"
            f"  (tensorrt-cu12 - the plain `tensorrt` name resolves to the cu13 "
            f"build, which has no wheel for this platform)") from None
    args.precision = parse_precision(args.precision)
    args.onnx = Path(args.onnx)
    if not args.onnx.is_file():
        raise SystemExit(f"no such ONNX: {args.onnx}")
    out = Path(args.out) if args.out else default_out(args.onnx, args.precision)
    # The precision is part of the name: fp16 and fp32 of one graph are two
    # different engines with two different verification results, and a shared
    # filename would silently keep only the last one built.
    report_path = Path(args.report) if args.report else \
        paths.outputs() / f"trt_{args.stage}_{args.onnx.stem}_{args.precision}.json"

    print(f"[trt] TensorRT {trt.__version__} | CUDA {_cuda_version()} "
          f"| device {_device_label()}")

    engine_bytes, log, seconds = build(args)
    print(f"[trt] built in {seconds:.1f} s, {len(engine_bytes) / 1e6:.2f} MB")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(engine_bytes)
    os.replace(tmp, out)
    print(f"[trt] -> {portable(out)}")

    engine = deserialize(engine_bytes)
    report = {"onnx": portable(args.onnx), "engine": portable(out),
              "stage": args.stage,
              "precision_requested": args.precision,
              "tensorrt": trt.__version__,
              "cuda": _cuda_version(),
              "device": _device_label(),
              "portable": False,
              "portability_note": "engine is compiled for this device, TRT and "
                                  "CUDA version; it will not load on other hardware",
              "workspace_MB": args.workspace_mb, "verbose_build": args.verbose}
    report.update(audit(engine, log, seconds, engine_bytes))
    if not args.no_verify:
        report["verify"] = numeric_check(args, engine_bytes)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[trt] report -> {report_path}")
    return 0


def _device_label():
    """GPU name + compute capability: the provenance an engine needs."""
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return f"{name} (sm_{major}{minor})"
    except Exception:                              # pragma: no cover
        return "unknown"


def _cuda_version():
    """CUDA the engine was compiled against, with a fallback.

    TRT's build-details dict is the authoritative answer but its key names have
    moved between releases, so torch's runtime version is the backup rather than a
    hard failure - an engine's provenance is important, not worth crashing over.
    """
    try:
        import tensorrt as trt
        details = trt.get_build_details()
        for key in ("cuda_version", "cuda"):
            if key in details:
                return str(details[key])
    except Exception:                              # pragma: no cover
        pass
    try:
        import torch
        return str(torch.version.cuda)
    except Exception:                              # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
