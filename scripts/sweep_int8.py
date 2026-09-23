#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which int8 configuration, if any, keeps stage 1 usable?

THE PROBLEM
-----------
The default int8 build (`quantized_dtype='w8a8'`, `quantized_algorithm='normal'`)
does two bad things:

  * it REINTRODUCES 4 `Transpose will fallback to CPU` nodes, which the fp16 graph
    does not have - so the quantised model is not fully NPU-resident either;
  * accuracy collapses: keypoints move 1.13 px on average with a 29 px worst case,
    and the descriptors fall to cosine 0.90.

The descriptor number is the more alarming of the two.  `cos 0.90` sounds
respectable in isolation, but the fp16 model sits at 0.9997 - so this is a 300x
increase in error, and the detector's own output is what feeds it.  A stage 1 that
moves keypoints by a pixel is a different detector.

WHY THE DETECTOR IS THE HARD PART TO QUANTISE
---------------------------------------------
Stage 1's outputs are not a classification or a feature alone; they are a dense
score map, then NMS, then `topk`, then a soft-argmax over a 5x5 window at
temperature 0.1.  The soft-argmax is the sensitive piece: at T=0.1 the weights are
essentially `argmax` with a little smoothing, so a small perturbation of the score
map changes WHICH pixel wins, and the keypoint jumps by a whole pixel rather than
drifting smoothly.  There is no way to average that error away.

WHAT THIS SWEEP VARIES
----------------------
`quantized_dtype`  w8a8 (weights and activations 8-bit), w8a16 (8-bit weights,
                    16-bit activations - the activations are where the score map
                    lives, so this is the obvious candidate), w16a16i.
`quantized_algorithm`  normal, mmse, kl_divergence, gdq - these choose the
                    per-tensor clipping ranges, which is where a sharp score map
                    gets either preserved or flattened.

Each row is built and measured against the fp32 torch reference on the same
`--ref` dump the fp16 path uses, so the columns are directly comparable with
the accuracy report.

Usage
    python scripts/sweep_int8.py --ref refs/alike_stage_bil_ref.npz \
        --dataset refs/calib_stage1.txt --out <reports>/int8_sweep.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from convert_to_rknn import (capture_build_log, count_fallbacks,  # noqa: E402
                             drop_optimizer_dumps)

# (label, quantized_dtype, algorithm).  Ordered cheapest-likely-fix first.
GRID = [
    ("w8a8-normal", "w8a8", "normal"),
    ("w8a8-mmse", "w8a8", "mmse"),
    ("w8a8-kl", "w8a8", "kl_divergence"),
    ("w8a16-normal", "w8a16", "normal"),
    ("w8a16-mmse", "w8a16", "mmse"),
    ("w16a16i-normal", "w16a16i", "normal"),
]


def audit(onnx, dataset, target, dtype, algo, ref, hybrid=False):
    """Build one configuration and return its ledger + accuracy against fp32."""
    from rknn.api import RKNN

    rk = RKNN(verbose=True)
    cfg = {"target_platform": target, "float_dtype": "float16",
           "mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]],
           "quantized_dtype": dtype, "quantized_algorithm": algo}
    rk.config(**cfg)
    if rk.load_onnx(model=str(onnx)) != 0:
        raise RuntimeError("load_onnx failed")

    cap = capture_build_log()
    try:
        with cap:
            rc = rk.build(do_quantization=True, dataset=str(dataset))
        log = cap.read()
    finally:
        cap.cleanup()
        drop_optimizer_dumps()
    if rc != 0:
        raise RuntimeError(f"build failed (see {cap.path if cap.path else 'log'})")
    _, ops, fb = count_fallbacks(log)

    # Accuracy against the fp32 reference dump.
    z = np.load(ref)
    u8 = np.transpose(z["image_u8"], (0, 2, 3, 1)).astype(np.uint8)   # NHWC
    k_ref = z["keypoints"].astype(np.float64)
    d_ref = z["descriptors"].astype(np.float64)

    rk.init_runtime(target=None)
    from scipy.spatial import cKDTree
    outs = [np.asarray(o) for o in
            rk.inference(inputs=[u8], data_format="nhwc")]
    k_sim = next(o for o in outs if o.ndim == 3 and o.shape[-1] == 2).astype(np.float64)
    d_sim = next(o for o in outs if o.ndim == 3 and o.shape[-1] == 128).astype(np.float64)
    rk.release()

    kp_med, kp_max, cos_med, p1 = [], [], [], []
    for b in range(k_ref.shape[0]):
        # Order-matched: `topk` returns score order and that order is
        # backend-specific, so comparing element i of the two outputs compares two
        # different points.  This project has been misled by that twice.
        dist, idx = cKDTree(k_ref[b]).query(k_sim[b], k=1)
        kp_med.append(float(np.median(dist)))
        kp_max.append(float(dist.max()))
        d_r = d_ref[b][idx]
        cos = ((d_sim[b] * d_r).sum(-1) /
               (np.linalg.norm(d_sim[b], axis=-1) *
                np.linalg.norm(d_r, axis=-1) + 1e-12))
        cos_med.append(float(np.median(cos)))
        p1.append(float(np.sort(cos)[max(1, int(0.01 * cos.size)) - 1]))
    return {"kp_median_px": float(np.mean(kp_med)),
            "kp_max_px": float(np.max(kp_max)),
            "desc_cos_median": float(np.mean(cos_med)),
            "desc_cos_p1": float(np.mean(p1)),
            "fallback_nodes": fb["real"],
            "n_will_fallback": fb["n_will_fallback"],
            "log_bytes": len(log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(REPO / "weights/original/alike_stage_bil.onnx"))
    ap.add_argument("--ref", default=str(REPO / "refs/alike_stage_bil_ref.npz"))
    ap.add_argument("--dataset", default=str(REPO / "refs/calib_stage1.txt"))
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    grid = [g for g in GRID if not args.only or g[0] in args.only]
    print(f"{'config':18s} {'fallbacks':>10s} {'kp med px':>10s} "
          f"{'kp max px':>10s} {'desc cos':>9s} {'cos p1':>8s}")
    print("-" * 70)
    rows = []
    for label, dtype, algo in grid:
        try:
            r = audit(args.onnx, args.dataset, args.target, dtype, algo, args.ref)
            r.update({"label": label, "quantized_dtype": dtype,
                      "quantized_algorithm": algo})
            rows.append(r)
            print(f"{label:18s} {r['fallback_nodes']:10d} "
                  f"{r['kp_median_px']:10.4f} {r['kp_max_px']:10.2f} "
                  f"{r['desc_cos_median']:9.5f} {r['desc_cos_p1']:8.5f}")
        except Exception as exc:                       # noqa: BLE001
            rows.append({"label": label, "quantized_dtype": dtype,
                         "quantized_algorithm": algo,
                         "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            print(f"{label:18s}  FAILED: {type(exc).__name__}: {str(exc)[:60]}")

    print()
    print("for reference, fp16: kp med 0.0430 px, kp max 4.02 px, "
          "desc cos 0.9997")
    print("and the fp16 graph has 0 fallback nodes; anything above 0 here means "
          "int8 ADDED host nodes")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"grid": rows}, indent=2))
        print(f"\n[sweep] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
