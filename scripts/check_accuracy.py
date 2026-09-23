#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end accuracy: torch vs ONNX vs RKNN, on REAL images.

Why the earlier per-tensor comparison was not enough
---------------------------------------------------
A per-tensor cosine is easy to compute and easy to misread. This project already
hit that twice: a descriptor "failure" at cosine 0.88 that was an order-matching
artefact (the comparison ran by position, and `topk` returns score order), and a
keypoint "error" of 484 px that was the backend returning the same points in a
different ORDER. Both were reported by a naive comparison and neither was a
conversion bug; the residual that IS real is the 0.043 px median keypoint jitter
an fp16 NPU introduces (per-pair median over 8 HPatches pairs).

So this script does two things the tensor comparison cannot:

1. **Attributes the residual**, by sampling the TORCH descriptor map at the
   BACKEND's keypoints. That separates "the gather computes the wrong thing" from
   "the keypoints moved slightly", which have completely different fixes.
2. **Measures what the model is for** - matches between two real images - and
   scores them against a geometry, so a shift in keypoint ORDER cancels out.

Usage
    python scripts/check_accuracy.py --pairs 8 --size 512
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import paths  # noqa: E402
from model import load_alike_stage  # noqa: E402
from model.gather_ops import GatherSampleBilinear, to_pixel  # noqa: E402



def load_hpatches_pairs(n):
    """Real image pairs with a known homography, from HPatches."""
    out = []
    for seq in sorted(paths.hpatches().iterdir()):
        if not seq.is_dir() or seq.name[0] == "i":
            continue          # viewpoints only: illumination pairs are too easy
        ref = seq / "1.ppm"
        for q in range(2, 7):
            qf = seq / f"{q}.ppm"
            hf = seq / f"H_1_{q}"
            if ref.is_file() and qf.is_file() and hf.is_file():
                out.append((ref, qf, hf))
                break
        if len(out) >= n:
            break
    return out


def read_gray_img(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = np.stack([img] * 3, -1)                       # ALIKE takes 3 channels
    return rgb


def as_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def concat_views(*views):
    """Stack per-image results along the batch dim, as the batch-2 graph did.

    Stage 1 is per-image now, so a pair is two inference calls.  Re-building the
    `(2, ...)` tensor keeps every comparison and the residual attribution below
    unchanged, which is what makes this a change to the inference calls only.
    """
    return tuple(torch.cat([v[i] for v in views], dim=0) for i in range(3))


def concat_views_np(*views):
    """`concat_views` for numpy-returning backends (ONNX Runtime, RKNN)."""
    return tuple(np.concatenate([v[i] for v in views], axis=0) for i in range(3))


def attribute_residual(model, image, k_sim, d_sim, k_ref, d_ref):
    """Split the descriptor error into gather error and keypoint-jitter error.

    BOTH comparisons are done with the correspondence MATCHED FIRST, by nearest
    neighbour.  That is not a refinement - it is the difference between a valid
    measurement and a meaningless one.  `topk` returns its selection in score
    order and that order is backend-specific, so comparing `d_sim[i]` against
    `d_ref[i]` compares descriptors of two DIFFERENT points.  This project has
    already been misled twice by exactly that mistake (a reported 484 px keypoint
    error, and a descriptor cosine of 0.88 that looked like a conversion bug).

    Returns
    -------
    cos_gather : cosine between the backend's descriptor and the torch descriptor
                 map sampled at the BACKEND's keypoints (order-matched).
                 ~1.0 means the sampling operator is exact.
    cos_total  : cosine between the backend's descriptor and the torch
                 descriptor at the CORRESPONDING torch keypoint (order-matched).
                 Anything below cos_gather is attributable to keypoint movement.
    """
    h = w = image.shape[-1]
    with torch.no_grad():
        _, dm = model.dense_maps(image)                    # torch descriptor map
        bil = GatherSampleBilinear(batch=k_sim.shape[0], channels=128, hw=h * w)

        def sample_at(kpts):
            xn, yn = to_pixel(model.dkd._normalise(kpts, h, w), h, w)
            d = bil(dm, xn, yn)
            return F.normalize(d, p=2, dim=1).transpose(1, 2)

        d_at_sim = sample_at(k_sim)                       # torch sampler, backend kpts

    # Order-matched descriptors from the torch reference.  `k_ref`/`d_ref` are a
    # single view, i.e. (K, 2) and (K, D); `k_sim`/`d_sim` keep the batch dim.
    k_ref_np = k_ref.numpy() if hasattr(k_ref, "numpy") else np.asarray(k_ref)
    if k_ref_np.ndim == 3:
        k_ref_np = k_ref_np[0]
    k_sim_np = k_sim[0].cpu().numpy()
    idx = torch.from_numpy(cKDTree(k_ref_np).query(k_sim_np, k=1)[1])
    if d_ref.dim() == 3:
        d_ref = d_ref[0]
    d_ref_matched = d_ref[idx].unsqueeze(0)

    cos_gather = F.cosine_similarity(d_sim, d_at_sim, dim=-1)
    cos_total = F.cosine_similarity(d_sim, d_ref_matched, dim=-1)
    return cos_gather, cos_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--onnx", default=str(REPO / "weights/original/alike_stage_bil.onnx"))
    ap.add_argument("--rknn", default=None,
                    help="RKNN stage-1 model; omit to skip the backend column")
    ap.add_argument("--out", default=str(paths.outputs() / "accuracy.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    # gathered_head=False: the residual attribution samples the torch DENSE
    # descriptor map at the backend's keypoints, so the map must exist.
    torch_model = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                                   descriptor_interp="bilinear",
                                   gathered_head=False).eval()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    rk = None
    if args.rknn and Path(args.rknn).is_file():
        # Imported lazily: without --rknn this script runs in the export env,
        # which does not carry the RKNN toolkit.
        from rknn.api import RKNN
        rk = RKNN(verbose=False)
        rk.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                  target_platform="rk3588", float_dtype="float16")
        rk.load_onnx(model=args.onnx)
        rk.build(do_quantization=False)
        rk.init_runtime(target=None)

    pairs = load_hpatches_pairs(args.pairs)
    print(f"[acc] {len(pairs)} real HPatches pairs at {args.size}px, "
          f"{args.keypoints} keypoints\n")

    rows = []
    for i, (pa, pb, hf) in enumerate(pairs):
        ra, rb = read_gray_img(pa, args.size), read_gray_img(pb, args.size)
        if ra is None or rb is None:
            continue
        ia, ib = as_tensor(ra), as_tensor(rb)
        u8a = np.stack([ra]).transpose(0, 3, 1, 2)         # (1,3,S,S) uint8 colour
        u8b = np.stack([rb]).transpose(0, 3, 1, 2)

        # --- torch -----------------------------------------------------------
        # ONE IMAGE PER CALL: the extractor is per-image, so a pair is two calls.
        def run_torch(img):
            with torch.no_grad():
                return torch_model(img)        # (1,K,2), (1,K,128), (1,K)

        k_t, d_t, s_t = concat_views(run_torch(ia), run_torch(ib))
        # --- ONNX ------------------------------------------------------------
        k_o, d_o, s_o = concat_views_np(
            sess.run(None, {"image": ia.numpy()}),
            sess.run(None, {"image": ib.numpy()}))
        # --- RKNN ------------------------------------------------------------
        k_r = d_r = s_r = None
        if rk is not None:
            def run_rknn(img_u8):
                outs = rk.inference(inputs=[np.transpose(img_u8, (0, 2, 3, 1))],
                                    data_format="nhwc")
                return tuple(np.asarray(o, dtype=np.float32) for o in outs)

            k_r, d_r, s_r = concat_views_np(run_rknn(u8a), run_rknn(u8b))

        row = {"pair": f"{pa.parent.name}/{pa.name}"}
        # keypoint agreement, order-independent
        for tag, k in (("onnx", k_o), ("rknn", k_r)):
            if k is None:
                continue
            d, _ = cKDTree(k_t[0].numpy()).query(k[0], k=1)
            row[f"kp_{tag}_median_px"] = float(np.median(d))
            row[f"kp_{tag}_max_px"] = float(d.max())
            row[f"kp_{tag}_over_1px"] = int((d > 1).sum())

        if d_r is not None:
            # Attribute the descriptor residual, with the correspondence matched
            # first - see `attribute_residual` for why that is mandatory.
            with torch.no_grad():
                k_ref_t, d_ref_t, _ = torch_model(ia)     # torch, same single view
            cos_gather, cos_total = attribute_residual(
                torch_model, ia, torch.from_numpy(k_r[0:1]).float(),
                torch.from_numpy(d_r[0:1]).float(),
                k_ref_t[0].cpu(), d_ref_t[0].cpu())
            row["desc_gather_median"] = float(cos_gather.median())
            row["desc_total_median"] = float(cos_total.median())
            row["desc_gather_p1"] = float(cos_gather.kthvalue(
                max(1, int(0.01 * cos_gather.numel()))).values)
            row["desc_total_p1"] = float(cos_total.kthvalue(
                max(1, int(0.01 * cos_total.numel()))).values)
        rows.append(row)
        print(f"  [{i+1}/{len(pairs)}] {row['pair']:24s} "
              f"kp_onnx {row.get('kp_onnx_median_px', float('nan')):.4f}px  "
              f"kp_rknn {row.get('kp_rknn_median_px', float('nan')):.4f}px  "
              f"desc_gather {row.get('desc_gather_median', float('nan')):.5f}  "
              f"desc_total {row.get('desc_total_median', float('nan')):.5f}")

    if rk is not None:
        rk.release()

    def agg(key):
        """MEAN over pairs of that pair's own statistic.

        Not a percentile pooled over all keypoints of all pairs, and for the
        `_max_px` key it is a mean of per-pair maxima rather than a maximum.  The
        distinction is written into the output file (`summary_aggregation`) so a
        reader cannot mistake the header for a pooled figure.
        """
        v = [r[key] for r in rows if key in r]
        return float(np.mean(v)) if v else None

    summary = {k: agg(k) for k in
               ("kp_onnx_median_px", "kp_rknn_median_px", "kp_rknn_max_px",
                "desc_gather_median", "desc_gather_p1",
                "desc_total_median", "desc_total_p1")}
    summary_aggregation = {
        "rule": f"mean over the {len(rows)} pairs of the per-pair statistic",
        "per_pair_statistic":
            "median / max / 1st percentile over that pair's keypoints; descriptor "
            "cosines order-matched by keypoint nearest neighbour",
        "caveat":
            "NOT a percentile pooled over all keypoints, and `kp_rknn_max_px` is "
            "a mean of the per-pair maxima rather than a global maximum; read the "
            "per-pair `pairs` entries for tails",
        "n_pairs": len(rows),
    }
    print("\n=== summary over real image pairs ===")
    print("  (each value is the MEAN OVER PAIRS of that per-pair statistic, not a"
          " pooled percentile - see `summary_aggregation` in the output file)")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:32s} {v:.6f}")
    print("\n  interpretation:")
    print("    desc_gather_median ~1.0 -> the sampling operator is exact")
    print("    desc_total_median       -> descriptor agreement INCLUDING the effect")
    print("                               of keypoints having moved slightly")
    print("    (both order-matched; an order-mismatched comparison is meaningless)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "summary_aggregation": summary_aggregation,
         "pairs": rows}, indent=2))
    print(f"\n[acc] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
