#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export the two deployment stages to ONNX, with a torch reference dump.

    stage 1  AlikeStage       image (1,3,S,S) f32 [0,1] -> keypoints, descriptors, scores
    stage 2  LightGlueStage   keypoints/descriptors      -> matches0, mscores0

Both are exported with STATIC shapes.  That is not a convenience: RKNN requires
fixed dimensions for every tensor, so a dynamic axis anywhere (a batch dim, an
image size, a variable keypoint count) makes the model unconvertible.  The whole
pipeline is therefore pinned to the deployed configuration - `S = 512` and
`K = 512` - and the export asserts it rather than hoping.

STAGE 1 IS PER-IMAGE.  It takes one image per call and a pair is two calls, so
`--ref` writes a SINGLE-image dump and stage 2 is exported on TWO of them
(`--stage1-ref a.npz,b.npz`, in view0/view1 order).  The sampler's row-offset
tables are baked at `--batch`/`--size`, which is why those have to match the graph
being converted rather than being guessed at conversion time.

The `--ref` dump is what makes the accuracy protocol possible: it stores the
exact inputs plus every output of the TORCH model, so the ONNX and RKNN stages
can each be scored against the same ground truth without re-deriving it.

Input convention
----------------
The ONNX model takes float32 in [0, 1] (bit-identical to what the torch
reference sees).  At RKNN conversion time the model is configured with
`mean=0, std=255` and fed uint8 directly, so the NPU performs the /255 itself and
no float preprocessing is needed in C++ - see `convert_to_rknn.py`.  Storing BOTH
forms of the input in the reference dump is what lets the two backends be
compared on identical pixels.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import load_alike_stage, load_lightglue_stage  # noqa: E402


def first_conv_weights_moved(model) -> int:
    """Cheap sanity probe: how many `block1` tensors are non-default.

    A stage that silently failed to load its weights still exports and still
    runs - it just produces garbage.  Counting tensors that differ from a fresh
    init catches that before anything is converted.
    """
    fresh = type(model)(variant=model.variant, top_k=model.top_k)
    moved = 0
    cur, ref = model.state_dict(), fresh.state_dict()
    for k, v in cur.items():
        if k in ref and ref[k].shape == v.shape and not torch.equal(v, ref[k]):
            moved += 1
    return moved


def export_stage1(args, dev):
    stage = load_alike_stage(args.checkpoint, variant=args.variant,
                             top_k=args.keypoints, nms_radius=args.nms_radius,
                             descriptor_interp=args.descriptor_interp,
                             gathered_head=args.gathered_head,
                             batch=args.batch, size=args.size)
    moved = first_conv_weights_moved(stage)
    if moved == 0:
        raise RuntimeError("stage 1 loaded NO non-default weights - the "
                           "checkpoint prefix filter is wrong")
    print(f"[export] stage 1: {moved} state tensors differ from a fresh init "
          f"(weights really loaded)")

    stage.eval().to(dev)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    image_u8 = torch.randint(0, 256, (args.batch, 3, args.size, args.size),
                             dtype=torch.uint8, generator=g)
    image = (image_u8.float() / 255.0).to(dev)

    with torch.no_grad():
        kpts, desc, scores = stage(image)

    dynamic = None  # static everywhere - required by RKNN
    torch.onnx.export(
        stage, (image,), args.out,
        input_names=["image"], output_names=["keypoints", "descriptors", "scores"],
        opset_version=args.opset, do_constant_folding=True, dynamic_axes=dynamic,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"[export] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")

    if args.ref:
        np.savez_compressed(
            args.ref,
            image_u8=image_u8.numpy(),          # for the RKNN path
            image=image.cpu().numpy(),          # for the ONNX path
            keypoints=kpts.cpu().numpy(),
            descriptors=desc.cpu().numpy(),
            scores=scores.cpu().numpy(),
        )
        print(f"[export] torch reference -> {args.ref}")
    return kpts, desc, scores


def export_stage2(args, dev):
    stage = load_lightglue_stage(args.checkpoint, image_size=(args.size, args.size))

    if args.attention == "matmul":
        # 27 of the 29 nodes that fall off the NPU in this graph are `Einsum`
        # (attention), and `Einsum` has no RKNN lowering.  The rewrite is exact -
        # `scripts/ablate_attention.py --verify` shows a max |Δscore| of 0 - so it
        # is opt-in here rather than default only because it needs that
        # verification to have been run.
        import model.matchers.lightglue as _lg
        from ablate_attention import patch_cross_block, patch_match_assignment
        n = 0
        for block in stage.matcher.transformers:
            patch_cross_block(block.cross_attn)
            n += 1
        if n == 0:
            raise RuntimeError("--attention matmul patched no block; the export "
                               "would still contain Einsum")
        m = 0
        for la in stage.matcher.log_assignment:
            patch_match_assignment(la, orig_fn=_lg.sigmoid_log_double_softmax)
            m += 1
        if m == 0:
            raise RuntimeError("--attention matmul patched no MatchAssignment; the "
                               "export would still contain ScatterElements")
        print(f"[export] attention -> MatMul in {n} CrossBlock(s); "
              f"Scatter* -> Concat in {m} MatchAssignment(s)")

    # The matcher never sees an image; its inputs are what stage 1 emits.  Export
    # it on the ACTUAL stage-1 outputs rather than on random tensors, so the
    # reference dump is a reachable input and the parity numbers mean something.
    # Stage 1 is per-image, so ONE view is one dump and a pair is TWO of them.
    refs = [p for p in (args.stage1_ref or "").split(",") if p]
    if len(refs) == 2 and all(os.path.isfile(p) for p in refs):
        z0, z1 = (np.load(p) for p in refs)
        for p, z in zip(refs, (z0, z1)):
            if z["keypoints"].shape[0] != 1:
                raise ValueError(
                    f"{p} holds {z['keypoints'].shape[0]} views; stage 1 is exported "
                    f"one image per call, so each reference has to be a "
                    f"single-image dump (re-export it with --batch 1)")
        k0 = torch.from_numpy(z0["keypoints"]).to(dev).float()
        d0 = torch.from_numpy(z0["descriptors"]).to(dev).float()
        k1 = torch.from_numpy(z1["keypoints"]).to(dev).float()
        d1 = torch.from_numpy(z1["descriptors"]).to(dev).float()
        print(f"[export] stage 2 inputs: view0 {refs[0]}, view1 {refs[1]}")
    else:
        if args.stage1_ref:
            raise ValueError(
                f"--stage1-ref takes exactly two comma-separated single-image "
                f"stage-1 dumps (view0,view1); got {args.stage1_ref!r}")
        g = torch.Generator(device="cpu").manual_seed(args.seed)

        def random_view():
            k = torch.rand(1, args.keypoints, 2, generator=g).to(dev) * args.size
            d = torch.nn.functional.normalize(
                torch.randn(1, args.keypoints, 128, generator=g), dim=-1).to(dev)
            return k, d

        k0, d0 = random_view()
        k1, d1 = random_view()
        print("[export] WARNING: stage-1 reference missing, using random matcher "
              "inputs - the parity dump will not be a reachable input")

    stage.eval().to(dev)
    with torch.no_grad():
        m0, ms0 = stage(k0, k1, d0, d1)

    torch.onnx.export(
        stage, (k0, k1, d0, d1), args.out,
        input_names=["keypoints0", "keypoints1", "descriptors0", "descriptors1"],
        output_names=["matches0", "mscores0"],
        opset_version=args.opset, do_constant_folding=True, dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"[export] wrote {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")

    if args.ref:
        np.savez_compressed(
            args.ref,
            keypoints0=k0.cpu().numpy(), keypoints1=k1.cpu().numpy(),
            descriptors0=d0.cpu().numpy(), descriptors1=d1.cpu().numpy(),
            matches0=m0.cpu().numpy(), mscores0=ms0.cpu().numpy(),
        )
        print(f"[export] torch reference -> {args.ref}")
    n_matched = int((m0[0] >= 0).sum())
    print(f"[export] sanity: {n_matched}/{k0.shape[1]} keypoints matched in the "
          f"reference forward")
    return m0, ms0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="alike", choices=["alike", "lightglue"],
                    help="which stage to export")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None,
                    help="write the torch reference (inputs + outputs) here; with "
                         "--batch 1 (the default) this is ONE image")
    ap.add_argument("--stage1-ref", default=None,
                    help="stage-2 only: two comma-separated SINGLE-image stage-1 "
                         "reference npz files, view0 first (`a.npz,b.npz`)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1,
                    help="images per stage-1 call.  1 is the deployment: the "
                         "extractor runs once per image and a pair is two calls.")
    ap.add_argument("--variant", default="alike-n")
    ap.add_argument("--nms-radius", type=int, default=2)
    ap.add_argument("--gathered-head", dest="gathered_head", action="store_true",
                    default=True,
                    help="compute the descriptor only at the keypoints instead of "
                         "materialising the dense 129-channel map (default, and "
                         "verified exact).  `--no-gathered-head` reproduces the "
                         "previous graph for comparison.")
    ap.add_argument("--no-gathered-head", dest="gathered_head",
                    action="store_false",
                    help="export the dense-head graph (the pre-restructure form)")
    ap.add_argument("--descriptor-interp", default="bilinear",
                    choices=["bilinear", "floor"],
                    help="descriptor sampling at sub-pixel keypoints; bilinear is "
                         "continuous in the keypoint and therefore robust to the "
                         "0.043px median jitter an fp16 NPU introduces, floor is "
                         "upstream ALIKE's rule")
    ap.add_argument("--attention", default="matmul", choices=["matmul", "einsum"],
                    help="stage-2 attention implementation.  'matmul' (default) "
                         "rewrites the two Einsum contractions as MatMul, which is "
                         "what removes the 27 NPU fallback nodes - it is exact "
                         "(verified at 0 delta).  'einsum' keeps upstream's form "
                         "for reproducing the old graph.")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.size % 32:
        ap.error(f"--size {args.size} must be a multiple of 32 (ALIKE downsamples "
                 f"by 32 and this export only supports the no-padding path)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dev = torch.device(args.device)
    print(f"[export] stage={args.stage} size={args.size} "
          f"keypoints={args.keypoints} batch={args.batch} opset={args.opset} "
          f"attention={args.attention} device={args.device}")
    if args.stage == "alike":
        export_stage1(args, dev)
    else:
        export_stage2(args, dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
