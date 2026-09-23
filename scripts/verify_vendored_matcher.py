#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prove `model/matchers/lightglue.py` is a faithful port of the upstream class.

Three claims are checked, and each one is a thing that could silently break the
deployed models:

1. **Same architecture.**  The vendored class and the training framework's class,
   built from the same checkpoint conf, must produce the same state dict: same
   keys, same shapes.  A dropped submodule shows up here.

2. **Same numbers.**  Load the SAME checkpoint into both and run the SAME input.
   `matches0` must be identical as integers, and `matching_scores0` /
   `log_assignment` / the per-layer `ref_descriptors` must be bit-identical, not
   merely close.  Two implementations of the same graph that differ only in
   formatting produce exactly equal floats; anything else means a real change.

3. **The removed weights-loading branch was dead.**  This is the one that
   actually mattered.  Every checkpoint's matcher conf carries
   `weights: /path/to/aliked_lightglue.pth`, and upstream reads that file - a
   47 MB external dependency sitting in the training repo - on every
   construction.  The port ignores it.  Claim: the caller's checkpoint load
   overwrites every tensor that branch touched, so the two are equivalent.  The
   test builds upstream WITH the branch and the vendored class WITHOUT it, loads
   the same checkpoint into both, and compares the resulting tensors.

Claim 3 is the reason this script exists rather than a note in a docstring: it is
the checking of a "this load is redundant" assumption, and that assumption is
exactly the kind that is true until the day a checkpoint stops covering every
tensor and the model silently keeps a pretrained weight.

WHY THIS SCRIPT IS ALLOWED TO DEPEND ON THE TRAINING REPO
--------------------------------------------------------
Everything on the *deployable* path (export, refine, convert, evaluate) is now
self-contained.  This script is not on that path: it is the tool that certifies
the port, so it must be able to see both implementations.  Without the training
repo it skips with a clear message instead of failing, and the last recorded
result is in the reports directory (see `paths.outputs()`).

    python scripts/verify_vendored_matcher.py --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import paths  # noqa: E402
from portable_path import portable  # noqa: E402


def _weights_label(weights_field):
    """Report the checkpoint's `weights` field without publishing a real path.

    Every checkpoint carries a `conf.matcher.weights` pointing at the released
    `aliked_lightglue.pth`.  The finding worth reporting is that the field names a
    file the vendored class ignores entirely - the specific location on the
    machine that ran this is not, so only the basename survives.
    """
    if not weights_field:
        return None
    return f"<pretrained:{Path(str(weights_field)).name}>"


def upstream_matcher(model_conf, gluefactory_root=None):
    """Build the matcher the way the training framework does, or return None."""
    if gluefactory_root:
        sys.path.insert(0, str(gluefactory_root))
    try:
        from gluefactory.models.two_view_pipeline import TwoViewPipeline
    except ModuleNotFoundError:
        return None
    pipe = TwoViewPipeline(model_conf)
    return pipe.matcher.eval()


def resolve_upstream_root(explicit):
    if explicit:
        return explicit
    if os.environ.get("GLUEFACTORY_ROOT"):
        return os.environ["GLUEFACTORY_ROOT"]
    for cand in (REPO.parent / "glue-factory", REPO / "glue-factory"):
        if (cand / "gluefactory").is_dir():
            return str(cand)
    return None


def compare_state_dicts(a, b, label):
    ka, kb = set(a), set(b)
    diffs = {}
    if ka - kb:
        diffs["only_upstream"] = sorted(ka - kb)[:8]
    if kb - ka:
        diffs["only_vendored"] = sorted(kb - ka)[:8]
    worst, worst_key = 0.0, None
    shape_mismatch = []
    for k in sorted(ka & kb):
        if tuple(a[k].shape) != tuple(b[k].shape):
            shape_mismatch.append((k, tuple(a[k].shape), tuple(b[k].shape)))
            continue
        d = float((a[k].float() - b[k].float()).abs().max())
        if d > worst:
            worst, worst_key = d, k
    if shape_mismatch:
        diffs["shape_mismatch"] = shape_mismatch[:5]
    return {"label": label, "n_upstream": len(ka), "n_vendored": len(kb),
            "max_abs_diff": worst, "max_abs_diff_key": worst_key,
            "identical": worst == 0.0 and not diffs, "problems": diffs}


def compare_forward(out_a, out_b):
    """Bit-equality per output tensor.  No tolerance: this is the same graph."""
    problems, rows = [], []
    for k in sorted(out_a):
        va, vb = out_a[k], out_b[k]
        if not torch.is_tensor(va) or not torch.is_tensor(vb):
            continue
        if tuple(va.shape) != tuple(vb.shape):
            problems.append(f"{k}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
            continue
        if va.dtype != vb.dtype:
            problems.append(f"{k}: dtype {va.dtype} vs {vb.dtype}")
            continue
        if "int" in str(va.dtype) or "bool" in str(va.dtype):
            same = bool(torch.equal(va, vb))
            rows.append((k, str(va.dtype), 0.0 if same else 1.0, same))
        else:
            d = float((va.float() - vb.float()).abs().max())
            rows.append((k, str(va.dtype), d, d == 0.0))
            if d != 0.0:
                problems.append(f"{k}: max|Δ| = {d:.3e}")
    return rows, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(
        REPO / "weights/checkpoints/alike_native_gl_s1_9L.tar"))
    ap.add_argument("--gluefactory-root", default=None)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--json", default=str(paths.outputs() / "vendored_matcher.json"))
    args = ap.parse_args()

    root = resolve_upstream_root(args.gluefactory_root)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_conf = ck["conf"]["model"]
    if not isinstance(model_conf, dict):
        from omegaconf import OmegaConf
        model_conf = OmegaConf.to_container(OmegaConf.create(model_conf), resolve=True)
    mconf = model_conf["matcher"]
    print(f"checkpoint : {args.checkpoint}")
    print(f"matcher    : input_dim={mconf['input_dim']} n_layers={mconf['n_layers']} "
          f"heads={mconf['num_heads']} agg={mconf.get('aggregation', 'last')}")
    print(f"upstream   : {root or '(not found)'}")

    from model.matchers.lightglue import LightGlue as VendoredLightGlue

    # --- claims 1 and 2 need both implementations -------------------------------
    up = upstream_matcher(model_conf, root)
    if up is None:
        payload = {"status": "skipped",
                   "reason": "training repo not importable; set GLUEFACTORY_ROOT "
                             "to re-run, or read the last recorded result",
                   "checkpoint": args.checkpoint}
        print(f"\nSKIP: {payload['reason']}")
    else:
        # BOTH must be in eval mode, and this is not cosmetic.  `forward` branches
        # on `self.training` when it collects `ref_descriptors` - train mode keeps
        # every layer's descriptors, eval keeps only the last - so a mismatched
        # pair compares (B, n_layers, K, D) against (B, 1, K, D) and reports a
        # difference that is a mode flag, not a port defect.  The deploy path is
        # eval-only, so that is the mode under test.
        vend = VendoredLightGlue(mconf).eval()
        state = {k[len("matcher."):]: v for k, v in ck["model"].items()
                 if k.startswith("matcher.")}
        up.load_state_dict(state, strict=False)
        vend.load_state_dict(state, strict=False)
        assert not up.training and not vend.training, "both sides must be .eval()"

        sd = compare_state_dicts(up.state_dict(), vend.state_dict(), "arch")
        print(f"\n[1] architecture: {sd['n_upstream']} vs {sd['n_vendored']} tensors, "
              f"max|Δ| = {sd['max_abs_diff']:.3e} -> "
              f"{'IDENTICAL' if sd['identical'] else 'DIFFERS'}")

        torch.manual_seed(0)
        k = args.keypoints
        data = {
            "keypoints0": torch.rand(2, k, 2) * 512,
            "keypoints1": torch.rand(2, k, 2) * 512,
            "descriptors0": torch.randn(2, k, mconf["input_dim"]),
            "descriptors1": torch.randn(2, k, mconf["input_dim"]),
            "view0": {"image_size": torch.tensor([[512.0, 512.0]]).expand(2, 2)},
            "view1": {"image_size": torch.tensor([[512.0, 512.0]]).expand(2, 2)},
        }
        with torch.no_grad():
            oa, ob = up(dict(data)), vend(dict(data))
        rows, problems = compare_forward(oa, ob)
        for name, dt, d, ok in rows:
            print(f"    {name:22s} {dt:9s} max|Δ| = {d:.3e}  {'ok' if ok else 'MISMATCH'}")
        exact = not problems
        print(f"[2] forward: {len(rows)} tensors, "
              f"{'BIT-IDENTICAL' if exact else 'DIFFERS: ' + '; '.join(problems[:3])}")

        # --- claim 3 -----------------------------------------------------------
        # `up` was built WITH the weights branch (it read the .pth); `vend` was
        # built without it.  Both then had the same checkpoint loaded, so equal
        # state dicts mean the branch contributed nothing.
        print("[3] removed weights-load: comparing the two AFTER the checkpoint "
              f"load\n    (upstream read {mconf.get('weights')})")
        sd3 = compare_state_dicts(up.state_dict(), vend.state_dict(),
                                  "after-checkpoint")
        print(f"    max|Δ| = {sd3['max_abs_diff']:.3e} -> "
              f"{'redundant, confirmed' if sd3['identical'] else 'NOT redundant'}")

        payload = {
            "status": "verified" if (exact and sd3["identical"]) else "FAILED",
            # Repo-relative / role-labelled rather than absolute - see
            # scripts/portable_path.py.  `upstream_root` is intentionally omitted:
            # it is the training checkout's location, which is exactly the kind of
            # machine detail a committed report should not carry.
            "checkpoint": portable(args.checkpoint, "checkpoint"),
            "checkpoint_weights_field": _weights_label(mconf.get("weights")),
            "n_tensors": sd["n_vendored"],
            "arch_identical": sd["identical"],
            "arch_problems": sd["problems"],
            "forward_exact": exact,
            "forward_tensors": [{"name": n, "dtype": d, "max_abs_diff": v, "exact": ok}
                                for n, d, v, ok in rows],
            "weights_branch_redundant": sd3["identical"],
            "weights_branch_max_abs_diff": sd3["max_abs_diff"],
        }

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    if payload["status"] == "FAILED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
