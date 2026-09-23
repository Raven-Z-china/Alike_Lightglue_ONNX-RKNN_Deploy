#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structural ablations of the matcher, measured against the TRAINED WEIGHTS.

THE PROBLEM THIS SOLVES
-----------------------
LightGlue's depth, head count and width are hyperparameters, and the obvious way
to try a new value is to retrain.  That is the wrong way to search this space, for
three reasons:

1. **Cost.** Eight configs x a training run is days.  Most of the search can be
   done in minutes if the question is asked correctly.
2. **It answers the wrong question.** Retraining measures "how good is this
   configuration when optimised for it".  What a deployment needs to know first
   is "how much is this configuration worth", and that is a property of the
   weights we already have.  If layers 9->8 costs almost nothing with the
   EXISTING weights, it will cost nothing after fine-tuning either.
3. **The result transfers.** Truncating a trained model yields a real accuracy
   number for a real parameter count, on the actual evaluator, with no training
   in the loop.  A retrained config gives a number that is not comparable to the
   incumbent because it has had more optimisation.

So: truncate, measure, and only retrain the survivors.  `--refit` does the
retraining for the ones that need it.

WHAT CAN BE CHANGED WITHOUT TOUCHING A WEIGHT
--------------------------------------------
`depth`      keep the first N `SelfBlock`s and the first (N-1) `CrossBlock`s.
             The blocks are numbered, so dropping the tail is a pure slice -
             every surviving tensor keeps its shape and its weights.
`heads`      keep the first H attention heads.  `head_dim = dim // heads` is
             unchanged, so the projections slice cleanly.
`aggregation`'s `uniform` replaces the learned token weighting with a mean.
             It drops `token.0/1` but the branch still exists, so the state
             dict stays loadable with `strict=False`.
`width`      **NOT** a truncation.  `dim` 256 -> 128 changes the shape of
             nearly every weight, and slicing them means taking the first 128
             channels of a layer that was trained with all 256 - which is not an
             ablation, it is damage.  It is reported here for completeness and
             labelled as requiring `--refit`.

READ THIS BEFORE TRUSTING A ROW
-------------------------------
A truncation's accuracy number is a LOWER BOUND on what that configuration can
do, because the surviving layers were trained in the presence of the ones that
were removed.  That is the useful direction for a search: a config that is
already good truncated is a safe choice, and a config that is bad truncated may
still be good retrained.  The reverse inference - "this config is fine, ship it"
from a truncated number - is NOT valid.

Usage
    python scripts/ablate_matcher.py \
        --checkpoint /path/to/alike_native_gl_s1/checkpoint_best.tar \
        --hpatches 240 \
        --report <reports>/ablation_matcher.md
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))
from portable_path import portable  # noqa: E402

# Reuse the judges rather than reimplementing them: two implementations of
# "mAA" drift, and then two reports disagree for reasons that are not the model.
import run_accuracy as RA  # noqa: E402


# --------------------------------------------------------------------------- #
# what is being swept
# --------------------------------------------------------------------------- #
#: (label, kwargs).  Ordered cheapest-change-first so a run that has to be cut
#: short still yields the most informative rows.
#: `heads` is DISABLED.  `slice_attention_heads` is not a valid weight truncation
#: for this matcher - `SelfBlock.Wqkv` does not scale with `num_heads` and its head
#: layout is interleaved, while `CrossBlock.to_*` has the SAME shape at 4 and 2
#: heads, so slicing produces tensors the framework cannot load.  The rows are
#: commented out rather than deleted so the numbers already published from them can
#: be traced to the code that produced them; they are labelled unreproducible in the
#: README.  A `num_heads: 2` retrain is the sound measurement.
GRID = [
    ("baseline",            dict()),
    ("depth-8",             dict(depth=8)),
    ("depth-7",             dict(depth=7)),
    ("depth-6",             dict(depth=6)),
    ("agg-uniform",         dict(aggregation="uniform")),
    ("depth-7+agg-uniform", dict(depth=7, aggregation="uniform")),
    # ("heads-2",          dict(heads=2)),          # NOT a valid truncation
    # ("depth-8+heads-2",  dict(depth=8, heads=2)), # NOT a valid truncation
    # ("depth-7+heads-2",  dict(depth=7, heads=2)), # NOT a valid truncation
]


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def describe(cfg):
    bits = []
    for k in ("depth", "heads", "aggregation"):
        if k in cfg:
            bits.append(f"{k}={cfg[k]}")
    return ", ".join(bits) if bits else "as trained"


# The trained matcher keeps its layers in a single `nn.ModuleList` called
# `transformers`, NOT in separate `self_blocks`/`cross_blocks` lists - whether a
# block is self- or cross-attention is a property of the block.  The per-layer
# auxiliaries (`log_assignment`, `token_confidence`) are ModuleLists indexed by
# layer, so all three have to be cut in lockstep.  Cutting only `transformers`
# does not raise: `forward` keeps indexing `log_assignment[i]` and quietly reads
# the wrong layer's logits, producing a model that runs and is wrong.
#
# THE THREE LISTS DO NOT HAVE THE SAME LENGTH, AND THAT IS THE SUBTLE PART
# ----------------------------------------------------------------------
# `LightGlue.__init__` builds:
#
#     self.transformers     = [TransformerLayer(d, h) for _ in range(n)]      # n
#     self.log_assignment   = [MatchAssignment(d)   for _ in range(n)]        # n
#     self.token_confidence = [TokenConfidence(d)   for _ in range(n - 1)]    # n-1
#
# `n - 1`, because the last layer produces the final assignment and needs no
# "should I stop here" head.  So `del lst[depth:]` applied uniformly to all three
# is right for the first two and one too few for the third: truncating to 6 keeps
# `token_confidence[0:6]` when the framework builds only 5.
#
# That extra layer does NOT look like a problem at the time.  The fine-tune loads
# with `strict=False`, which reports the surplus tensor as `unexpected` and carries
# on - so a 6-layer truncation quietly ships one `TokenConfidence` block that is
# never in the graph it was compared against, while the parameter count still looks
# plausible.  It was caught only by building the model from the config and comparing
# the three list lengths.  `_layer_lists` therefore returns each list with the
# length it is SUPPOSED to have, and `apply_ablation` truncates to that target
# rather than to a single shared number.
def _layer_lists(model):
    """(list, target_length_when_n_layers_is) for each per-layer ModuleList.

    The second element is a callable because `token_confidence` is built from
    `n - 1` and the others from `n`; encoding that here is what stops the two from
    being truncated to the same number.
    """
    return (
        (model.transformers, lambda n: n),
        (model.log_assignment, lambda n: n),
        (model.token_confidence, lambda n: n - 1),
    )


def apply_ablation(model, cfg):
    """Mutate `model` in place.  Returns a note describing what changed."""
    note = ""
    cfg = dict(cfg)
    out_dim = cfg.pop("out_dim", None)
    depth = cfg.pop("depth", None)
    heads = cfg.pop("heads", None)
    aggreg = cfg.pop("aggregation", None)
    if cfg:
        raise ValueError(f"unknown ablation keys: {sorted(cfg)}")

    if out_dim not in (None, model.conf.descriptor_dim):
        raise NotImplementedError(
            f"descriptor_dim {model.conf.descriptor_dim}->{out_dim} is not a "
            f"truncation: `input_proj`, every `qkv`/`out_proj` and the residual "
            f"stream all change width, and slicing them gives a damaged model "
            f"rather than a smaller one.  It needs a retrain, so it is not part "
            f"of this grid.")

    conf = model.conf
    if depth is not None:
        if depth < 2:
            raise ValueError(f"depth={depth}: LightGlue needs >= 2 layers - the "
                             f"first to mix the two images and the last to "
                             f"produce the assignment")
        if depth > conf.n_layers:
            raise ValueError(f"depth={depth} > trained n_layers={conf.n_layers}: "
                             f"there are no extra layers to keep")
        # Each list is cut to ITS OWN target length, because `token_confidence` is
        # built with `n - 1` entries rather than `n`.  A shared `del lst[depth:]`
        # leaves it one layer too long, and the surplus tensor loads as
        # `unexpected` under `strict=False` without any error - see `_layer_lists`.
        for lst, target in _layer_lists(model):
            del lst[target(depth):]
        # `confidence_thresholds` is a plain parameter of length `n_layers` - it is
        # NOT in one of the three ModuleLists, so cutting them leaves it oversized.
        # That matters beyond tidiness: `--conf` describes the architecture, so the
        # framework builds a 7-length buffer and `load_state_dict` then fails with
        # `size mismatch for matcher.confidence_thresholds: copying a param with
        # shape [9] from checkpoint, the shape in current model is [7]`.
        #
        # The values are only read when `depth_confidence > 0`, which every config
        # here sets to -1, so the contents are dead - but the SHAPE is not, and a
        # checkpoint whose shapes disagree with its config is one the framework
        # refuses.  Truncating it keeps the two in step.
        if hasattr(model, "confidence_thresholds"):
            model.confidence_thresholds.data = \
                model.confidence_thresholds.data[:depth].clone()
        note += f"layers {conf.n_layers}->{depth}"
        conf.n_layers = depth

    if heads is not None:
        raise NotImplementedError(
            "changing `num_heads` is NOT a weight truncation for this matcher. "
            "`SelfBlock.Wqkv` is (3*dim, dim) at every head count and its heads are "
            "interleaved as (h, d, 3), while `CrossBlock.to_qk`/`to_v`/`to_out` have "
            "identical shapes at 4 and 2 heads - so there is no correct slice. A "
            "checkpoint produced this way saves fine and then fails to load with "
            "100+ `size mismatch` errors. Retrain with `matcher.num_heads: 2` "
            "instead; the config builds the right shapes with no weight surgery.")

    if aggreg is not None:
        if aggreg != "uniform":
            raise ValueError(f"aggregation={aggreg!r}: only 'uniform' differs "
                             f"from the deployed 'last'")
        if conf.aggregation == "uniform":
            raise ValueError("aggregation is already 'uniform'")
        conf.aggregation = "uniform"
        note += (("; " if note else "") +
                 "aggregation 'last'->'uniform' (layer scales become a plain "
                 "mean; `token_confidence` weights stay but stop being used)")

    return note


def _slice_out(lin, keep):
    lin.weight.data = lin.weight.data[:keep].contiguous()
    if lin.bias is not None:
        lin.bias.data = lin.bias.data[:keep].contiguous()
    lin.out_features = keep


def _slice_in(lin, keep):
    lin.weight.data = lin.weight.data[:, :keep].contiguous()
    lin.in_features = keep


def slice_attention_heads(model, heads):
    """Physically slice every attention projection down to `heads` heads.

    WHY SLICING HEADS IS A REAL ABLATION AND SLICING `width` IS NOT
    --------------------------------------------------------------
    Multi-head attention splits the projections into `heads` independent blocks
    of `head_dim` and concatenates them; only the output projection mixes them
    back.  Keeping the first `heads` blocks leaves every SURVIVING head computing
    exactly what it computed before - the retained rows of `Wqkv`/`to_qk`/`to_v`
    and the retained columns of `out_proj`/`to_out` still line up.  The result is
    a genuine smaller member of the same family, so its score means something.

    Cutting `descriptor_dim` has no such property: the surviving rows of `Wqkv`
    would be scored against an `out_proj` trained to expect all 256 features.
    That is damage, not ablation.

    THE LAYOUT IS NOT UNIFORM, AND GETTING IT WRONG IS SILENT
    ---------------------------------------------------------
    Each projection splits `dim` differently, and the split decides which way it
    is cut:

      `SelfBlock.Wqkv`    (3*dim, dim)  `unflatten(-1, (h, -1, 3))` -> the LAST
                                        axis is q/k/v, so head `h` owns a CONTIGUOUS
                                        run of `3*head_dim` rows, not 3 separate
                                        runs.  Cut the output with `3*h*head_dim`.
      `CrossBlock.to_qk`  (dim, dim)    `unflatten(-1, (h, -1))` -> plain per-head
                                        blocks; cut the output with `h*head_dim`.
      `CrossBlock.to_v`   (dim, dim)    same; cut the output with `h*head_dim`.
      `out_proj`/`to_out` (dim, dim)    consume the concatenated heads; cut the
                                        INPUT with `h*head_dim`.

    An earlier version of this function cut `to_qk` at `2*h*head_dim`, by analogy
    with qkv, which left a 32-against-64 mismatch and a hard crash.  It also
    keyed off `m.n_heads` and `m.qkv`, neither of which exists on these blocks
    (`SelfBlock` uses `num_heads`/`Wqkv`, `CrossBlock` uses `heads`/`to_qk`), so
    it silently sliced NOTHING and the sweep reported a head count that had never
    been applied.  `n_matched` exists to make that outcome impossible to mistake
    for a result.

    Returns `(n_sliced, n_matched)`.  `n_matched == 0` means no attention module
    was recognised and the ablation is a no-op.

    THIS IS NOT A VALID ABLATION, AND THE SHAPES SAY SO
    ---------------------------------------------------
    It assumes each head owns a contiguous block on both sides of every projection.
    That is wrong for this matcher in BOTH directions, and the two errors happen to
    cancel in the parameter COUNT while the tensors they produce are ones the
    training framework refuses to load:

      `SelfBlock.Wqkv`     is `nn.Linear(embed_dim, 3 * embed_dim)` - its width
                           does NOT depend on `num_heads` at all.  The heads are
                           recovered in `forward` by
                           `qkv.unflatten(-1, (num_heads, -1, 3))`, so each head's
                           q, k and v are the three INTERLEAVED values of a
                           contiguous `head_dim * 3` run.  Keeping a prefix of
                           `3*heads*hd` rows does not keep a set of whole heads, and
                           no reordering of that buffer makes it one.
      `CrossBlock.to_qk`   is `nn.Linear(embed_dim, dim_head * num_heads)`, and
      `to_v` / `to_out`    `dim_head * num_heads == embed_dim` for every valid head
                           count.  Their SHAPES are therefore identical at 4 and at
                           2 heads, and slicing them is what CREATES the mismatch:
                           the framework builds `[256, 256]` from the config and this
                           function produces `[256, 128]`.

    Measured consequence: a `heads=2` checkpoint saves cleanly at 82.6 % of the
    parameters and then fails to load with
    `size mismatch for matcher.posenc.Wr.weight: copying a param with shape [32, 2]
    ... the shape in current model is [64, 2]` and 100+ more.

    The parameter count was never evidence, because the errors roughly cancel - 45
    "slices" totalling ~2.07 M.  A count that agrees for the wrong reason is what
    made the earlier version of this function look correct.

    KEPT RATHER THAN DELETED, because the `heads 4->2` rows in the ablation table
    came from it and deleting it would orphan them.  Those rows are marked
    unreproducible; the sound way to measure that configuration is a retrain with
    `num_heads: 2`, which needs no weight surgery.
    """
    n_sliced = 0
    n_matched = 0
    for m in model.modules():
        cls = type(m).__name__
        if cls == "SelfBlock":
            n_heads, head_dim = m.num_heads, m.head_dim
            if heads >= n_heads:
                continue
            n_matched += 1
            _slice_out(m.Wqkv, 3 * heads * head_dim)
            _slice_in(m.out_proj, heads * head_dim)
            m.num_heads = heads
            n_sliced += 2
        elif cls == "CrossBlock":
            n_heads = m.heads
            head_dim = m.to_qk.out_features // n_heads
            if heads >= n_heads:
                continue
            n_matched += 1
            _slice_out(m.to_qk, heads * head_dim)
            _slice_out(m.to_v, heads * head_dim)
            _slice_in(m.to_out, heads * head_dim)
            m.heads = heads
            n_sliced += 3
    return n_sliced, n_matched


def _probe_pair(n_kpt=256, n_cand=768, seed=0):
    """A non-degenerate matching problem.

    THE INPUT MATTERS MORE THAN THE CHECK
    -------------------------------------
    The first version of this probe used N random keypoints with N random
    unit-norm descriptors and got `idx_diff = 0` at every depth - which looked
    like a broken ablation and was actually a broken INPUT.  Two effects compound:

    * The descriptors are the same tensor for both images, so the true
      correspondence is exactly each point to itself.  Mutual nearest neighbour
      is therefore trivially correct and NO transformer layer is needed to find
      it.
    * A permutation matrix is the identity of the assignment problem, so it is
      the fixed point of every layer.  Removing layers cannot move it.

    So a degenerate probe reports "no change" for any ablation, and a check built
    on it rejects every row.  This version makes the problem real: `n_cand`
    candidate keypoints per image against `n_kpt` source points, so the
    assignment is a nontrivial argmax, and the descriptors of the two images are
    the same points in a DIFFERENT ORDER with noise, so a correct match is not
    the identity.
    """
    g = torch.Generator().manual_seed(seed)
    k0 = torch.rand(1, n_kpt, 2, generator=g) * 512
    # image 1 shares the first `n_kpt` points (permuted) plus `n_cand - n_kpt`
    # decoys, which is what makes finding the right partner require work.
    perm = torch.randperm(n_kpt, generator=g)
    k1 = torch.cat([k0[:, perm], torch.rand(1, n_cand - n_kpt, 2, generator=g) * 512], 1)
    d0 = torch.randn(1, n_kpt, 128, generator=g)
    d1 = torch.randn(1, n_cand, 128, generator=g)
    d0 = d0 / d0.norm(dim=-1, keepdim=True)
    d1 = d1 / d1.norm(dim=-1, keepdim=True)
    # make the permuted partner the genuinely best match, with some signal
    d1[:, :n_kpt] = 0.85 * d0[:, perm] + 0.15 * d1[:, :n_kpt]
    d1 = d1 / d1.norm(dim=-1, keepdim=True)
    return k0, k1, d0, d1


def prove_ablation_took_effect(candidate, baseline, seed=0):
    """Fail loudly if the candidate's output matches the baseline's.

    THE FAILURE THIS CATCHES, WHICH ALREADY HAPPENED ONCE
    -----------------------------------------------------
    A depth sweep reported `layers 9->8` and `layers 9->7` and every row produced
    byte-identical matches.  Two readings fit: the model changed and is
    insensitive, or the model did not change.  Identical output makes the second
    far more likely, and the run is worthless either way.

    So the check is not "did the code run" but "did the OUTPUT move".  It is
    measured on the SCORES rather than on the match indices, because the indices
    are thresholded and can coincide even when the underlying logits differ.
    """
    k0, k1, d0, d1 = _probe_pair(seed=seed)
    with torch.no_grad():
        m_a, s_a = candidate(k0, k1, d0, d1)
        m_b, s_b = baseline(k0, k1, d0, d1)
    if s_a.shape != s_b.shape:
        return True, "output shape changed"
    max_ds = float((s_a - s_b).abs().max())
    n_idx = int((m_a != m_b).sum())
    if max_ds == 0.0:
        return False, ("scores are bit-identical to the baseline - the ablation "
                       "did not take effect")
    return True, (f"max |Δscore| = {max_ds:.4f}, {n_idx}/{m_a.numel()} "
                  f"assignments moved")


def evaluate(pipe, hp, size, tag):
    out = {}
    if hp:
        print(f"    [{tag}] HPatches {len(hp)} pairs")
        out["hpatches"] = RA.run_hpatches(pipe, hp, verbose_every=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--hpatches", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these labels from the grid")
    ap.add_argument("--report", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-identical", action="store_true",
                    help="accept an aggregation row even if its output matches the "
                         "baseline; only meaningful for aggregation, where a "
                         "learned scale can legitimately equal a mean")
    args = ap.parse_args()

    hp = RA.load_hpatches(args.hpatches, args.size) if args.hpatches else []
    print(f"[ablate] {len(hp)} HPatches pairs, "
          f"{args.size}px / {args.keypoints} kpts")
    print("[ablate] truncation gives a LOWER BOUND on the config's accuracy - "
          "the survivors were trained with the removed layers present, so a good "
          "truncated number is conclusive and a bad one is not.\n")

    grid = GRID
    if args.only:
        want = set(args.only)
        grid = [g for g in GRID if g[0] in want]
        missing = want - {g[0] for g in grid}
        if missing:
            ap.error(f"unknown label(s): {sorted(missing)}")

    # Stage 1 is IDENTICAL in every row - it is the same detector feeding a
    # different matcher.  Building it once and reusing it is not just an
    # optimisation: it guarantees the detector's keypoints are bit-identical
    # across rows, so a difference in the matcher's score cannot be a difference
    # in what it was given.
    s1 = RA.load_alike_stage(args.checkpoint, top_k=args.keypoints,
                             descriptor_interp="bilinear").eval()
    base_s2 = RA.load_lightglue_stage(args.checkpoint,
                                      image_size=(args.size, args.size)).eval()
    n_base = param_count(base_s2)
    print(f"[ablate] matcher baseline: {n_base/1e6:.3f} M params, "
          f"layers={base_s2.matcher.conf.n_layers}, "
          f"heads={base_s2.matcher.conf.num_heads}, "
          f"dim={base_s2.matcher.conf.descriptor_dim}, "
          f"agg={base_s2.matcher.conf.aggregation}\n")

    rows = []
    for label, cfg in grid:
        print(f"[ablate] === {label}  ({describe(cfg)}) ===")
        s2 = copy.deepcopy(base_s2)
        try:
            note = apply_ablation(s2.matcher, dict(cfg))
            heads = cfg.get("heads")
            if heads:
                # `conf.num_heads` has already been lowered, but the projection
                # TENSORS are still full width - the truncation is not real until
                # they are physically cut.  A model with `num_heads=2` and 4-head
                # projections still computes 4 heads' worth of work and answers
                # a question about a model that does not exist.
                n_sliced, n_matched = slice_attention_heads(s2.matcher, heads)
                if n_sliced == 0:
                    raise RuntimeError(
                        f"heads={heads} sliced nothing ({n_matched} attention "
                        f"module(s) matched) - the projection attribute names do "
                        f"not exist on this matcher, so the ablation would be a "
                        f"no-op reported as a real result")
                note += (f"; sliced {n_sliced} projection(s) across {n_matched} "
                         f"attention module(s)")
        except Exception as exc:                       # noqa: BLE001
            print(f"    !! cannot build this config: {type(exc).__name__}: {exc}")
            rows.append({"label": label, "config": describe(cfg),
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        if note:
            print(f"    changed: {note}")

        # An ablation that did not change the output is not a result, it is a
        # baseline run with a different label.  Verify before spending 480 pairs
        # of evaluation on it.
        if cfg and label != "baseline":
            took, why = prove_ablation_took_effect(s2, base_s2)
            allowed = args.allow_identical and "aggregation" in cfg
            if not took and not allowed:
                print(f"    !! REJECTED: {why}")
                rows.append({"label": label, "config": describe(cfg),
                             "error": f"ablation did not take effect: {why}"})
                continue
            print(f"    verified: {why}"
                  + ("  (accepted on --allow-identical)" if not took else ""))

        pipe = RA.TorchPipeline.__new__(RA.TorchPipeline)
        pipe.stage1, pipe.stage2 = s1, s2
        res = evaluate(pipe, hp, args.size, label)
        n = param_count(s2)
        row = {"label": label, "config": describe(cfg),
               "params_M": round(n / 1e6, 4),
               "params_ratio": round(n / n_base, 4),
               **res}
        rows.append(row)
        print(f"    params {n/1e6:.3f} M ({100*n/n_base:.1f}% of baseline)")

    # ---- compare against the baseline row --------------------------------- #
    base = next((r for r in rows if r["label"] == "baseline"), None)
    # `n_inl_gt` is deterministic (counted against the ground-truth homography);
    # `n_inl` comes from RANSAC and carries an RNG-order offset large enough
    # (~13 inliers on 59 pairs) to swamp the effects this sweep is looking for.
    hp_keys = ["mAA", "@3px", "@5px", "mprec@3px", "n_inl_gt", "n_inl"]

    if base and "error" not in base:
        print("\n" + "=" * 100)
        print("MATCHER ABLATIONS  (delta vs baseline; negative is worse)")
        print("=" * 100)
        for judge, keys in (("hpatches", hp_keys),):
            if not base.get(judge):
                continue
            print(f"\n-- {judge} --")
            print(f"{'label':22s}{'params%':>9s}" +
                  "".join(f"{k:>13s}" for k in keys))
            for r in rows:
                if "error" in r or not r.get(judge):
                    continue
                line = f"{r['label']:22s}{100*r['params_ratio']:8.1f}%"
                for k in keys:
                    b, v = base[judge].get(k), r[judge].get(k)
                    if not isinstance(b, float) or not isinstance(v, float):
                        line += f"{'  -  ':>13s}"
                        continue
                    d = v - b
                    mark = "" if abs(d) < 0.002 else ("-" if d < 0 else "+")
                    line += f"{f'{d:+.4f}{mark}':>13s}"
                print(line)
        print("\nReading: `n_inl` and `n_inl@3` are counts, not ratios - a config "
              "that drops 30 inliers out of 280 has lost 10% of its usable "
              "correspondences even though `mAA` barely moves.")
        print("`rep@k` is a DETECTOR metric and must be IDENTICAL in every row; "
              "if it is not, the stage-1 reuse is broken and no other number in "
              "that row is trustworthy.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"baseline_params_M": round(n_base / 1e6, 4),
             "hpatches_pairs": len(hp),
             "size": args.size, "keypoints": args.keypoints,
             # Recorded, but repo-relative: the committed report must not carry
             # the absolute path this happened to be run from.
             "checkpoint": portable(args.checkpoint, "checkpoint"),
             "rows": rows}, indent=2))
        print(f"\n[ablate] raw numbers -> {args.json}")

    if args.report:
        write_report(Path(args.report), args, rows, base, hp_keys, n_base)
        print(f"[ablate] report -> {args.report}")
    return 0


def write_report(path, args, rows, base, hp_keys, n_base):
    L = [
        "# Matcher structural ablations",
        "",
        f"Trained weights truncated, not retrained. `{args.size}px`, "
        f"`{args.keypoints}` keypoints, baseline matcher "
        f"{n_base/1e6:.3f} M params.",
        f"Open-source HPatches viewpoint sequences: {args.hpatches} pairs, "
        f"seed {args.seed}.",
        "",
        "**Every number is a lower bound.** The surviving layers were trained "
        "with the removed ones present, so a config that scores well truncated is "
        "a safe choice and a config that scores badly may still be fine after a "
        "retrain. The reverse reading is not valid.",
        "",
    ]
    if base and "error" not in base:
        for judge, keys in (("hpatches", hp_keys),):
            if not base.get(judge):
                continue
            L += [f"## {judge} (delta vs baseline)", "",
                  "| config | params | " + " | ".join(f"`{k}`" for k in keys) + " |",
                  "|---|---|" + "---|" * len(keys)]
            for r in rows:
                if "error" in r or not r.get(judge):
                    continue
                cells = []
                for k in keys:
                    b, v = base[judge].get(k), r[judge].get(k)
                    cells.append("  -  " if not isinstance(b, float)
                                 or not isinstance(v, float)
                                 else f"{v - b:+.4f}")
                L.append(f"| `{r['label']}` | {100*r['params_ratio']:.1f}% | "
                         + " | ".join(cells) + " |")
            L.append("")
    L += ["## Failures", ""]
    errs = [r for r in rows if "error" in r]
    L += [f"* `{r['label']}` ({r['config']}): {r['error']}" for r in errs] or \
         ["* none"]
    L += ["",
          "## Which lever this says to pull", "",
          "Read `n_inl` before `mAA`. A matcher that keeps 95% of its inliers is "
          "still usable; one that drops to 60% is not, and `mAA` can stay flat "
          "through both because the surviving matches are still consistent with "
          "the ground truth. On a table that size, `n_inl_gt` (deterministic) is "
          "the sharpest signal and `n_inl` (RANSAC) the noisiest.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
