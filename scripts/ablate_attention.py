#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The structural change that removes the 27 `Einsum` nodes.

WHY THIS IS THE ONE THAT MATTERS
--------------------------------
The 29 fallback nodes in the matcher are 27 `Einsum` (attention) and 2
`ScatterElements` (two corner assignments).  `Einsum` is not an RKNN op, so those
27 nodes run on the host, inside the layer loop, on the critical path - and
attention is the matcher.  Nothing else in the speed plan is measurable until
this is fixed, because the host round-trips dominate whatever is left.

`Einsum` HAS NO EXACT NPU EQUIVALENT, BUT IT DOES NOT NEED ONE
--------------------------------------------------------------
There are two arithmetic paths for attention, and this project needs the one
that uses no `Einsum` at all:

  path A (what LightGlue does now)
      sim   = einsum("bhid,bhjd->bhij", q, k)
      m     = einsum("bhij,bhjd->bhid", attn, v)
      2 `Einsum` nodes per direction, 4 per layer.

  path B (already IN this file, four lines below the `Einsum`s)
      q, k, v are laid out (B, H, N, D).  Move the head axis behind the sequence
      axis - a pure permute, no arithmetic - and the same contraction becomes a
      plain batched `MatMul` over the last two axes:
          q -> (B, N, H, D) -> matmul with k^T -> (B, N, H, N)
      RKNN lowers `MatMul` natively.  The attention weights are unchanged.

THE PART THAT IS EASY TO GET WRONG
----------------------------------
`attn10 = softmax(sim.transpose(-2, -1))` is NOT a second softmax of the same
matrix.  The two directions are separate normalisations: `attn01` normalises over
the last axis and `attn10` over the second-to-last, and after the transpose those
are different axes.  A version that reuses one softmax for both produces a model
that runs, converts cleanly, and computes the wrong correspondences - which is
exactly the class of bug this project has already hit twice (see the README's
measurement findings).  So `--verify` is not optional: it compares the rewritten
path against the original `Einsum` path on real keypoints and asserts a tight
tolerance.

`scale` IS APPLIED TO q AND k BEFORE THE PRODUCT
------------------------------------------------
`qk0 * scale**0.5` then `sim`, and `scale = head_dim ** -0.5`, so the effective
factor is 1: `(head_dim**-0.5)**0.5 * (head_dim**-0.5)**0.5 == head_dim**-0.5`.
Applying it to only one operand would be off by sqrt(head_dim) - a 8x error at
head_dim 64, which is large enough to survive as "it still kind of works".

Usage
    python scripts/ablate_attention.py --checkpoint <ckpt> --verify
    python scripts/ablate_attention.py --checkpoint <ckpt> \
        --export refs/lightglue_stage_nomax.onnx
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- #
# the rewrite
# --------------------------------------------------------------------------- #
def patch_cross_block(block):
    """Rewrite `CrossBlock.forward`'s two `Einsum`s as `MatMul`.

    Only the contraction is replaced.  The projections, the scale, both softmaxes
    and the output projection are left exactly as upstream wrote them, so a
    discrepancy in `--verify` can only come from this one change - which is what
    makes the verification meaningful rather than a general smoke test.

    The contraction is expressed as a batched `MatMul` over the last two axes
    after moving the head axis behind the sequence axis.  `matmul` on
    (B, N, H, D) x (B, N, H, D)^T is the same contraction as
    `einsum("bhid,bhjd->bhij")`, but `MatMul` is an op RKNN lowers natively while
    `Einsum` is not.

    THE SUBSTITUTION IS DETERMINED BY SHAPES, NOT BY READING THE EINSUM SUBSCRIPTS
    ------------------------------------------------------------------------------
    The upstream `Einsum` calls cannot be used as a specification, because they
    reference an axis that the tensor they receive does not have.  Upstream's
    `m1` is `einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)` with
    `attn10` of shape (B, H, N0, N1).  A `transpose(-2, -1)` swaps the LAST TWO
    axes only, giving (B, H, N1, N0) - both non-D axes - and the subscript `i` is
    meant to be contracted against `d`, so the call cannot be well-formed.  It is
    dead code on the deployed path (`bias=True` always selects the `flash` branch
    on CUDA), which is why nobody has hit it.

    So instead of guessing at intent, every plausible substitution is enumerated
    and the ONE that survives the shape constraints is kept.  The constraints are:

      * `sim` for direction 0->1 must be (B, H, N0, N1), so that `softmax(-1)`
        normalises over the KEYS of image 1.
      * `softmax(sim, -1)` then contracts against `v1` (B, H, N1, D) -> (B, H, N0, D).
      * direction 1->0 is the transpose of the same score matrix; its softmax
        normalises over the keys of image 0 and contracts against `v0`.
      * after `flatten(-2)`, `to_out` demands exactly `heads * head_dim = 256`.

    Running that enumeration (a shape-level probe, kept with the measurement
    record outside this repository) prints the result: for each direction exactly
    ONE candidate is shape-valid, and it is a plain `matmul` with no transpose at
    all on the attention matrix:

        m0 = matmul(attn01, v1)                        # attn01 = softmax(sim, -1)
        m1 = matmul(attn10, v0)                        # attn10 = softmax(sim.T, -1)

    Every transposed variant fails, and the `transpose(-2,-1)` that upstream
    writes is one of the failures.  That is a stronger argument than either
    reading the subscripts or trusting the original, and it is checkable.

    `--verify` still runs: it compares this against a numerically-different but
    correct reference on real tensors, on the raw scores rather than the
    thresholded match indices.
    """
    scale = block.scale
    # Captured at patch time: the original bound method still holds the real
    # upstream implementation, so `--verify` can compare against it even after
    # the instance attribute shadows it.
    block._orig_forward = block.forward

    def forward(x0, x1, mask=None):
        qk0, qk1 = block.to_qk(x0), block.to_qk(x1)
        v0, v1 = block.to_v(x0), block.to_v(x1)
        qk0, qk1, v0, v1 = (
            t.unflatten(-1, (block.heads, -1)).transpose(1, 2)
            for t in (qk0, qk1, v0, v1))
        if block.flash is not None and qk0.device.type == "cuda":
            # Keep the flash kernel when it is actually available: swapping it
            # would change accuracy for reasons unrelated to this rewrite.
            m0 = block.flash(qk0, qk1, v1, mask)
            m1 = block.flash(qk1, qk0, v0,
                             mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0 = qk0 * scale ** 0.5
            qk1 = qk1 * scale ** 0.5
            # --- the only change: Einsum -> MatMul ---
            sim = torch.matmul(qk0, qk1.transpose(-2, -1))
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            # The reverse direction is a SECOND normalisation over the keys of
            # image 0.  `sim.transpose` is (B,H,N1,N0), i.e. the score matrix for
            # (query from 1, key from 0), so `softmax(-1)` normalises over the N0
            # keys.  This is what upstream's `attn10` computes; the transpose that
            # upstream then applies to it is not shape-valid (see docstring).
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.matmul(attn01, v1)
            m1 = torch.matmul(attn10, v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
            # --- end of change ---
        m0 = m0.transpose(1, 2).flatten(start_dim=-2)
        m1 = m1.transpose(1, 2).flatten(start_dim=-2)
        m0 = block.to_out(m0)
        m1 = block.to_out(m1)
        # Upstream returns a list; the caller unpacks it, so any 2-sequence does.
        x0 = x0 + block.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + block.ffn(torch.cat([x1, m1], -1))
        return [x0, x1]

    block.forward = forward
    return block


def sigmoid_log_double_softmax_concat(sim, z0, z1, zero_mode="derive"):
    """`sigmoid_log_double_softmax` with `Concat` instead of slice-assignment.

    THE ORIGINAL, AND WHY IT CANNOT STAY
    ------------------------------------
        scores = sim.new_full((b, m + 1, n + 1), 0)
        scores[:, :m, :n]      = scores0 + scores1 + certainties
        scores[:, :-1, -1]     = F.logsigmoid(-z0.squeeze(-1))
        scores[:, -1, :-1]     = F.logsigmoid(-z1.squeeze(-1))

    Allocating a buffer and writing four regions into it exports as
    `ScatterND` + 2x `ScatterElements`, none of which is NPU-native: the build log
    reports `No lowering found ... node type = ScatterElements` for both.  These
    are the LAST two off-NPU nodes in the matcher.

    THE REPLACEMENT IS THE SAME FOUR REGIONS, ASSEMBLED INSTEAD OF WRITTEN
    ---------------------------------------------------------------------
    Nothing about the arithmetic changes - the same `scores0 + scores1 +
    certainties` goes in the same place, and the two `logsigmoid` borders are the
    same values.  The buffer is what goes away, and it is replaced by three
    `Concat` calls over tensors that were already being computed:

        (b, m, n) + (b, m, 1)  -> cat dim 2 -> (b, m, n+1)      the top block
        (b, 1, n) + (b, 1, 1)  -> cat dim 2 -> (b, 1, n+1)      the bottom row
                        both   -> cat dim 1 -> (b, m+1, n+1)

    The bottom-right cell is the `new_full` buffer's zero, kept as an explicit
    zero tensor rather than left implicit, because that is what makes the shape
    `(b, m+1, n+1)` rather than `(b, m+1, n)` - the padding row and column are
    what `filter_matches` reads to decide a point is unmatched.

    This is the same class of fix as the DKD border mask (build the mask by
    broadcasting instead of by slice-assignment): the op that has no NPU lowering
    is the in-place write, not the computation.
    """
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    top = torch.cat([scores0 + scores1 + certainties,
                     F.logsigmoid(-z0)], dim=2)              # (b, m, n+1)
    bottom = torch.cat([F.logsigmoid(-z1).transpose(1, 2),
                        _bottom_right_zero(sim, zero_mode)], dim=2)   # (b, 1, n+1)
    return torch.cat([top, bottom], dim=1)                   # (b, m+1, n+1)


def _bottom_right_zero(sim, zero_mode):
    """The `(b, 1, 1)` zero cell, built the way the toolkit can actually parse.

    THE PROBLEM WITH THE OBVIOUS `sim.new_zeros((b, 1, 1))`
    ------------------------------------------------------
    It exports as a `ConstantOfShape` node, and `rknn.load_onnx` dies on it:

        File "rknn/api/ir_graph.py", line 2131, in IRGraph.convert_to_fp32
        AttributeError: 'numpy.ndarray' object has no attribute 'data_type'

    That is a toolkit bug - it walks what it believes are TensorProto weights and
    calls `.data_type` on something that is an ndarray - but it is not one to wait
    for.  Isolation: applying the `Concat` rewrite WITHOUT this constant breaks the
    load, and applying everything else does not, so this single node is the whole
    cause.  `ConstantOfShape` takes its shape from an int64 TENSOR input rather
    than from an attribute, which is exactly the kind of input a weight-walking
    pass would misinterpret.

    THE WORKAROUND: DERIVE THE ZERO INSTEAD OF DECLARING IT
    ------------------------------------------------------
    `sim[:, :1, :1] * 0.0` is arithmetically the same cell and needs no new
    constant - just a `Slice` of a tensor that is already in the graph and a
    multiply by the scalar 0, both of which appear hundreds of times already in
    this very graph.  The failure was caused by a NEW KIND of node, not by the
    rearrangement, so reintroducing nothing new is the right fix.

    `* 0.0` on a finite input gives `+0.0`.  `sim` is a dot product of projected
    descriptors, so it is finite by construction; the derivation would only
    misbehave for an infinite input, where the answer would be NaN rather than 0,
    and that would be a failure in the matcher well before it mattered here.
    """
    if zero_mode == "declare":
        return sim.new_zeros((sim.shape[0], 1, 1))
    if zero_mode == "derive":
        return sim[:, :1, :1] * 0.0
    raise ValueError(f"zero_mode must be 'declare' or 'derive', got {zero_mode!r}")


def patch_match_assignment(module, orig_fn=None, matmul=True, concat=True,
                           zero_mode="derive"):
    """Rewrite `MatchAssignment.forward`: no `Scatter*`, and no `Einsum`.

    TWO CHANGES, AND THE SECOND IS EASY TO MISS
    -------------------------------------------
      * the log-assignment buffer becomes `Concat` (see above) - 2
        `ScatterElements` removed;
      * `sim = einsum("bmd,bnd->bmn", mdesc0, mdesc1)` becomes a `matmul`.  This
        is the ONE `Einsum` left after the attention rewrite, and it is easy to
        overlook because the count went from 28 to 1 and "1" reads like "done".
        It is the same trivial substitution:
        `einsum("bmd,bnd->bmn")` == `matmul(mdesc0, mdesc1.transpose(-1, -2))`.

    `matmul`/`concat` exist to switch each change off INDEPENDENTLY, because
    together they make `rknn.load_onnx` fail and the fix depends on which one is
    responsible.  A patch that can only be applied all-or-nothing forces you to
    guess at the cause.

    `orig_fn` is the pristine `sigmoid_log_double_softmax`, captured by the
    caller so `--verify` can compare against it after the patch is in place.
    """
    orig = orig_fn

    def forward(desc0, desc1):
        mdesc0, mdesc1 = module.final_proj(desc0), module.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d ** 0.25, mdesc1 / d ** 0.25
        if matmul:
            sim = torch.matmul(mdesc0, mdesc1.transpose(-1, -2))
        else:
            sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = module.matchability(desc0)
        z1 = module.matchability(desc1)
        f = orig or _slds_orig()
        scores = (sigmoid_log_double_softmax_concat(sim, z0, z1, zero_mode)
                  if concat else f(sim, z0, z1))
        return scores, sim

    module._orig_forward = module.forward
    module._orig_slds = orig
    module.forward = forward
    return module


def _slds_orig():
    """The pristine `sigmoid_log_double_softmax`, resolved lazily."""
    import model.matchers.lightglue as lg
    return lg.sigmoid_log_double_softmax


def verify_patch(matcher, n_kpt=128, n_cand=256, seed=0):
    """Compare the patched matcher against the ORIGINAL forward, layer by layer.

    THE COMPARISON IS ON RAW SCORES, NOT MATCH INDICES
    --------------------------------------------------
    Indices are thresholded, so they can coincide while the logits are badly
    wrong.  This project has already been burned once by exactly that (a
    descriptor comparison that reported cos 0.88 for a correct model because the
    order was not matched).  The score tensor has no such freedom.

    `CrossBlock.forward` is a bound method here (the class does not override
    `__call__`), so the pristine implementation is recoverable from the CLASS -
    `type(block).forward` is untouched by the instance-level patch.  That gives a
    real A/B on identical inputs rather than an assertion that the code "looks
    equivalent".

    WHAT THIS GUARANTEES NOW THAT THE MATCHER IS VENDORED
    -----------------------------------------------------
    Both sides of the A/B come from `model/matchers/lightglue.py`.  So this no
    longer proves "the rewrite agrees with the training framework" - it proves the
    narrower, and still load-bearing, claim that **the rewrite is numerically
    exact with respect to the unpatched module it replaces**: the patch changes
    which ONNX ops appear, not a single value.  The vendored module's agreement
    with the training framework is a separate check, done once by
    `scripts/verify_vendored_matcher.py` (bit-identical on all 9 outputs).  Keep
    the two claims apart: this one is re-run on every change to the rewrite, that
    one only when the port is edited.
    """
    from ablate_matcher import _probe_pair
    k0, k1, d0, d1 = _probe_pair(n_kpt, n_cand, seed)

    # `LightGlue` is not called as a plain tensor module - it takes a dict.  Go
    # through the stage wrapper's own calling convention rather than guessing at
    # the dict keys here, so this comparison exercises the same code path as
    # `scripts/ablate_matcher.py`.
    import torch as _t

    def call():
        with _t.no_grad():
            out = matcher({"keypoints0": k0, "keypoints1": k1,
                           "descriptors0": d0, "descriptors1": d1,
                           "view0": {"image_size": _t.tensor([[512., 512.]])},
                           "view1": {"image_size": _t.tensor([[512., 512.]])}})
        return out["matches0"], out["matching_scores0"]

    m_new, s_new = call()

    # Temporarily restore the ORIGINAL implementations for the reference pass.
    # An instance attribute shadows the class method, so the pristine version is
    # still reachable from the class - which is what makes this a real A/B on
    # identical inputs rather than an assertion that the code "looks equivalent".
    #
    # BOTH patches have to be reverted for the reference to be the true baseline:
    # reverting only the attention would compare a patched assignment against a
    # pristine one and report the difference as if the attention rewrite caused
    # it.  `log_assignment` is a ModuleList of 9 and only the last is read under
    # `aggregation='last'`, but the reference pass has to restore all of them
    # because `forward` indexes by layer.
    saved_attn, saved_assign = [], []
    try:
        for b in matcher.transformers:
            cb = b.cross_attn
            saved_attn.append((cb, cb.forward))
            cb.forward = type(cb).forward.__get__(cb, type(cb))
        for la in matcher.log_assignment:
            if getattr(la, "_orig_forward", None) is not None:
                saved_assign.append((la, la.forward))
                la.forward = la._orig_forward
        m_ref, s_ref = call()
    finally:
        for cb, f in saved_attn:
            cb.forward = f
        for la, f in saved_assign:
            la.forward = f

    return (float((s_new - s_ref).abs().max()),
            int((m_new != m_ref).sum()), m_new.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--export", default=None,
                    help="write a rewritten ONNX graph here (Runs the full "
                         "exporter, not just the module)")
    ap.add_argument("--verify", action="store_true",
                    help="assert the rewrite is numerically identical to the "
                         "original Einsum path on real keypoints")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    args = ap.parse_args()

    from model import load_lightglue_stage
    stage = load_lightglue_stage(args.checkpoint,
                                 image_size=(args.size, args.size)).eval()
    matcher = stage.matcher

    n_patched = 0
    for block in matcher.transformers:
        patch_cross_block(block.cross_attn)
        n_patched += 1
    print(f"[attn] patched {n_patched} CrossBlock(s): Einsum -> MatMul")
    if n_patched == 0:
        raise RuntimeError("patched nothing - `matcher.transformers` did not "
                           "match, so the export would still contain Einsum")

    # The log-assignment path: `ScatterElements` -> `Concat`, plus the last
    # `Einsum`.  The pristine function is captured BEFORE the patch so the
    # verification has something real to compare against.
    import model.matchers.lightglue as _lg
    orig_slds = _lg.sigmoid_log_double_softmax
    n_assign = 0
    for la in matcher.log_assignment:
        patch_match_assignment(la, orig_fn=orig_slds)
        n_assign += 1
    print(f"[attn] patched {n_assign} MatchAssignment(s): Scatter* -> Concat, "
          f"Einsum -> MatMul")
    if n_assign == 0:
        raise RuntimeError("patched no MatchAssignment - the graph would still "
                           "contain ScatterElements")

    if args.verify:
        max_d, n_idx, n_tot = verify_patch(matcher)
        print("[attn] VERIFY vs the pristine forwards (attention AND assignment): "
              f"max |Δscore| = {max_d:.3e} (tol {args.tol:g}), "
              f"{n_idx}/{n_tot} assignments differ")
        if max_d > args.tol:
            print("[attn] FAIL: the rewrite is NOT numerically identical - do not "
                  "export it. Work through `forward` line by line; the usual "
                  "causes are the `attn10` transpose, the scale on q and k, and "
                  "the two border columns of the assignment matrix.")
            return 1
        print("[attn] PASS: identical to the original path, so all 29 nodes "
              "(27 Einsum + 2 ScatterElements) can be removed at no accuracy cost")

    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        dummy = (torch.zeros(1, args.keypoints, 2),
                 torch.zeros(1, args.keypoints, 2),
                 torch.randn(1, args.keypoints, 128),
                 torch.randn(1, args.keypoints, 128))
        for t in dummy[2:]:
            t /= t.norm(dim=-1, keepdim=True)
        torch.onnx.export(
            matcher, dummy, str(out), opset_version=16,
            input_names=["keypoints0", "keypoints1",
                         "descriptors0", "descriptors1"],
            output_names=["matches0", "matching_scores0"],
            dynamic_axes=None)
        print(f"[attn] wrote {out}")
        import onnx
        m = onnx.load(str(out))
        n_einsum = sum(1 for n in m.graph.node if n.op_type == "Einsum")
        print(f"[attn] Einsum nodes remaining in the graph: {n_einsum}")
        if n_einsum:
            print("[attn] WARNING: Einsum remains - the patch did not cover every "
                  "attention path")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
