#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Turn a truncation ablation into a checkpoint that a fine-tune can start from.

WHY THIS IS A SEPARATE STEP
---------------------------
`scripts/ablate_matcher.py` truncates a model IN MEMORY and scores it.  That is
the right shape for a sweep - it is fast and it touches no files - but it is the
wrong shape for a recovery run, which needs its starting point to exist as a
checkpoint on disk in the format the training framework loads.

There is a second, less obvious reason.  A truncation is not just "fewer layers":
`transformers`, `log_assignment` and `token_confidence` all have to be cut in
lockstep and `conf.n_layers` has to agree with them, or `forward` indexes a
shortened list and reads the wrong layer's logits without raising.  Re-deriving
that inside a training config would mean trusting the same code twice.  This
script applies the exact ablation from `ablate_matcher.py`, verifies it changed the
output, and freezes it to disk - so the run that starts from it starts from
something already checked.

WHY THE CONFIG IS REWRITTEN RATHER THAN THE WEIGHTS ALONE
--------------------------------------------------------
The framework rebuilds the model from `config.yaml`, so a checkpoint whose tensors
are 7 layers deep but whose config still says `n_layers: 9` will not load - or
worse, will load partially and train a model that is not the one that was measured.
`config.n_layers` is therefore updated in the same pass, and the two are checked
against each other before writing.

Usage
    python pruning/make_pruned_ckpt.py \
        --checkpoint /path/to/alike_native_gl_s1/checkpoint_best.tar \
        --depth 7 --out <experiment-dir>   # e.g. outputs/training/alike_native_gl_s1_d7
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--depth", type=int, default=None,
                    help="truncate the matcher to this many layers")
    ap.add_argument("--heads", type=int, default=None,
                    help="NOT SUPPORTED.  Changing the head count is not a weight "
                         "truncation for this matcher (see "
                         "`slice_attention_heads`); retrain with "
                         "`matcher.num_heads: N` instead.  Accepted only so the "
                         "refusal can explain itself.")
    ap.add_argument("--out", required=True,
                    help="output EXPERIMENT directory (config.yaml + checkpoint)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    from ablate_matcher import (apply_ablation, prove_ablation_took_effect,
                                slice_attention_heads)
    from model import load_lightglue_stage

    if args.depth is None and args.heads is None:
        ap.error("pass --depth and/or --heads; a run with neither would fine-tune "
                 "the baseline and report it as a pruning result")

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stage = load_lightglue_stage(str(ckpt), image_size=(args.size, args.size)).eval()
    before = sum(p.numel() for p in stage.parameters())

    cfg = {}
    if args.depth is not None:
        cfg["depth"] = args.depth
    if args.heads is not None:
        cfg["heads"] = args.heads
    if args.heads is not None:
        # Refuse BEFORE any weights are touched.  Producing an unloadable checkpoint
        # and finding out at fine-tune launch wastes the run and leaves a misleading
        # artifact on disk - which is exactly what happened once already.
        raise SystemExit(
            "`--heads` is not supported.  `SelfBlock.Wqkv` is (3*dim, dim) at every "
            "head count and packs (h, d, 3), while `CrossBlock.to_qk`/`to_v`/"
            "`to_out` have identical shapes at 4 and 2 heads - so there is no "
            "correct slice, and a sliced checkpoint fails to load with 100+ size "
            "mismatches.  The sound measurement is a retrain with "
            "`matcher.num_heads: N`; the config builds the right shapes and no "
            "weight surgery is needed.")
    note = apply_ablation(stage.matcher, cfg)
    if args.heads is not None:
        # `apply_ablation` only lowers `conf.num_heads`; the projection TENSORS
        # stay full width until they are physically cut.  Skipping this leaves a
        # model that CLAIMS 2 heads while still computing 4 - a no-op wearing a
        # label, which is exactly the bug the sweep hit on its first attempt.
        n_sliced, n_matched = slice_attention_heads(stage.matcher, args.heads)
        if n_sliced == 0:
            raise SystemExit(
                f"heads={args.heads} sliced nothing ({n_matched} attention "
                f"module(s) matched) - the projection attribute names do not "
                f"exist on this matcher, so the run would train the baseline")
        note += f"; sliced {n_sliced} projection(s) across {n_matched} module(s)"
    after = sum(p.numel() for p in stage.parameters())
    print(f"[prune] {note}: {before/1e6:.3f} M -> {after/1e6:.3f} M "
          f"({100*after/before:.1f} %)")

    if args.verify:
        # A truncation that did not change the output is not a result, and starting
        # a multi-hour fine-tune from one would waste the run.  Same check as the
        # sweep, for the same reason.
        full = load_lightglue_stage(str(ckpt),
                                    image_size=(args.size, args.size)).eval()
        took, why = prove_ablation_took_effect(stage, full)
        print(f"[prune] verified: {why}")
        if not took:
            raise SystemExit("the truncation did not take effect - not writing a "
                             "checkpoint for a run that would train the baseline")

    # Save in the layout the framework expects: a `checkpoint` entry holding the
    # state dict, next to the config it belongs to.
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    probe = raw
    state = raw.get("model", raw)

    # Build the target model FIRST, so the per-list lengths come from the framework
    # rather than from a hard-coded rule.  The framework builds `n` `transformers`,
    # `n` `log_assignment` but only `n - 1` `token_confidence`, and filtering all
    # three by a single `depth` keeps one surplus `TokenConfidence` for the third -
    # which then loads as `unexpected` under `strict=False`, silently, and leaves a
    # checkpoint whose param count is not the model's.
    depth = args.depth if args.depth is not None else 10 ** 9
    targets = _list_targets_from_model(stage.matcher, depth)
    print(f"[prune] target lengths at n_layers={depth if depth < 10**9 else 'unchanged'}: "
          f"{targets}")

    def _surplus(k):
        i = _layer_index(k)
        if i is None:
            return False
        for lst_name, limit in targets.items():
            if k.startswith(f"matcher.{lst_name}."):
                return i >= limit
        return False

    keep = {k: v for k, v in state.items() if not _surplus(k)}
    n_dropped = len(state) - len(keep)
    dropped_by_list = {}
    for k in state:
        if _surplus(k):
            for lst_name in targets:
                if k.startswith(f"matcher.{lst_name}."):
                    dropped_by_list[lst_name] = dropped_by_list.get(lst_name, 0) + 1
    print(f"[prune] state dict: {len(state)} tensors -> {len(keep)} "
          f"({n_dropped} dropped: {dropped_by_list})")

    # The head cut changed TENSOR SHAPES, so the packed values must be replaced by
    # the model's own sliced tensors.  Reusing the checkpoint's originals would
    # write a [256, 256] `to_qk` next to a config that says 2 heads - and
    # `_fix_shapes` cannot rescue it, because its rule is "truncate when LONGER",
    # which only fires on the leading dimension.  These three are square, so
    # `to_qk.weight` is [256, 64] after slicing and [256, 256] before: neither
    # longer nor shorter in the way that rule tests.
    #
    # Taking the values from `stage` is also what makes the checkpoint PROVABLY the
    # model that was just verified above, rather than a re-derivation of it.
    if args.heads is not None:
        replaced = _sync_sliced_tensors(keep, stage)
        if replaced == 0:
            raise SystemExit("heads was set but no tensor was replaced from the "
                             "sliced model - the checkpoint would carry the "
                             "ORIGINAL projections and the fine-tune would silently "
                             "start from 4 heads")

    # Load the packed dict into the truncated model so a shape mismatch is caught
    # HERE rather than several hours into a training run.
    keep = _fix_shapes(keep, stage)
    missing, unexpected = stage.load_state_dict(
        {_strip(k): v for k, v in keep.items()}, strict=False)
    problematic = [k for k in missing if "matcher" in k]
    if problematic:
        raise SystemExit(
            f"the truncated model is missing matcher tensors after loading: "
            f"{problematic[:5]} - the checkpoint packing does not match the model "
            f"and a fine-tune would start from a partly-random matcher")
    # Only the matcher's keys are expected to be unexpected: the stage wrapper for
    # this job holds the matcher, so the extractor's tensors are carried in the
    # checkpoint but are not loaded here.  A MATCHER key being unexpected is a real
    # problem - it means a slice or a layer cut produced a name the model does not
    # have, and the tensor would be silently absent from the fine-tune.
    unexpected_matcher = [k for k in unexpected if "matcher" in k]
    if unexpected_matcher:
        raise SystemExit(
            f"{len(unexpected_matcher)} matcher tensor(s) do not exist in the "
            f"pruned model: {unexpected_matcher[:5]} - the fine-tune would start "
            f"without them")
    if unexpected:
        print(f"[prune] note: {len(unexpected)} unexpected non-matcher key(s) "
              f"ignored (extractor weights, carried through unloaded)")

    # Round-trip the ORIGINAL nesting: the framework's `load_experiment` reads
    # `checkpoint["model"]` and expects the `matcher.`/`extractor.` prefixes, so
    # re-wrapping a dict that never had a `model.` prefix is correct and adding one
    # would break the load.  Asserted rather than assumed, because a wrong nesting
    # here surfaces as "no pretrained weights found" - which reads like a missing
    # file, not like a packing bug.
    # Carry EVERYTHING else through untouched.  The framework's `load_experiment`
    # reads `checkpoint["conf"]` to merge the training config, so a checkpoint that
    # contains only the weights fails at startup with a bare `KeyError: 'conf'` -
    # which names the field but not the fact that a whole sub-dict was dropped.
    packed = dict(probe)
    packed["model"] = keep
    # `conf` must describe the PRUNED architecture too, or the merged config
    # rebuilds 9 layers and the weights do not line up.
    if "conf" in packed:
        from omegaconf import OmegaConf
        c = OmegaConf.create(packed["conf"])
        try:
            # BOTH architecture knobs have to move, and `num_heads` is the one
            # that is easy to forget: a config still saying 4 heads rebuilds
            # 4-head projections, and the sliced tensors then fail to load with a
            # size mismatch in `to_qk`/`to_v`/`to_out`.
            changed = []
            if args.depth is not None:
                old = c.model.matcher.n_layers
                c.model.matcher.n_layers = args.depth
                changed.append(f"n_layers {old} -> {args.depth}")
            if args.heads is not None:
                old = c.model.matcher.get("num_heads")
                c.model.matcher.num_heads = args.heads
                changed.append(f"num_heads {old} -> {args.heads}")
            packed["conf"] = OmegaConf.to_container(c, resolve=True)
            print("[prune] checkpoint conf: " + "; ".join(changed))
        except Exception as exc:                       # noqa: BLE001
            raise SystemExit("could not set the pruned architecture in the "
                             f"checkpoint's conf ({type(exc).__name__}: {exc}); the "
                             f"framework would rebuild the unpruned model")
    else:
        raise SystemExit("the source checkpoint has no `conf` entry, which "
                         "`train.load_experiment` requires")
    a = next(iter(state))
    b = next(iter(keep))
    if not ((a.startswith("model.") and b.startswith("model."))
            or (not a.startswith("model.") and not b.startswith("model."))):
        raise SystemExit(f"nesting mismatch: source key {a!r} vs packed key {b!r}")
    # Name it `checkpoint_best.tar`: `train.load_experiment` looks for exactly that
    # filename, so a differently-named file is not a cosmetic difference - the run
    # dies at startup with a bare `FileNotFoundError` that names the path but not
    # the convention.
    for name in ("checkpoint_best.tar", "checkpoint_pruned.tar"):
        torch.save(packed, str(out / name))
    print("[prune] wrote checkpoint_best.tar + checkpoint_pruned.tar "
          f"({len(keep)} tensors, keys {list(packed)})")

    # The config must describe the pruned architecture, or the framework will
    # rebuild 9 layers and the checkpoint will not line up.
    src_cfg = ckpt.parent / "config.yaml"
    if src_cfg.is_file():
        import re
        text = src_cfg.read_text()
        edits = []
        # Only rewrite the knobs that were actually cut.  A single `subn` that
        # always writes both is how `n_layers:` became the literal string `None`
        # in a heads-only run: `re.subn(r"n_layers:\s*)\d+", ...None)` matches the
        # OLD value, so the replacement is `n_layers: None` - which the framework
        # then builds an enumeration over.  The failure is loud at load time, but
        # the checkpoint on disk is already wrong by then.
        if args.depth is not None:
            text, n = re.subn(r"(n_layers:\s*)\d+", rf"\g<1>{args.depth}", text)
            if n == 0:
                raise SystemExit(f"no `n_layers:` in {src_cfg}")
            edits.append(f"n_layers={args.depth}")
        if args.heads is not None:
            text, n = re.subn(r"(num_heads:\s*)\d+", rf"\g<1>{args.heads}", text)
            if n == 0:
                raise SystemExit(f"no `num_heads:` in {src_cfg} - the framework "
                                 f"would rebuild 4-head attention")
            edits.append(f"num_heads={args.heads}")
        (out / "config.yaml").write_text(text)
        print(f"[prune] config.yaml written with {', '.join(edits)}")
    else:
        raise SystemExit(f"{src_cfg} not found; the framework would rebuild the "
                         f"unpruned architecture")
    print(f"[prune] -> {out}")
    return 0


def _sync_sliced_tensors(keep, model):
    """Replace every packed tensor whose SHAPE the head cut changed.

    Generic on purpose: rather than naming `to_qk`, `to_v` and `to_out`, compare
    every key's shape against the sliced model and copy where they differ.  A named
    list would have to be kept in step with `_HEAD_PROJECTIONS` in
    `ablate_matcher.py`, and the failure mode of them drifting apart is a
    checkpoint that mixes sliced and unsliced tensors - which loads without error
    whenever the shapes happen to agree and silently trains the wrong model when
    they do not.

    Returns the number of tensors replaced, so a call that changed nothing is
    caught by the caller instead of producing a checkpoint labelled `heads=2` that
    contains 4-head weights.
    """
    target = dict(model.state_dict())
    n = 0
    for k, v in list(keep.items()):
        want = target.get(_strip(k))
        if want is not None and tuple(v.shape) != tuple(want.shape):
            keep[k] = want.detach().clone()
            n += 1
    if n:
        print(f"[prune] replaced {n} tensor(s) with the sliced model's values "
              f"(shapes the head cut changed)")
    return n


def _fix_shapes(keep, model):
    """Reconcile every packed tensor against the TRUNCATED model's expectations.

    THE FILTER ABOVE IS BY NAME, AND ONE TENSOR IS NOT NAMED BY LAYER
    -----------------------------------------------------------------
    Cutting `transformers`/`log_assignment`/`token_confidence` removes the layer
    lists, but `matcher.confidence_thresholds` is a length-`n_layers` PARAMETER
    that carries no layer index in its name.  It stays at `[9]` while the model
    built from the pruned config expects `[7]`, and the failure is:

        size mismatch for matcher.confidence_thresholds: copying a param with
        shape [9] from checkpoint, the shape in current model is [7]

    which surfaces only when the framework loads the checkpoint - i.e. at the start
    of a training run, after the checkpoint has already been written and after the
    ablation has already been scored.

    So rather than patch this one name, every tensor whose leading dimension is
    longer than the model's is truncated to match.  A generic rule catches the next
    per-layer parameter without anyone having to remember it exists, and anything it
    truncates is reported, so an unexpected hit is visible instead of silent.
    """
    fixed, adjusted = {}, []
    target = dict(model.state_dict())
    for k, v in keep.items():
        want = target.get(_strip(k))
        if want is not None and tuple(v.shape) != tuple(want.shape) \
                and v.shape[0] > want.shape[0]:
            v = v[: want.shape[0]].clone()
            adjusted.append(f"{_strip(k)} {tuple(target[_strip(k)].shape)}"
                            f"->{tuple(v.shape)}")
        fixed[k] = v
    if adjusted:
        print(f"[prune] truncated {len(adjusted)} per-layer parameter(s) to match "
              f"the pruned model: {adjusted}")
    return fixed


def _list_targets_from_model(matcher, depth):
    """Per-list truncation targets, read from the model rather than assumed.

    `LightGlue.__init__` builds the three per-layer ModuleLists with DIFFERENT
    lengths (`transformers` and `log_assignment` from `range(n)`, `token_confidence`
    from `range(n - 1)`), and that `- 1` is the whole reason this helper exists.  A
    single shared `depth` applied to all three produces a checkpoint with one
    surplus `TokenConfidence` block, and because the fine-tune loads with
    `strict=False` that block is merely reported as `unexpected` - so the run
    proceeds while the checkpoint no longer matches the model it is supposed to be.

    The targets are taken from the truncated model's own list lengths, so if the
    framework ever changes the `- 1` this follows automatically instead of silently
    disagreeing.  `depth` is clamped to what the model actually has, so asking for
    more layers than exist is a no-op rather than a ValueError here.
    """
    out = {}
    for name in ("transformers", "log_assignment", "token_confidence"):
        lst = getattr(matcher, name, None)
        if lst is not None:
            out[name] = min(depth, len(lst))
    return out


def _layer_index(key):
    """The trailing layer index of a `matcher.<list>...<i>...` key, or None."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part in ("transformers", "log_assignment", "token_confidence"):
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
    return None


def _strip(key):
    """`matcher.x` as-is, for loading into the stage wrapper.

    The packed state dict uses `matcher.`/`extractor.` prefixes and the stage
    wrapper expects exactly those (`self.matcher`, `self.extractor`), so nothing is
    stripped in this project's checkpoint layout.  The `model.` case is handled for
    a differently-nested checkpoint, but the round-trip above is what actually
    validates it.
    """
    return key[len("model."):] if key.startswith("model.") else key


if __name__ == "__main__":
    sys.exit(main())
