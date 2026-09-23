#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accuracy recovery after structured pruning.

Pruning zeroes weights that were doing work, so the model is no longer in a
trained basin and a short fine-tune is what brings it back.  This script uses the
training recipe that was validated for this project rather than inventing a new
one - deviating from it here would confound "did pruning cost accuracy" with "did
the recovery schedule cost accuracy".

The recipe, and where each setting comes from
--------------------------------------------
    matcher only, extractor frozen   the extractor is bit-identical under
                                     `trainable=False` + the BN pin, so freezing
                                     it makes the comparison exact
    lr 1e-4, flat, no `lr_scaling`   flat was measured better than the split; the
                                     split is a SUBSTRING match on parameter names
                                     and silently drags `desc_head` to 1e-5
    epochs 1                         three epochs measured WORSE than one on this
                                     metric, and the third was the worst of the
                                     three, so "train longer" is not a safe default
    seed 0, 512 keypoints            same as every recorded reference number

Acceptance is NOT the training loss.  The in-loop validation saturates near 0.98
for every model and cannot rank arms; what decides is `eval/run_accuracy.py`
re-run end-to-end, with no metric below its line.

THE ONE SCRIPT THAT STILL NEEDS THE TRAINING REPO
-------------------------------------------------
Everything else here is self-contained: export, refinement, conversion and the
accuracy harness all run off `model/` and `weights/`.  This one cannot be, and the
reason is not a missing port - it *launches a training run*:

    python -m gluefactory.train <experiment>

The data pipeline, augmentation, optimizer construction and schedule that the
recipe above names all live in the training framework, and the point of the
recipe is to reuse the validated ones rather than reimplement them.  So this
script finds the checkout (see `gluefactory_root`) and runs it.  If you only need
to deploy the models, you never invoke this file.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402

DEFAULT_RECIPE = {
    "model.extractor.trainable": "False",
    "model.extractor.freeze_batch_normalization": "True",
    "model.extractor.max_num_keypoints": "512",
    "model.extractor.force_num_keypoints": "True",
    "model.matcher.input_dim": "128",
    "model.matcher.descriptor_dim": "256",
    "model.matcher.filter_threshold": "0.0",
    "model.matcher.depth_confidence": "-1",
    "model.matcher.width_confidence": "-1",
    "train.lr": "1e-4",
    "train.epochs": "1",
    "train.seed": "0",
}


def gluefactory_root() -> Path:
    """Locate the training framework, the same way the exporter does.

    `python -m gluefactory.train` only resolves if the CWD is the repository root,
    and this script used to run the subprocess from wherever the caller happened to
    be - so the documented usage (`python pruning/finetune.py ...` from the RKNN
    project) died with a bare `No module named 'gluefactory'`.
    """
    chosen = paths.glovefactory()
    if chosen is None:
        raise SystemExit(
            "cannot find the glue-factory checkout; set GLUEFACTORY_ROOT.  "
            "Nothing else in this repository needs it - only this script, which "
            "launches a training run.")
    return chosen


def check_architecture(args) -> None:
    """Refuse a config whose architecture does not match the checkpoint.

    WHY THIS IS NOT OPTIONAL
    ------------------------
    `train.load_experiment` merges `conf.model = merge(ckpt_conf.model, cli_conf)`,
    so the `--conf` WINS.  Passing the original config next to a pruned checkpoint
    therefore builds the ORIGINAL architecture and then loads the pruned weights
    into it with `strict=False` - which tolerates MISSING keys, leaving the extra
    layers at their random initialisation.  The run trains happily, the loss curve
    looks healthy, and the result is a model with randomly-initialised layers.

    That is not hypothetical: it happened, and the run was caught only by reading
    the saved `n_layers` afterwards.

    A shape mismatch IS fatal even under `strict=False` (`confidence_thresholds`,
    a length-`n_layers` buffer), and that is the ONLY reason the mistake now raises
    instead of passing silently.  That is a lucky tripwire, not a check - it exists
    because a parameter happens to be shaped by the layer count.  This function is
    the actual check.
    """
    if not args.resume_pruned:
        return
    import torch
    from omegaconf import OmegaConf

    ck_dir = Path(args.resume_pruned)
    for name in ("checkpoint_best.tar", "checkpoint_pruned.tar"):
        ck = ck_dir / name
        if ck.is_file():
            break
    else:
        raise SystemExit(f"no checkpoint in {ck_dir} (looked for "
                         f"checkpoint_best.tar, checkpoint_pruned.tar)")
    ckpt = torch.load(str(ck), map_location="cpu", weights_only=False)
    if "conf" not in ckpt:
        raise SystemExit(f"{ck} has no `conf` entry, which train.load_experiment "
                         f"requires")
    want = OmegaConf.create(ckpt["conf"]).model.matcher
    got = OmegaConf.load(args.conf).model.matcher
    for key in ("n_layers", "num_heads", "descriptor_dim", "input_dim"):
        a, b = want.get(key), got.get(key)
        if a is not None and b is not None and int(a) != int(b):
            raise SystemExit(
                f"architecture mismatch: the checkpoint at {ck_dir} is "
                f"{key}={a} but --conf says {key}={b}.  `--conf` WINS over the "
                f"checkpoint's stored config, so this run would build the "
                f"{b}-layer model and load the {a}-layer weights into it with "
                f"`strict=False`, silently leaving the extra layers at their "
                f"random initialisation.  Pass the config that belongs to the "
                f"pruned checkpoint - `pruning/make_pruned_ckpt.py` writes it "
                f"next to the checkpoint.")
    # DISTINCT layer indices, not keys: `matcher.transformers.<i>.*` has 22
    # tensors per layer, so counting keys reports 154 for a 7-layer matcher - a
    # number that looks like a result and is not one.
    import re
    idx = {int(m.group(1)) for m in
           (re.match(r"matcher\.transformers\.(\d+)\.", k) for k in ckpt["model"])
           if m}
    print(f"[finetune] architecture check OK: checkpoint has {len(idx)} transformer "
          f"layer(s) (indices {min(idx)}..{max(idx)}), conf agrees "
          f"(n_layers={want.get('n_layers')})")


def build_command(args) -> list:
    """Assemble the glue-factory training command.

    Launched through the training framework rather than a bespoke loop on
    purpose: the data pipeline, the augmentation, the val split and the
    optimiser are all factors in the result, and re-implementing them would make
    the recovered model non-comparable with every existing measurement.
    """
    recipe = dict(DEFAULT_RECIPE)
    if args.resume_pruned:
        # `load_experiment` makes the run start from the pruned checkpoint.
        recipe["train.load_experiment"] = args.resume_pruned
    cmd = [args.python, "-u", "-m", "gluefactory.train", args.experiment,
           "--conf", args.conf]
    for k, v in recipe.items():
        cmd.append(f"{k}={v}")
    cmd += list(args.extra)
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True,
                    help="config for the PRUNED architecture (the one written "
                         "next to the pruned checkpoint by make_pruned_ckpt.py). "
                         "The original config would build the original model.")
    ap.add_argument("--experiment", required=True,
                    help="output experiment name (measured after recovery)")
    ap.add_argument("--resume-pruned", default=None,
                    help="experiment holding the PRUNED checkpoint; the recovery "
                         "runs start from it, so its pruned architecture must "
                         "already be saved as a checkpoint")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra key=value overrides")
    ap.add_argument("--gluefactory-root", default=None,
                    help="checkout to run `-m gluefactory.train` from; defaults to "
                         "$GLUEFACTORY_ROOT or a sibling directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    check_architecture(args)

    cmd = build_command(args)
    root = Path(args.gluefactory_root) if args.gluefactory_root \
        else gluefactory_root()
    print(f"[finetune] cwd={root}")
    print("[finetune] " + " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    sys.exit(main())
