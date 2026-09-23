#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 2 - LightGlue matcher, wrapped for ONNX export.

Input   keypoints0 (B, K, 2) pixels      keypoints1 (B, K, 2) pixels
        descriptors0 (B, K, 128)          descriptors1 (B, K, 128)
Output  matches0 (B, K) int64, -1 where unmatched
        mscores0 (B, K) float32

Why a thin wrapper rather than a rewrite
----------------------------------------
The matcher used here is `model.matchers.lightglue.LightGlue` (vendored from the
training framework - see that file for provenance), and it is already
export-compatible with the deployed configuration:

* `filter_matches` returns **statically shaped** `(B, M)` / `(B, N)` tensors with
  `-1` padding.  Some LightGlue ports use boolean-mask compaction
  (`indices[valid]`), which produces a data-dependent output shape - that is the
  usual reason an ONNX export of LightGlue fails, and this implementation does
  not do it.
* `normalize_keypoints` is arithmetic only (shift/scale by the image size).
* With `depth_confidence = -1` and `width_confidence = -1` there is no early-stop
  or point-pruning branch, so the graph is a straight 9-layer stack.
* `aggregation = last` (the trained setting) takes only the final layer's
  assignment map; the accumulate-averaging path is inactive.
* No `grid_sample`, no `NonZero`, no dynamic reshape anywhere in the forward.

So the wrapper's only job is to adapt tensor I/O to the dict I/O the module
expects, and to pin the image size as a constant so `normalize_keypoints` bakes
its scale into the graph.
"""
from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["LightGlueStage", "load_from_checkpoint"]


class LightGlueStage(nn.Module):
    """Tensor-in / tensor-out adapter around the trained LightGlue.

    `matcher` must already be in `eval()` mode with the deployed configuration
    (`input_dim=128`, `descriptor_dim=256`, `n_layers=9`, `num_heads=4`,
    `depth_confidence=width_confidence=-1`, `aggregation='last'`).
    """

    def __init__(self, matcher: nn.Module, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.matcher = matcher
        self.image_size = tuple(image_size)
        # A constant (not a Parameter/buffer with learnable meaning): it only
        # feeds `normalize_keypoints`, and pinning it is what lets the scale and
        # shift become graph constants instead of runtime tensor reads.
        self.register_buffer(
            "size_const",
            torch.tensor([self.image_size[1], self.image_size[0]], dtype=torch.float32),
            persistent=False)

    def forward(self, keypoints0, keypoints1, descriptors0, descriptors1):
        b = keypoints0.shape[0]
        view0 = {"image_size": self.size_const.unsqueeze(0).expand(b, 2)}
        view1 = {"image_size": self.size_const.unsqueeze(0).expand(b, 2)}
        pred = self.matcher({
            "keypoints0": keypoints0, "keypoints1": keypoints1,
            "descriptors0": descriptors0, "descriptors1": descriptors1,
            "view0": view0, "view1": view1,
        })
        return pred["matches0"], pred["matching_scores0"]

    def extra_repr(self) -> str:
        return f"image_size={self.image_size}"


def load_from_checkpoint(ckpt_path: str, image_size: Tuple[int, int] = (512, 512),
                         verbose: bool = True) -> LightGlueStage:
    """Build stage 2 from a trained checkpoint.

    The matcher configuration is read from the checkpoint itself rather than
    re-declared here, so an export can never silently disagree with the config
    the weights were trained under (a mismatch `input_dim` is the classic case:
    it makes `input_proj` an `nn.Identity` and drops 33,024 parameters without
    any error).

    This used to build the training framework's `TwoViewPipeline` and take
    `pipe.matcher` back out of it.  That was a lot of machinery for one line of
    work - the pipeline's `_init` is `get_model(conf.matcher.name)(conf.matcher)`
    and the extractor/`filter`/`solver` components were all `None`.  Building the
    matcher directly removes the last import of the training repo from the
    deployable code path; the checkpoint's `matcher` conf is already complete,
    including `name`, which is now simply unused.
    """
    from omegaconf import OmegaConf

    from .matchers.lightglue import LightGlue

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    conf = OmegaConf.create(ck["conf"])
    model_conf = OmegaConf.to_container(conf.model, resolve=True)

    if model_conf["matcher"].get("input_dim") is None:
        raise KeyError("checkpoint matcher has no `input_dim`; refusing to "
                       "guess (a wrong value turns `input_proj` into Identity)")

    matcher = LightGlue(model_conf["matcher"])

    # The checkpoint stores the matcher under a `matcher.` prefix because it was
    # trained inside the two-view pipeline.  Strip it rather than rebuilding the
    # pipeline wrapper the prefix came from - and check the strip was complete,
    # so a prefix the loader did not expect cannot silently drop half the weights.
    prefix = "matcher."
    state = {k[len(prefix):]: v for k, v in ck["model"].items()
             if k.startswith(prefix)}
    missing, unexpected = matcher.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("source_emb.")]
    if missing or unexpected:
        raise KeyError(
            f"checkpoint does not match the matcher built from its own conf: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}")
    matcher = matcher.eval()

    if verbose:
        n = sum(p.numel() for p in matcher.parameters())
        mconf = model_conf["matcher"]
        print(f"[lightglue-stage] {n:,} params loaded from {ckpt_path}")
        print(f"[lightglue-stage] input_dim={mconf['input_dim']} "
              f"descriptor_dim={mconf['descriptor_dim']} "
              f"n_layers={mconf['n_layers']} heads={mconf['num_heads']} "
              f"agg={mconf.get('aggregation', 'last')}")

    return LightGlueStage(matcher, image_size=image_size).eval()
