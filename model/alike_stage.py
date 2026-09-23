#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 1 - ALIKE backbone + DKD + descriptor sampling, as ONE exportable graph.

Input   image        (1, 3, S, S) float32 in [0, 1]   (S = 512, a multiple of 32)
Output  keypoints    (1, K, 2)     float32, PIXELS
        descriptors  (1, K, 128)   float32, L2-normalised per keypoint
        scores       (1, K)        float32, keypoint confidence

ONE IMAGE PER CALL
------------------
The extractor is per-image by design: a pair is TWO calls to this stage, and the
two feature sets are then handed to stage 2 together.  That is what makes a single
extraction reusable (one image against N others) and halves the peak activation
footprint; the cost is one extra dispatch per pair.  `batch`/`size` are
constructor arguments because the sampling tables are baked into the graph - for
this deployment they are 1 and 512.

Why one graph and not three
---------------------------
The three outputs come from one forward pass over a full-resolution dense map.
Splitting them would re-run the backbone or force the 128-channel map through
host memory (2 x 512 x 512 x 128 x 4 B = 268 MB per call at float32, which the
measurements in the speedup notes show is the dominant cost on a PCIe host).
Keeping it fused means the dense map never leaves the NPU.

Descriptor source
-----------------
This deployment uses ALIKE's **own** `convhead2` descriptors
(`descriptor_source: alike`), NOT the SDDH head.  The SDDH head is therefore not
part of this module and its weights are not exported - see `load_from_checkpoint`
for how the checkpoint's unused head is skipped.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .dkd import DKDExport
from .gather_ops import GatherSample, GatherSampleBilinear, to_pixel
from .gathered_head import GatheredDescriptorHead

__all__ = ["AlikeStage", "load_from_checkpoint"]


class AlikeStage(nn.Module):
    """Backbone -> dense maps -> DKD keypoints -> sampled descriptors."""

    def __init__(self, variant: str = "alike-n", top_k: int = 512,
                 nms_radius: int = 2, descriptor_dim: int = 128,
                 descriptor_interp: str = "bilinear",
                 gathered_head: bool = True,
                 batch: int = 1, size: int = 512):
        """`descriptor_interp` selects how descriptors are sampled at the
        sub-pixel keypoints.

        "bilinear" (default here, NOT upstream) - interpolates, so the descriptor
            is a CONTINUOUS function of the keypoint position.  This matters for
            portability: on an fp16 NPU the keypoint itself is only accurate to
            about 0.043 px (per-pair median over 8 HPatches pairs), and with
            "floor" a jitter of that size that
            happens to cross an integer boundary swaps the descriptor for a
            NEIGHBOURING PIXEL's.  The descriptor map is high-frequency -
            adjacent pixels of the L2-normalised map sit at cosine mean 0.859,
            min 0.556 - so a flip costs a ~0.14 cosine step no matter how small
            the jitter is: a zeroth-order error in the keypoint displacement,
            where bilinear's is first-order.  Worst case measured, on the noise
            reference dump: minimum cosine 0.54 with "floor" versus 0.90 with
            "bilinear".  A property of the sampling rule, not of the conversion.
        "floor" - upstream ALIKE's rule (`.long()` indexing).  Kept, because it
            is what the released weights were trained with, so any accuracy claim
            has to be made against it at least once.
        """
        super().__init__()
        if descriptor_interp not in ("bilinear", "floor"):
            raise ValueError(f"descriptor_interp must be 'bilinear' or 'floor', "
                             f"got {descriptor_interp!r}")
        self.variant = variant
        self.top_k = int(top_k)
        self.descriptor_dim = int(descriptor_dim)
        self.descriptor_interp = descriptor_interp
        # DEFAULT ON.  When True the descriptor is computed only at the keypoints
        # (`gathered_head`), which removes a dense 129-channel map that is 24 % of
        # the pipeline's MACs and 99.8 % discarded.  It is a RESTRUCTURING, verified
        # exact to 1.19e-07, not a truncation - see `gathered_head.py`.
        #
        # It is nevertheless a flag rather than unconditional, because it changes the
        # shape of the exported graph and every measured number in this project was
        # taken with the dense form.  `False` reproduces the previous graph exactly,
        # so the two can be compared on the same evaluator.
        self.gathered_head = bool(gathered_head)
        self.batch = int(batch)
        self.size = int(size)
        self.net = build_backbone(variant)
        self.dkd = DKDExport(radius=nms_radius, top_k=top_k, batch=self.batch,
                             size=self.size)
        if self.gathered_head:
            self.desc_sample = GatheredDescriptorHead(
                self.net.convhead2, channels=descriptor_dim,
                hw=self.size * self.size, batch=self.batch)
        else:
            self.desc_sample = (
                GatherSampleBilinear(batch=self.batch, channels=descriptor_dim,
                                     hw=self.size * self.size)
                if descriptor_interp == "bilinear"
                else GatherSample(mode="floor", batch=self.batch,
                                  channels=descriptor_dim,
                                  hw=self.size * self.size))

    def _check_size(self, h: int, w: int):
        if h % 32 or w % 32:
            raise ValueError(
                f"input {h}x{w} is not a multiple of 32; this export only "
                f"supports the no-padding path (use 512x512)")
        if (h, w) != (self.size, self.size):
            raise ValueError(
                f"input {h}x{w} does not match the size this module was built for "
                f"({self.size}x{self.size}); the sampling tables are baked at "
                f"construction time, so build with size={h} to run at that "
                f"resolution instead of silently addressing the wrong rows")

    def dense_maps(self, image: torch.Tensor):
        """Reproduce `ALikeSDDH._extract_dense_map` exactly.

        At S = 512 (`512 % 32 == 0`) the padding branch is never taken, so no
        `F.pad` appears in the exported graph - which is deliberate, because a
        conditional pad would not be representable.  The assertion below makes
        that a checked precondition rather than a silent assumption.

        UNAVAILABLE under `gathered_head=True`, and it raises rather than
        approximating: the whole point of that mode is that the dense map is never
        materialised, so a caller asking for it is asking for the thing that was
        removed.  Use `forward`.
        """
        if self.gathered_head:
            raise RuntimeError(
                "dense_maps() does not exist under gathered_head=True - the dense "
                "descriptor map is exactly what that mode removes. Use forward(). "
                "Callers that need it (the per-tensor parity harness) must build "
                "the stage with gathered_head=False.")
        _, _, h, w = image.shape
        self._check_size(h, w)
        scores_map, descriptor_map = self.net(image)
        # Load-bearing: the matcher was trained on normalised descriptors, and
        # `convhead2` is bias-free so raw magnitudes would shift the statistics.
        descriptor_map = F.normalize(descriptor_map, p=2, dim=1)
        return scores_map, descriptor_map

    def forward(self, image: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = image.shape
        self._check_size(h, w)

        if self.gathered_head:
            # Score channel densely (NMS and top-k need the whole map), then the
            # descriptor only at the keypoints.  `x1234` is `convhead2`'s input, so
            # the descriptor rows are applied to a gathered view of the SAME tensor
            # the dense path would have convolved.
            x1234 = self.net.trunk(image)
            scores_map = self.net.score_map(x1234)
            keypoints, scores = self.dkd(scores_map)
            x_pix, y_pix = to_pixel(self.dkd._normalise(keypoints, h, w), h, w)
            desc = self.desc_sample(x1234, x_pix, y_pix)          # (B, 128, K)
            desc = F.normalize(desc, p=2, dim=1)                  # per-point
            desc = desc.transpose(1, 2).contiguous()              # (B, K, 128)
            return keypoints, desc, scores

        scores_map, descriptor_map = self.dense_maps(image)
        keypoints, scores = self.dkd(scores_map)                 # (B,K,2) px, (B,K)

        # Sample descriptors at the sub-pixel keypoints.  `to_pixel` inverts the
        # normalisation the sampler expects; `floor` reproduces the original
        # truncating index.
        x_pix, y_pix = to_pixel(self.dkd._normalise(keypoints, h, w), h, w)
        desc = self.desc_sample(descriptor_map, x_pix, y_pix)     # (B, 128, K)
        desc = F.normalize(desc, p=2, dim=1)                      # per-point
        desc = desc.transpose(1, 2).contiguous()                  # (B, K, 128)

        return keypoints, desc, scores

    def extra_repr(self) -> str:
        return (f"variant={self.variant}, top_k={self.top_k}, "
                f"size={self.size}, batch={self.batch}")


def load_from_checkpoint(ckpt_path: str, variant: str = "alike-n", top_k: int = 512,
                         nms_radius: int = 2, strict: bool = True,
                         verbose: bool = True,
                         descriptor_interp: str = "bilinear",
                         gathered_head: bool = True,
                         batch: int = 1, size: int = 512) -> AlikeStage:
    """Build a stage-1 module and load the extractor weights from a checkpoint.

    The checkpoint stores the extractor as `extractor.net.*` (backbone) plus
    `extractor.desc_head.*` (SDDH).  SDDH is **not** part of this deployment -
    the matcher reads `convhead2` descriptors - so those tensors are dropped
    rather than loaded into a head this module does not build.

    `strict=True` refers to the BACKBONE only: the check is that every one of the
    backbone's tensors was found and consumed.  Silently missing backbone weights
    would still produce a model that runs, and would score far worse for a reason
    no test would localise - so missing keys raise.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    backbone = {k[len("extractor.net."):]: v for k, v in state.items()
                if k.startswith("extractor.net.")}
    if not backbone:
        raise KeyError(
            f"no `extractor.net.*` tensors in {ckpt_path}; found prefixes "
            f"{sorted({k.split('.')[0] for k in state})}")

    stage = AlikeStage(variant=variant, top_k=top_k, nms_radius=nms_radius,
                       descriptor_interp=descriptor_interp,
                       gathered_head=gathered_head, batch=batch, size=size)
    missing, unexpected = stage.net.load_state_dict(backbone, strict=False)
    if missing:
        raise KeyError(f"backbone weights missing from the checkpoint: {missing}")
    # `unexpected` is expected to be empty here because we filtered by prefix.
    if unexpected:
        raise KeyError(f"unexpected backbone keys: {unexpected}")

    if verbose:
        n_backbone = sum(v.numel() for v in stage.net.state_dict().values())
        skipped = sum(v.numel() for k, v in state.items()
                      if k.startswith("extractor.desc_head."))
        print(f"[alike-stage] backbone {n_backbone:,} params loaded from "
              f"{ckpt_path}")
        if skipped:
            print(f"[alike-stage] skipped SDDH desc_head ({skipped:,} params) - "
                  f"not used by this deployment")
    return stage.eval()
