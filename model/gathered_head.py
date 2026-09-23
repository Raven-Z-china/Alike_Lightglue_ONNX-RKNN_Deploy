#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descriptor at sub-pixel keypoints, with the head applied PER CORNER.

WHAT THIS REPLACES, AND WHY
---------------------------
`ALikeNet` computes its descriptor by running `convhead2` - a 1x1 convolution with
129 output channels - over every pixel of the feature map, and `AlikeStage` then
samples 128 of those channels at ~512 keypoints.  At 512x512 that is

    129 x 128 x 512 x 512  =  4.33G MACs   (24 % of the WHOLE pipeline)

to produce 33.6 M descriptor values of which **65,536 are read - 0.2 %**.

A 1x1 convolution is a per-pixel linear map with no spatial mixing, so it commutes
with point sampling: `(W x)[:, P] == W x[:, P]`.  The descriptor rows can therefore
be applied to the gathered FEATURES instead of gathering the dense output, and the
dense descriptor map never has to exist.

WHY IT IS NOT JUST "GATHER, THEN APPLY THE HEAD"
------------------------------------------------
`AlikeStage` normalises the descriptor map PER PIXEL before sampling it:

    descriptor_map = F.normalize(descriptor_map, p=2, dim=1)   # per pixel
    desc = self.desc_sample(descriptor_map, x_pix, y_pix)      # bilinear
    desc = F.normalize(desc, p=2, dim=1)                       # per keypoint

and per-pixel normalisation is NON-LINEAR, so it does **not** commute with bilinear
interpolation:

    normalize_perpixel(W @ bilinear(x))  !=  bilinear(normalize_perpixel(W @ x))

Measured on real keypoints, the naive form (`GatherSampleBilinear` applied to raw
features and the head applied afterwards - "variant B" of the head-restructure
probe, kept with the measurement record outside this repository) reaches a
descriptor cosine of only **0.9997** against the shipped path - a real change, not
float noise.

So the head is applied to EACH of the four integer corners, each corner is
normalised on its own - which is exactly what the dense path does per pixel - and the
four normalised corners are then combined with the bilinear weights.  That is the
shipped arithmetic restricted to the corners that are read, and it is exact:

    variant C : sampler max |Δ| = 1.19e-07,  cos min 0.9999995

against the fp16 NPU conversion's own 0.9997, i.e. four orders of magnitude smaller
than the error already accepted.

THE ORDER OF THE VALIDITY MASK IS LOAD-BEARING
----------------------------------------------
`GatherSampleBilinear.corner` gathers with a clamp and THEN multiplies by the
in-bounds mask, on the already-normalised map.  Applying the mask BEFORE the head
would leave the bias, and - if the convolution had one - `normalize(bias) * 0` is
still zero, but `normalize(bias)` alone is a full-strength unit vector.  The mask is
therefore applied after the normalise, at the same point in the pipeline.

This file's own predecessor got that wrong in the probe and the error was visible as
`max |Δ| = 0.029` sitting next to a cosine of 0.9999996 - two numbers that cannot
describe the same comparison.  (`convhead2` turns out to have no bias, so the
particular failure did not fire; the ordering is kept correct regardless, because it
depends on a property of the weights rather than on the code.)

OPERATORS, AND WHY THE CONVERSION RISK IS LOW
---------------------------------------------
`Gather`, `MatMul`, `Mul`, `Add`, `ReduceSum`/`Sqrt` (the normalise) and the
reshape/arithmetic around them are all NPU-native.  More to the point, the graph
ALREADY contains four 128-channel gathers at K points - that is what
`GatherSampleBilinear` does - so this change does not introduce an op the graph has
not been converted with.  It removes one large convolution and adds four small
matmuls over the same gathered data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gather_ops import _flatten_gather

__all__ = ["GatheredDescriptorHead"]


class GatheredDescriptorHead(nn.Module):
    """Sample a descriptor map at sub-pixel points WITHOUT materialising it.

    Parameters
    ----------
    head_conv : nn.Conv2d
        The 1x1 convolution whose rows produce the descriptor.  It is held by
        REFERENCE, not copied, so the checkpoint keeps a single source of truth for
        the weights and this module adds nothing to the state dict.
    channels : int
        Descriptor width (128).  Must equal `head_conv.in_channels`.
    hw : int
        Flattened spatial size of the feature map, used to build the constant row
        offset table (see `_flatten_gather` for why the offset is precomputed).
    batch : int
        Number of images per call.  1 for this deployment: the extractor runs once
        per image and the two views are matched afterwards.  Baked into the offset
        table, so it is fixed at construction.

    Inputs
    ------
    features  : (B, C, H, W)  the tensor `head_conv` would have consumed
    x_pix, y_pix : (B, K)     sub-pixel coordinates in PIXELS

    Returns
    -------
    (B, C, K)  the same values `convhead2` followed by `GatherSampleBilinear` would
    have produced, up to float32 associativity.
    """

    def __init__(self, head_conv: nn.Conv2d, channels: int = 128,
                 hw: int = 512 * 512, batch: int = 1):
        super().__init__()
        self.head_conv = head_conv
        self.channels = int(channels)
        self.hw = int(hw)
        self.register_buffer(
            "_row_off",
            (torch.arange(batch * self.channels, dtype=torch.long) * self.hw)
            .reshape(batch, self.channels, 1),
            persistent=False)

    def _rows(self):
        """The descriptor rows of the head as a (channels, in_channels) matrix.

        `head_conv.weight` is (out, in, 1, 1); the trailing kernel dims are 1 and
        have to be reshaped away before it can be used as a matrix.  The LAST output
        row is the score channel and is excluded - it is computed densely elsewhere,
        because NMS and top-k act on the whole map.
        """
        w = self.head_conv.weight
        return w[:-1].reshape(w.shape[0] - 1, w.shape[1])

    def _bias(self):
        b = self.head_conv.bias
        if b is None:
            # `conv1x1` in the backbone is built without a bias, so this is the
            # normal path - but it is derived from the module rather than assumed,
            # so a head WITH a bias would work too.
            return None
        return b[:-1]

    def forward(self, features: torch.Tensor, x_pix: torch.Tensor,
                y_pix: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        dtype = features.dtype
        w_desc = self._rows()
        bias = self._bias()

        x0 = torch.floor(x_pix)
        y0 = torch.floor(y_pix)
        wx = (x_pix - x0).unsqueeze(1)                 # (B, 1, K)
        wy = (y_pix - y0).unsqueeze(1)
        x0 = x0.long()
        y0 = y0.long()

        def corner(xi, yi):
            """Head + normalise at ONE corner, then mask out-of-bounds to zero.

            The mask goes on AFTER the normalise - see the module docstring for why
            that order is not cosmetic.
            """
            vx = (xi >= 0).to(dtype) * (xi <= w - 1).to(dtype)
            vy = (yi >= 0).to(dtype) * (yi <= h - 1).to(dtype)
            valid = (vx * vy).unsqueeze(1)
            g = _flatten_gather(
                features, yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1),
                self._row_off)
            # `matmul` broadcasts the batch dim of `g`: (C, in) x (B, in, K)
            d = torch.matmul(w_desc, g)
            if bias is not None:
                d = d + bias[None, :, None]
            return F.normalize(d, p=2, dim=1) * valid

        d00 = corner(x0, y0)
        d01 = corner(x0 + 1, y0)
        d10 = corner(x0, y0 + 1)
        d11 = corner(x0 + 1, y0 + 1)

        # Same combination as `GatherSampleBilinear.forward`, on per-corner
        # normalised descriptors instead of raw map values.
        top = d00 * (1.0 - wx) + d01 * wx
        bot = d10 * (1.0 - wx) + d11 * wx
        return top * (1.0 - wy) + bot * wy

    def extra_repr(self) -> str:
        return (f"channels={self.channels}, hw={self.hw}, "
                f"head={type(self.head_conv).__name__}")
