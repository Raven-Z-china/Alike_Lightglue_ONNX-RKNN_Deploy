#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NPU-safe coordinate sampling: round|floor + clip + gather.

Why this module exists
----------------------
`F.grid_sample` is not executable on Rockchip NPUs.  When an ONNX graph contains
it, the runtime either refuses to build the model or silently schedules the node
on the CPU, which stalls the whole pipeline on every inference (the NPU idles
while a 512x512 map is copied to the host and back).

Both lookups this project needs are *point samples* of a dense map at known
integer-ish positions, so they are expressible with `Round/Floor -> Clip -> Cast
-> Gather`, all of which the NPU executes natively.

The numerical contract
----------------------
The original code samples with `grid_sample(..., mode="bilinear",
align_corners=True, padding_mode="zeros")`.  With `align_corners=True` the
normalised coordinate `n` in [-1, 1] maps to pixel `p = (n + 1) / 2 * (N - 1)`.
This module reproduces that mapping EXACTLY and then rounds/clips, so switching
to it changes one thing and one thing only: **bilinear interpolation becomes
nearest-neighbour**.  That difference is measured, not assumed - see
`scripts/check_accuracy.py`.

Two rounding modes, deliberately
--------------------------------
* `round`  - used for the keypoint SCORE lookup.  This is the deliberate
             replacement of a `grid_sample` call.
* `floor`  - used for the DESCRIPTOR lookup, because that code path never used
             `grid_sample`: it did `p.long()`, which truncates toward zero (and
             equals floor for the non-negative coordinates here).  Replacing it
             with `round` would silently move descriptor sample positions, so it
             is kept byte-compatible instead.

Out-of-range coordinates are clipped rather than zero-padded.  For the
descriptor path the original relied on keypoints always being interior (ALIKE's
DKD zeroes a 3 px border before top-k, so the surviving points are at least 3 px
from every edge and never come close to clipping).  The clip is therefore a
safety net that should be a numerical no-op; layer L1 verifies that it is.
"""
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["to_pixel", "GatherSample", "GatherSampleBilinear"]


def _flatten_gather(map4d: torch.Tensor, flat_idx: torch.Tensor,
                    row_off: torch.Tensor) -> torch.Tensor:
    """Gather from a (B, C, H, W) map at per-(b, k) flat indices -> (B, C, K).

    WHY THIS IS NOT `torch.gather`
    ------------------------------
    `torch.gather(x, dim, idx)` with a rank-2 index exports as **GatherElements**,
    and the RKNN simulator mishandles it: the converted model still runs, the
    output ranges still look plausible, and the descriptors come out at cosine
    0.88 against the reference instead of ~1.0.  That failure is invisible without
    a numeric check, and it survived one rewrite (the row-offset version still
    emitted GatherElements) - the only formulation that emits a plain ONNX
    **Gather** is `index_select` on a 1-D tensor with a 1-D index.

    So: flatten the whole map to one dimension, add each element's row offset
    (`(b * C + c) * H * W`, precomputed as a constant so no `Range` op is needed),
    and do ONE `index_select`.  Reshape / Add / Gather / Reshape - all NPU-native.
    """
    b, c = map4d.shape[0], map4d.shape[1]
    k = flat_idx.shape[-1]
    idx = flat_idx.reshape(b, 1, k).expand(b, c, k) + row_off[:b]
    out = torch.index_select(map4d.reshape(-1), 0, idx.reshape(-1))
    return out.reshape(b, c, k)


def to_pixel(norm_xy: torch.Tensor, h: int, w: int):
    """`grid_sample(align_corners=True)` convention: [-1,1] -> pixel.

    `norm_xy` is (..., 2) in (x, y) order; returns two tensors of the same shape.
    """
    x = (norm_xy[..., 0] + 1.0) * 0.5 * (w - 1)
    y = (norm_xy[..., 1] + 1.0) * 0.5 * (h - 1)
    return x, y


class GatherSample(nn.Module):
    """Sample a dense map at per-point pixel coordinates.

    Parameters
    ----------
    mode : "round" | "floor"
        Rounding applied to the pixel coordinate before gathering.

    Inputs
    ------
    map4d : (B, C, H, W)
    x_pix, y_pix : (B, K) float - coordinates in PIXELS, already de-normalised.

    Returns
    -------
    (B, C, K)

    Implementation notes
    --------------------
    * The flat index is built as `y * W + x` and gathered from a `(B, 1, H*W)`
      view, so the channel dimension is carried by the gather rather than by a
      Python loop.
    * Every op is shape-static (`Round`/`Floor`, `Clip`, `Cast`, `Mul`, `Add`,
      `Gather`), which is what the NPU requires - no `NonZero`, no dynamic
      reshape.
    * `batch` and `hw` are baked into `_row_off`, so they must match the graph
      being exported.  The deployment extracts ONE image per call (`batch=1`).
    """

    def __init__(self, mode: Literal["round", "floor"] = "round",
                 batch: int = 1, channels: int = 1, hw: int = 512 * 512):
        super().__init__()
        if mode not in ("round", "floor"):
            raise ValueError(f"mode must be 'round' or 'floor', got {mode!r}")
        self.mode = mode
        self.register_buffer(
            "_row_off",
            (torch.arange(batch * channels, dtype=torch.long) * hw)
            .reshape(batch, channels, 1),
            persistent=False)

    def forward(self, map4d: torch.Tensor, x_pix: torch.Tensor, y_pix: torch.Tensor):
        b, c, h, w = map4d.shape
        if self.mode == "round":
            xi = torch.round(x_pix)
            yi = torch.round(y_pix)
        else:
            xi = torch.floor(x_pix)
            yi = torch.floor(y_pix)
        # Clamp BEFORE the cast: `long()` on an out-of-range float is UB-ish and
        # differs between backends, whereas Clamp is well defined and maps to one
        # NPU op.
        xi = xi.clamp(0, w - 1).long()
        yi = yi.clamp(0, h - 1).long()
        return _flatten_gather(map4d, yi * w + xi, self._row_off)


class GatherSampleBilinear(nn.Module):
    """`grid_sample(mode="bilinear", align_corners=True, padding_mode="zeros")`
    reproduced with four gathers plus arithmetic - no `grid_sample` in the graph.

    WHY THIS EXISTS, AND WHY IT IS THE DEFAULT
    ------------------------------------------
    The brief asked for `round + clip + gather` to keep `grid_sample` off the CPU.
    That works, but measuring it showed it is **not lossless**: the score map is
    extremely sharp at full resolution (adjacent pixels differ by up to 0.99), so
    moving the sample from the sub-pixel position to the nearest integer shifts
    the keypoint score by up to 0.66 (mean 0.065), and a downstream score
    threshold then keeps a 2-5 % different keypoint set.

    Bilinear interpolation is a weighted sum of four *integer* neighbours, which
    is just as gather-friendly as nearest - so the CPU-fallback problem and the
    accuracy requirement can both be satisfied at once, with no rounding error
    beyond float associativity.

    `GatherSample` (nearest) is kept and still selectable, because it is what was
    asked for and its cost is now measured rather than assumed.

    Numerical contract: matches `grid_sample` with `padding_mode="zeros"` - the
    out-of-range corners contribute zero, implemented as a multiply by a validity
    mask rather than by relying on padding.

    `batch` and `hw` are baked into `_row_off`, so they must match the graph being
    exported.  The deployment extracts ONE image per call (`batch=1`).
    """

    def __init__(self, batch: int = 1, channels: int = 1, hw: int = 512 * 512):
        super().__init__()
        self.register_buffer(
            "_row_off",
            (torch.arange(batch * channels, dtype=torch.long) * hw)
            .reshape(batch, channels, 1),
            persistent=False)

    def forward(self, map4d: torch.Tensor, x_pix: torch.Tensor, y_pix: torch.Tensor):
        b, c, h, w = map4d.shape
        dtype = map4d.dtype
        x0 = torch.floor(x_pix)
        y0 = torch.floor(y_pix)
        wx = (x_pix - x0).unsqueeze(1)                 # (B, 1, K)
        wy = (y_pix - y0).unsqueeze(1)
        x0 = x0.long()
        y0 = y0.long()

        def corner(xi, yi):
            """Gather one corner, zeroing samples that fall outside the map.

            The in-bounds test multiplies 0/1 floats instead of chaining `&`:
            Boolean `And` does not lower to the NPU (see `simple_nms` for the
            measured consequence), while the product of two 0/1 tensors is the
            same predicate using only Cast and Mul.
            """
            vx = (xi >= 0).to(dtype) * (xi <= w - 1).to(dtype)
            vy = (yi >= 0).to(dtype) * (yi <= h - 1).to(dtype)
            valid = vx * vy
            g = _flatten_gather(
                map4d, yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1), self._row_off)
            return g * valid.unsqueeze(1)

        s00 = corner(x0, y0)
        s01 = corner(x0 + 1, y0)
        s10 = corner(x0, y0 + 1)
        s11 = corner(x0 + 1, y0 + 1)

        top = s00 * (1.0 - wx) + s01 * wx
        bot = s10 * (1.0 - wx) + s11 * wx
        return top * (1.0 - wy) + bot * wy
