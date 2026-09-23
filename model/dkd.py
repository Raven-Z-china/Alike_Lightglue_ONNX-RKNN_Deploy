#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export-safe re-implementation of ALIKE's DKD keypoint detector.

What had to change, and why
---------------------------
The upstream detector (`gluefactory/models/extractors/alike_sddh.py::DKD`) is
written for training: it loops over the batch in Python, uses `nn.Unfold` to
materialise a `(B, 25, H*W)` patch tensor, writes the border mask **in place**,
and samples the keypoint score with `F.grid_sample`.  Of those, two are hard
blockers for an NPU:

1. `F.grid_sample` - not NPU-executable, falls back to CPU (see `gather_ops.py`).
2. `nn.Unfold` - lowers to `Im2Col`, which is not an ONNX/RKNN operator.  It is
   also enormously wasteful here: it builds 25 x H x W floats per image to then
   read only 25 values at each of 512 keypoints.

This module replaces both.  The neighbour window is gathered by index arithmetic
(one `Gather`), and the score lookup goes through `GatherSample`.  Everything
else - the two NMS rounds, the border zeroing, top-k, and the temperature-0.1
soft-argmax - is reproduced op-for-op.

What is deliberately NOT reproduced
-----------------------------------
`detect_keypoints` also returns `scoredispersitys`, which `ALikeSDDH._forward`
immediately discards (`keypoints, _, kptscores = ...`).  Computing it would cost
a 25-wide norm and a reduction per keypoint for a value nothing reads, so it is
omitted.  `forward` below returns `(keypoints_px, scores)` only.

Numerical contract
------------------
* The border mask is applied with `torch.where`, not in-place assignment.  Scores
  are sigmoid outputs in (0, 1) so `x * 0` would also work, but `where` is exactly
  equivalent to assignment and cannot be wrong for a negative or NaN input.
* Neighbour indices are clamped.  ALIKE zeroes a `radius + 1 = 3` px border
  before top-k, so a surviving keypoint sits at least 3 px from every edge and a
  +/-2 window can never leave the map.  The clamp is therefore a no-op safety net
  and layer L1 of the parity protocol confirms it does not change a single value.
* Keypoints are returned in PIXELS.  The upstream detector returns normalised
  [-1, 1] and the extractor converts back with `wh * (kp + 1) / 2`; doing the
  conversion here removes two ops from the exported graph and cannot change the
  result because the conversion is exact arithmetic on the same numbers.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gather_ops import GatherSample, GatherSampleBilinear, to_pixel

__all__ = ["DKDExport"]


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """ALIKE's `simple_nms`, rewritten in FLOAT arithmetic for the NPU.

    Upstream is written with boolean tensors:

        max_mask  = scores == max_pool(scores)                 # bool
        supp_mask = max_pool(max_mask.float()) > 0             # bool
        max_mask  = max_mask | (new_max_mask & (~supp_mask))   # And / Or / Not

    Two changes were forced by measurement, not by taste:

    * **The `Or`/`And`/`Not` chain does not lower to the NPU.**  The RKNN build
      reports `No lowering found for: node_bitwise_or, node type = Or, use
      CustomOperatorLower instead`, and the simulator then disagrees with the
      reference by 484 px on keypoints.  Boolean logic has to become arithmetic:
      a 0/1 float mask with `or -> a + b - a*b` (clamped) and `and -> a * b`.
      That is exact for 0/1 operands, and it is what makes the graph NPU-native.
    * **`>=` replaces `==`.**  `max_pool` can only return a value already present
      in the window, so `scores <= max_pool(scores)` always holds and
      `scores >= max_pool(scores)` is therefore exactly equivalent to equality -
      while `Equal` is a less reliably lowered operator than a comparison.

    Everything else (two fixed rounds, the same pooling kernel, the same final
    `where(mask, scores, 0)`) is unchanged, and `where(mask, s, 0)` becomes
    `s * mask` because scores are sigmoid outputs and therefore non-negative.

    The loop stays unrolled: the round count is a constant, so the graph is a
    fixed sequence of `MaxPool`/`GreaterOrEqual`/`Cast`/`Mul`/`Sub`/`Clamp`
    nodes with no control flow - which is what RKNN requires.
    """

    def max_pool(x):
        return F.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    dtype = scores.dtype
    # `>=` is `==` here (see docstring), and casting to float makes the mask
    # usable in arithmetic without any boolean operator.
    max_mask = (scores >= max_pool(scores)).to(dtype)

    for _ in range(2):
        supp_mask = (max_pool(max_mask) >= 1.0).to(dtype)      # > 0, as 0/1
        supp_scores = scores * (1.0 - supp_mask)               # where(supp, 0, s)
        new_max_mask = (supp_scores >= max_pool(supp_scores)).to(dtype)
        add = new_max_mask * (1.0 - supp_mask)                 # and(~supp)
        max_mask = torch.clamp(max_mask + add, 0.0, 1.0)       # or
    return scores * max_mask                                   # where(mask, s, 0)


class DKDExport(nn.Module):
    """Detect `top_k` keypoints from a score map; export- and NPU-safe.

    Parameters mirror upstream; the defaults are the deployed configuration
    (`radius=2`, `top_k=512`, `temperature=0.1`).  `temperature` is upstream's
    tuned constant and is not a tunable here.

    `batch` and `size` are baked into the score lookup's offset table, so they have
    to match the graph being exported: `size` is the square input side and `batch`
    is the number of images per call (1 for this deployment - one call per image).
    """

    def __init__(self, radius: int = 2, top_k: int = 512, temperature: float = 0.1,
                 score_mode: str = "bilinear", batch: int = 1, size: int = 512):
        """`score_mode` selects how the keypoint score is read off the score map.

        "bilinear" (default) - reproduces `grid_sample` exactly via four gathers;
                               keeps the deployment lossless.
        "nearest"            - the literal `round + clip + gather` brief.  Cheaper
                               (one gather instead of four) and still fully
                               NPU-resident, but not lossless: see `GatherSample`
                               and the L1 report.
        """
        super().__init__()
        if score_mode not in ("bilinear", "nearest"):
            raise ValueError(f"score_mode must be 'bilinear' or 'nearest', "
                             f"got {score_mode!r}")
        self.score_mode = score_mode
        self.radius = int(radius)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.kernel_size = 2 * self.radius + 1

        # Local (x, y) offset grid, built EXACTLY as upstream does so the
        # soft-argmax weights and the gather offsets stay mutually consistent.
        # Same construction as ALIKE's DKD.__init__.
        x = torch.linspace(-self.radius, self.radius, self.kernel_size)
        grid = torch.stack(torch.meshgrid([x, x], indexing="ij")).view(2, -1).t()
        self.register_buffer("hw_grid", grid[:, [1, 0]].contiguous())

        # Integer offsets for the gather, derived from the SAME buffer.  Entry k
        # of the unfolded window (row-major, kernel row -> y, column -> x) is
        # offset (dx, dy) = (hw_grid[k, 0], hw_grid[k, 1]); taking them from the
        # buffer rather than re-deriving them is what guarantees the gather and
        # the soft-argmax address the same neighbour.
        self.register_buffer("dx", self.hw_grid[:, 0].round().long())
        self.register_buffer("dy", self.hw_grid[:, 1].round().long())

        # The score lookup - the ONE place the original used grid_sample.
        # Dimensions are passed explicitly so the row-offset table is a constant in
        # the graph (see `gather_ops._flatten_gather`); `size` comes from the
        # caller rather than being hardcoded, so a non-512 export cannot end up
        # with a table that silently addresses the wrong rows.
        self.size = int(size)
        self.score_sample = (
            GatherSampleBilinear(batch=batch, channels=1,
                                 hw=self.size * self.size)
            if score_mode == "bilinear"
            else GatherSample(mode="round", batch=batch, channels=1,
                              hw=self.size * self.size))

    def extra_repr(self) -> str:
        return f"radius={self.radius}, top_k={self.top_k}"

    def forward(self, scores_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """scores_map: (B, 1, H, W) in [0, 1] -> keypoints_px (B, K, 2), scores (B, K)."""
        b, _, h, w = scores_map.shape
        r = self.radius

        # ---- 1. non-maximum suppression + border zeroing --------------------
        nms = simple_nms(scores_map, r)

        # Upstream zeroes the border in place with four slice assignments.  This
        # multiplies by a 0/1 float mask instead: same result (NMS scores are
        # non-negative), no in-place write, and no `Where` node - `Where` is one
        # more operator the NPU may not lower.
        nms = nms * self._border_mask(h, w, r, scores_map.dtype,
                                      scores_map.device)

        # ---- 2. top-k over the flattened map --------------------------------
        # `sorted=True` (the default) is kept: the reference output is sorted, and
        # ONNX/RKNN TopK honours the flag, so the keypoint ORDER matches too.
        idx = torch.topk(nms.reshape(b, -1), self.top_k).indices        # (B, K)

        ys = idx // w
        xs = idx % w
        flat_scores = scores_map.reshape(b, 1, h * w)

        # ---- 3. gather the 5x5 window around every keypoint -----------------
        # The flat index of the neighbour at offset (dx, dy) is
        #
        #     (y + dy) * w + (x + dx)  =  (y * w + x) + (dy * w + dx)  =  idx + off
        #
        # so the whole 25-neighbour window is ONE addition against a constant
        # offset table.  That is worth doing precisely because it avoids `//` and
        # `%`: the ONNX lowering of integer division emits `Less`/`Equal`/`And`,
        # and boolean ops are what fails to lower on the NPU (see `simple_nms`).
        # It also removes 25 clamp pairs.
        #
        # No clamping is needed and none is applied: ALIKE zeroes a `radius + 1`
        # px border, so every surviving keypoint is at least 3 px from an edge and
        # a +/-2 window cannot reach outside the map - it therefore cannot wrap
        # into the neighbouring row, which is the failure mode the offset trick
        # would otherwise have.  Layer L1 checks this rather than trusting it.
        offsets = (self.dy * w + self.dx).reshape(1, 1, -1)             # (1, 1, 25)
        nflat = (idx[:, :, None] + offsets).reshape(
            b, 1, self.top_k * self.kernel_size ** 2)
        patch = torch.gather(flat_scores, 2, nflat)                     # (B, 1, K*25)
        patch = patch.reshape(b, self.top_k, self.kernel_size ** 2)     # (B, K, 25)

        # ---- 4. temperature soft-argmax -------------------------------------
        # Identical to upstream: subtract the detached max for stability, exponentiate
        # at T=0.1, take the weighted mean of the offset grid.  `detach()` upstream
        # guards against a gradient loop that cannot exist in inference.
        max_v = patch.max(dim=2, keepdim=True).values
        x_exp = ((patch - max_v) / self.temperature).exp()              # (B, K, 25)
        x_sum = x_exp.sum(dim=2, keepdim=True)                          # (B, K, 1)
        xy_res = (x_exp @ self.hw_grid) / x_sum                         # (B, K, 2)

        # ---- 5. sub-pixel keypoints in pixels ------------------------------
        kp_pix = torch.stack([xs.float(), ys.float()], dim=2) + xy_res  # (B, K, 2)

        # ---- 6. keypoint score: round + clip + gather ----------------------
        # The original: grid_sample(scores_map, normalise(kp_pix), bilinear,
        # align_corners=True).  `to_pixel` inverts the same normalisation, so the
        # only difference is bilinear -> nearest.  Measured as layer L1.
        x_pix, y_pix = to_pixel(self._normalise(kp_pix, h, w), h, w)
        kptscore = self.score_sample(scores_map, x_pix, y_pix)[:, 0, :]  # (B, K)

        return kp_pix, kptscore

    @staticmethod
    def _border_mask(h: int, w: int, r: int, dtype, device) -> torch.Tensor:
        """0/1 float mask, 1 in the interior.

        Upstream zeroes `r + 1` px on the top/left and `r` px on the
        bottom/right (`:radius + 1` vs `h - radius`, which is asymmetric), so the
        kept band is `r + 1 <= i <= h - r - 1`.  Reproduced exactly here.

        Built by broadcasting the OUTER PRODUCT of two 1-D ramps, not by slice
        assignment: `m[:, :, :r+1, :] = 0` lowers to `ScatterND`, and an in-place
        write is exactly the kind of op that blocks a clean NPU lowering.  The
        separable form is `GreaterOrEqual -> Cast -> Mul`, all NPU-native, and is
        bit-identical for a 0/1 mask.
        """
        ys = torch.arange(h, dtype=dtype, device=device)
        xs = torch.arange(w, dtype=dtype, device=device)
        ry = (ys >= r + 1).to(dtype) * (ys <= h - r - 1).to(dtype)
        rx = (xs >= r + 1).to(dtype) * (xs <= w - r - 1).to(dtype)
        return (ry[:, None] * rx[None, :]).reshape(1, 1, h, w)

    @staticmethod
    def _normalise(kp_pix: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Pixels -> [-1, 1] exactly as upstream (`p / (N - 1) * 2 - 1`)."""
        size = kp_pix.new_tensor([w - 1, h - 1])
        return kp_pix / size * 2.0 - 1.0
