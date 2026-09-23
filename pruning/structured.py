#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Channel-sensitivity measurement for structured pruning decisions.

WHY STRUCTURED ONLY
-------------------
Unstructured (magnitude) pruning produces sparse tensors.  RKNN - like every
other NPU toolchain - does not exploit sparsity, so a sparse model is the same
size and the same speed as a dense one while being harder to train.  Only
removing whole channels or whole layers makes anything smaller or faster, so a
pruning decision starts from whole-block sensitivity.

WHAT SURVIVES HERE, AND WHAT WAS REMOVED
----------------------------------------
Measured on this project, the only viable pruning lever was LightGlue's depth
(9->7 layers, recovered by `finetune.py`), and that path runs through
`scripts/ablate_matcher.py` + `pruning/make_pruned_ckpt.py`, not through this
module.  ALIKE channel pruning was measured and REJECTED
(the channel-pruning report): every fractional cut moves the score
map or the descriptor more than the entire fp16 conversion costs, and cutting
attention heads by weight surgery is impossible for this matcher (the two
attention blocks have incompatible layouts - see
the head-ablation report).

The head-count and channel-removal helpers that used to live here were
therefore dead code and have been deleted.  What remains is the one function
the scans actually use: zero a block's output, re-run, and measure how much the
downstream output moved - the gate that has to pass before anything is cut.

This module does not fine-tune.  Pruning moves the weights outside their
trained basin - that is what `finetune.py` is for, and the acceptance test
after recovery is the full accuracy protocol, not a loss curve.
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn

__all__ = ["channel_sensitivity"]


@torch.no_grad()
def channel_sensitivity(model: nn.Module, block_names: Sequence[str],
                        forward_fn, tol: float = 0.02) -> Dict[str, float]:
    """How much does zeroing each block's OUTPUT hurt?

    Zeroing (rather than removing) is deliberate: it keeps every shape valid, so
    one measurement costs one forward pass and needs no surgery.  The returned
    number is the relative L1 change of the downstream output, and it is what
    orders the blocks before anything is actually cut - pruning a block whose
    sensitivity is already at the tolerance cannot be recovered.

    `forward_fn(model)` must return the tensor the downstream consumer reads; for
    the ALIKE stage that is the score map, and the descriptor has to be measured
    as well - it is a second consumer of the same trunk.
    """
    base = forward_fn(model).detach()
    scale = base.abs().mean().clamp(min=1e-9)
    out = {}
    for name in block_names:
        mod = dict(model.named_modules())[name]
        orig = mod.forward

        def zero_forward(*a, _orig=orig, **kw):
            return torch.zeros_like(_orig(*a, **kw))

        mod.forward = zero_forward
        try:
            out[name] = float((forward_fn(model).detach() - base).abs().mean() / scale)
        finally:
            mod.forward = orig
    return out
