#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ALIKE backbone, vendored so the deployment project is self-contained.

Source: ALIKE's `alnet.py`, as mirrored by
`gluefactory/models/extractors/alike_sddh.py` (classes `ConvBlock`, `ResBlock`,
`ALikeNet`).  Copied rather than imported because a deployment repository should
not need the whole training framework on the target-side host, and because the
architecture is frozen: the weights this project exports were trained against
exactly this module.

Every layer is a plain convolution, BatchNorm, ReLU, MaxPool, bilinear Upsample
and Concat - all NPU-native.  There is deliberately **no** `DeformableConv2d`
here (ALIKED's blocks 3/4 use one; ALIKE's do not), which is the single most
important structural fact for RKNN deployment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvBlock", "ResBlock", "ALikeNet", "ALIKE_CONFIGS"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gate=None, norm_layer=None):
        super().__init__()
        self.gate = nn.ReLU(inplace=True) if gate is None else gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))
        x = self.gate(self.bn2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    """Copied from torchvision's BasicBlock, as ALIKE does."""

    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, gate=None, norm_layer=None):
        super().__init__()
        self.gate = nn.ReLU(inplace=True) if gate is None else gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ResBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.gate(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.gate(out)


# ALIKE's variant table, verbatim from ALIKE's `alnet.py` as mirrored by
# `gluefactory/models/extractors/alike_sddh.py::ALikeSDDH.cfgs`.
#
# NOTE, because it is easy to get wrong: this is NOT ALIKED's table.  ALIKED's
# `aliked-n16` uses (c1, c2, c3, c4) = (16, 32, 64, 128) with `dim=128`, which
# happens to coincide with ALIKE's `alike-n` - but ALIKED's `aliked-t16` is
# (8, 16, 32, 64) with dim 64 and its `aliked-n32` reuses n16's block widths with
# a different SDDH geometry, so the two tables are not interchangeable.  An
# earlier revision of this file carried ALIKED's numbers under ALIKE's names and
# the load failed with channel-size mismatches on block2 onward.
ALIKE_CONFIGS = {
    "alike-t": {"c1": 8,  "c2": 16, "c3": 32,  "c4": 64,  "dim": 64,  "single_head": True},
    "alike-s": {"c1": 8,  "c2": 16, "c3": 48,  "c4": 96,  "dim": 96,  "single_head": True},
    "alike-n": {"c1": 16, "c2": 32, "c3": 64,  "c4": 128, "dim": 128, "single_head": True},
    "alike-l": {"c1": 32, "c2": 64, "c3": 128, "c4": 128, "dim": 128, "single_head": False},
}


class ALikeNet(nn.Module):
    """Backbone + single head, returning `(scores_map, descriptor_map_raw)`.

    The descriptor map is **not** normalised here - that happens in
    `AlikeStage`, exactly as ALIKE's `extract_dense_map` does it.  Keeping the
    split in the same place matters because the downstream matcher was trained
    against the normalised map.
    """

    def __init__(self, c1: int = 32, c2: int = 64, c3: int = 128, c4: int = 128,
                 dim: int = 128, single_head: bool = True) -> None:
        super().__init__()
        self.gate = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock(3, c1, self.gate, nn.BatchNorm2d)
        self.block2 = ResBlock(c1, c2, stride=1, downsample=nn.Conv2d(c1, c2, 1),
                               gate=self.gate, norm_layer=nn.BatchNorm2d)
        self.block3 = ResBlock(c2, c3, stride=1, downsample=nn.Conv2d(c2, c3, 1),
                               gate=self.gate, norm_layer=nn.BatchNorm2d)
        self.block4 = ResBlock(c3, c4, stride=1, downsample=nn.Conv2d(c3, c4, 1),
                               gate=self.gate, norm_layer=nn.BatchNorm2d)

        self.conv1 = conv1x1(c1, dim // 4)
        self.conv2 = conv1x1(c2, dim // 4)
        self.conv3 = conv1x1(c3, dim // 4)
        self.conv4 = conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)

        self.single_head = single_head
        if not self.single_head:
            self.convhead1 = conv1x1(dim, dim)
        self.convhead2 = conv1x1(dim, dim + 1)

    def trunk(self, image):
        """Everything up to and including the head's 1x1 prep, i.e. `convhead2`'s input.

        Split out so the descriptor can be computed only where it is read (see
        `gathered_head.GatheredDescriptorHead`).  `forward` calls this too, so the
        two paths cannot drift: there is one copy of the trunk, not two.
        """
        x1 = self.block1(image)
        x2 = self.block2(self.pool2(x1))
        x3 = self.block3(self.pool4(x2))
        x4 = self.block4(self.pool4(x3))

        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))
        x1234 = torch.cat([x1, self.upsample2(x2), self.upsample8(x3),
                           self.upsample32(x4)], dim=1)

        if not self.single_head:
            x1234 = self.gate(self.convhead1(x1234))
        return x1234

    def score_map(self, x1234):
        """The score channel, densely - the only channel that must be dense.

        NMS and top-k act on the whole map, so this one cannot be sampled.
        `convhead2`'s LAST output row is the score; the first `dim` rows are the
        descriptor.
        """
        w = self.convhead2.weight[-1:]
        b = self.convhead2.bias
        b = b[-1:] if b is not None else None
        return torch.sigmoid(F.conv2d(x1234, w, b))

    def forward(self, image):
        x1234 = self.trunk(image)
        x = self.convhead2(x1234)
        descriptor_map = x[:, :-1, :, :]
        scores_map = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)
        return scores_map, descriptor_map


def build_backbone(variant: str = "alike-n", **overrides) -> ALikeNet:
    if variant not in ALIKE_CONFIGS:
        raise KeyError(f"unknown ALIKE variant {variant!r}; "
                       f"choose from {sorted(ALIKE_CONFIGS)}")
    cfg = dict(ALIKE_CONFIGS[variant])
    cfg.update(overrides)
    return ALikeNet(**cfg)
