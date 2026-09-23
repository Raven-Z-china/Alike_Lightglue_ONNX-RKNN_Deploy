"""Export-ready model definitions for ALIKE + LightGlue.

Two exportable stages:

    AlikeStage      backbone + DKD + descriptor sampling   (image -> kpts, desc)
    LightGlueStage  matcher                                (kpts, desc -> matches)

Both are free of `grid_sample`, `Unfold`/`Im2Col`, `NonZero` and any
data-dependent shape, which is what makes them convertible to RKNN and fully
NPU-resident.

Typical use - load a trained checkpoint and export it:

    from model import load_alike_stage, load_lightglue_stage

    stage1 = load_alike_stage("checkpoint_best.tar", top_k=512)
    stage2 = load_lightglue_stage("checkpoint_best.tar")

The individual building blocks (`ALikeNet`, `DKDExport`, the gather samplers,
...) are importable from their submodules; only the two entry points every
caller needs are re-exported here.
"""
from .alike_stage import AlikeStage, load_from_checkpoint as load_alike_stage
from .lightglue_stage import LightGlueStage, load_from_checkpoint as load_lightglue_stage

__all__ = ["AlikeStage", "LightGlueStage",
           "load_alike_stage", "load_lightglue_stage"]
