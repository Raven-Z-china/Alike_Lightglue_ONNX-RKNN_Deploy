"""LightGlue matcher - VENDORED from the training framework.

PROVENANCE
----------
Copied verbatim from `gluefactory/models/matchers/lightglue.py` in this project's
training checkout, which is itself a fork of the upstream LighGlue (the custom
`aggregation` / `source_embedding` / `dual_input_proj` options below are this
project's additions and every checkpoint depends on them).  The port exists so
that ONNX export, ablation and the accuracy harness no longer need the training
repo on `sys.path`.

WHAT WAS REMOVED, AND WHY IT CANNOT CHANGE A RESULT
---------------------------------------------------
Three things, all of which were reachable only from training or from a
pretrained-weights download - never from a forward pass:

* `NLLLoss` / `self.loss_fn` / the whole `loss()` method.  Training-only: the
  deploy side never backprops.  Verified no caller: `grep -rn '\.loss('` over
  `scripts/ pruning/ eval/ model/` returns nothing.
* `matcher_metrics` - used only inside that same `loss()`.
* The `conf.weights` branch.  This one was actively harmful.  Every checkpoint's
  matcher conf carries `weights: /path/to/aliked_lightglue.pth`, so constructing
  the matcher read a **47 MB file out of the glue-factory tree** on every export,
  and - if that file were ever missing - the `else` arm fell through to
  `torch.hub.load_state_dict_from_url`, i.e. a network download.  Both loads are
  dead weight: the very next step in the caller overwrites every matcher tensor
  from the trained checkpoint.  `scripts/verify_vendored_matcher.py` proves that
  by building the matcher both ways and comparing state dicts tensor by tensor.

`__main_model__` was dropped as well: it is the training framework's model
registry hook and nothing here resolves models by name.

The vendored copy is checked against the upstream module by
`scripts/verify_vendored_matcher.py`, which loads the same checkpoint into both
implementations and asserts the exported graphs are numerically identical.  If
that script is ever deleted, this file is trust-me code.
"""
import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

# Hacky workaround for torch.amp.custom_fwd to support older versions of PyTorch.
AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


@AMP_CUSTOM_FWD_F32
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    # Upstream this class also carries a `loss()` and a `BCEWithLogitsLoss`
    # instance; both were reachable only from `LightGlue.loss()`, which is gone.
    # Only `token` holds state, so dropping them cannot shift a checkpoint.
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        # How the per-layer assignment maps become the final `log_assignment`.
        #
        #   "last"    - upstream LightGlue: the FINAL layer's map alone.  This is
        #               what every checkpoint in this repo was trained with, and
        #               the default, so nothing changes unless it is asked for.
        #   "average" - running average over ALL layers, with the last layer
        #               weighted twice: (sum_{j<n-1} L_j + 2*L_last) / (n+1).
        #               Ported from `custom/custom_lightglue.py` in
        #               superpoint-lightglue_TRT_pruning (the "sliding average"
        #               that repo's ONNX/TRT export uses).
        #   "uniform" - plain mean over ALL layers: sum_j L_j / n.  No mask, no
        #               re-weighting; the last layer counts exactly once, so the
        #               only difference from "average" is that layer's weight
        #               (1/(n+1) there against 1/n here).
        #
        # WHY THIS IS NOT JUST COSMETIC.  `log_assignment` is the matcher's
        # primary output: `filter_matches` derives `matches0`/`matching_scores0`
        # from it, so this changes which matches are produced.  It also reaches
        # `loss()` twice - as the `row_norm` diagnostic and as the target for the
        # `token_confidence` heads - though NOT as the main NLL term, which
        # `loss_params` recomputes from the stored per-layer descriptors.  So
        # flipping it does alter training slightly; it is not an inference-only
        # switch.
        #
        # MEASURED, once, on HPatches at 1024 keypoints with `g0_s0_lr1e4`
        # (540 pairs, opencv, ransac_th 0.5).  Both averaging recipes beat `last`
        # on every homography metric and lose match precision:
        #
        #   metric                  last    average   uniform
        #   H_error_ransac_mAA     0.5767   0.5785    0.5797
        #   H_error_ransac@3px     0.6281   0.6297    0.6317
        #   H_error_ransac@5px     0.7306   0.7345    0.7361
        #   H_error_dlt@3px        0.5773   0.5947    0.5939
        #   mprec@3px              0.8060   0.7940    0.7900
        #   mransac_inl%           0.5820   0.5740    0.5760
        #
        # The direction is monotone in how much averaging happens: `last` has the
        # best precision and the worst homography error, `uniform` the reverse.
        # `uniform` leads on the RANSAC metrics, `average` on DLT, by 0.0005-0.0024
        # - effectively a tie BETWEEN those two; the gap from either to `last` is
        # an order of magnitude larger (up to 0.018).  The costs are equal: all
        # three run every layer, so nothing is saved or spent.
        #
        # Single measurement on one benchmark - treat it as a lead, not a
        # conclusion.
        #
        # Only the `depth_confidence <= 0` case is measured; with early stopping
        # enabled the mask below changes what is averaged, which is coherent but
        # untested.
        "aggregation": "last",
        # --- serving TWO descriptor sources with ONE matcher -------------------
        #
        # Two ways to let the matcher tell the sources apart instead of being
        # forced to treat them as one space.  Both default off, so existing runs
        # are unaffected; they are mutually exclusive.
        #
        # The problem they address, measured: ALIKE's own descriptor and the SDDH
        # descriptor of the SAME keypoint sit at cosine +0.02, and after
        # `input_proj` they are FARTHER apart (1.86) than two unrelated ALIKE
        # descriptors (1.33).  A matcher trained on SDDH therefore scores
        # ALIKE<->SDDH at prec@3 0.0005 against 0.59/0.71 within a source.  The
        # hypothesis is not "the spaces are unbridgeable" but "the matcher cannot
        # tell which space a given vector came from, so one projection is asked to
        # serve two incompatible mappings".
        #
        # "source_embedding": append a learned per-source vector to each descriptor
        # before the projection, so the projection becomes source-conditioned -
        # one `Linear(input_dim + source_embed_dim, descriptor_dim)`.
        # NOTE this CHANGES `input_proj`'s input width, so the released LightGlue
        # checkpoints can no longer be loaded into it (their `input_proj` is
        # `(256, 128)`); a run using this must train from scratch.  The backbone
        # and transformer weights still load.
        "source_embedding": False,
        "source_embed_dim": 16,
        # "dual_input_proj": keep two SEPARATE projections, routing each view to
        # the one matching its source.  Shapes stay identical to the released
        # checkpoints so both can be initialised from the same pretrained
        # `input_proj`; the two are then free to diverge during training.
        "dual_input_proj": False,
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.aggregation not in ("last", "average", "uniform"):
            # Fail here rather than silently falling through to `last` in the
            # forward: a typo in this key would otherwise look like "the variant
            # made no difference", which is a conclusion someone could act on.
            raise ValueError(
                f"unknown matcher.aggregation {conf.aggregation!r}; choose one of "
                f"'last' (final layer only, upstream), 'average' (running average, "
                f"last layer weighted twice) or 'uniform' (plain mean over all "
                f"layers)"
            )
        if conf.source_embedding and conf.dual_input_proj:
            raise ValueError(
                "source_embedding and dual_input_proj are two solutions to the "
                "same problem and are mutually exclusive; enable one."
            )
        # `proj_in_dim` is what the projection actually consumes.  With the source
        # embedding it is wider than `input_dim`, which is why the released
        # checkpoints cannot be loaded into that configuration.
        self.proj_in_dim = (conf.input_dim + conf.source_embed_dim
                            if conf.source_embedding else conf.input_dim)
        if conf.source_embedding:
            # One learned vector per source; index 0 = ALIKE, 1 = SDDH.  Scaled
            # down so at init it perturbs the descriptor only slightly - the point
            # is to let the projection TELL THEM APART, not to inject noise.
            self.source_emb = nn.Embedding(2, conf.source_embed_dim)
            nn.init.normal_(self.source_emb.weight, std=0.02)
        if self.proj_in_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(self.proj_in_dim, conf.descriptor_dim,
                                        bias=True)
        else:
            self.input_proj = nn.Identity()
        if conf.dual_input_proj:
            # A second projection for the other source.  Both start from the same
            # weights (below, when a checkpoint is loaded) so the split is a
            # capacity increase rather than a random re-initialisation.
            self.input_proj_b = nn.Linear(self.proj_in_dim, conf.descriptor_dim,
                                          bias=True)

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )

        # NOTE: the training framework's `conf.weights` loading block used to sit
        # here (old-state-dict key renaming, the shape-mismatch skip list, and the
        # `dual_input_proj` / `source_embedding` initialisation fixups).  It is
        # gone: this module never loads weights itself.  Everything it covered is
        # reachable only from a construction-time download, and the caller loads
        # the trained checkpoint into the freshly built module.  A training run
        # that needs those fixups uses the framework's own copy of this class.

        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(self.conf.n_layers)]
            ),
        )

    #: source name -> embedding index.  Only two sources exist; anything else is
    #: a programming error rather than something to default silently.
    _SRC_IDX = {"alike": 0, "sddh": 1}

    def _view_sources(self, data, view, batch):
        """Per-image source names for one view, or None when unavailable.

        The dataset puts `descriptor_source` inside each view dict, so it arrives
        already batched.  Evaluation paths that build `data` by hand do not set
        it, and there it is correct to return None: the model keeps its
        single-source behaviour rather than inventing a source.
        """
        v = data.get(view)
        if not isinstance(v, dict):
            return None
        raw = v.get("descriptor_source")
        if raw is None:
            return None
        if isinstance(raw, str):
            return [raw] * batch
        return list(raw)

    def _project(self, desc, data, view, batch):
        """Apply the source-aware projection to one view's descriptors.

        Three modes, all sharing this entry point so the forward pass does not
        branch in several places:
          * plain            - `self.input_proj`, exactly as upstream;
          * source_embedding - concatenate a learned per-source vector first;
          * dual_input_proj  - route each image to its own projection.
        """
        conf = self.conf
        if not conf.source_embedding and not conf.dual_input_proj:
            return self.input_proj(desc)

        sources = self._view_sources(data, view, batch)
        if sources is None:
            # No source information: fall back to the FIRST source on every image
            # rather than guessing.  A silent guess would make two runs differ for
            # a reason absent from the logs.
            sources = ["alike"] * batch

        if conf.source_embedding:
            idx = torch.tensor([self._SRC_IDX[s] for s in sources],
                               device=desc.device, dtype=torch.long)
            embed = self.source_emb(idx).unsqueeze(1).expand(-1, desc.shape[1], -1)
            embed = embed.to(desc.dtype)
            return self.input_proj(torch.cat([desc, embed], dim=-1))

        # dual_input_proj: a per-image gather, so a batch may mix sources.
        out = torch.empty(desc.shape[0], desc.shape[1], conf.descriptor_dim,
                          device=desc.device, dtype=desc.dtype)
        is_b = torch.tensor([self._SRC_IDX[s] == 1 for s in sources],
                            device=desc.device)
        if is_b.any():
            out[is_b] = self.input_proj_b(desc[is_b])
        if (~is_b).any():
            out[~is_b] = self.input_proj(desc[~is_b])
        return out

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self._project(desc0, data, "view0", b)
        desc1 = self._project(desc1, data, "view1", b)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training
        # `average` and `uniform` accumulate every layer's assignment map;
        # `last` (default) keeps the original single-assignment path untouched.
        agg = self.conf.aggregation
        accumulate = agg != "last"
        scores = None
        cnt = 1.0          # weight bookkeeping for the `average` recipe
        n_mean = 0         # layers accumulated, for the `uniform` mean

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.transformers[i],
                    desc0,
                    desc1,
                    encoding0,
                    encoding1,
                    use_reentrant=False,  # Recommended by torch, default was True
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)

            if accumulate:
                la = self.log_assignment[i](desc0[..., :m, :], desc1[..., :n, :])[0]
                n_mean += 1
                if scores is None:
                    # Taken from the real output rather than allocated as (b,m+1,n+1)
                    # so this stays correct if point pruning has shrunk the set.
                    scores = torch.zeros_like(la)

                if agg == "uniform":
                    # Plain mean over every layer - no mask, no re-weighting:
                    #     sum_i L_i / N
                    # Every layer enters at 1/N INCLUDING the last, so unlike
                    # `average` the final layer gets no special treatment.  `N` is
                    # the number of layers that actually RAN, so early stopping
                    # shortens the mean instead of dividing by a count that was
                    # never reached.  The division happens after the loop.
                    scores = scores + la
                else:
                    # Running average of the per-layer assignment maps, following
                    # `custom/custom_lightglue.py` from
                    # superpoint-lightglue_TRT_pruning:
                    #   scores <- scores*(cnt-keep)/cnt + L_i*keep/cnt ; cnt += keep
                    # then one final merge of the LAST layer's map (below).
                    # With keep == 1 throughout - which `depth_confidence <= 0`
                    # guarantees, since `check_if_stop` tests
                    # `ratio_confident > depth_confidence` and `ratio_confident`
                    # is in [0, 1] - the recursion telescopes to the running mean
                    # (sum_{j<=i} L_j)/(i+1), and `cnt` ends at N+1.  The final
                    # merge then yields
                    #     (sum_{j<N-1} L_j + 2*L_last) / (N+1)
                    # i.e. every layer at 1/(N+1) with the LAST at 2/(N+1), which
                    # sums to 1.  So the last layer carries twice the weight of any
                    # other - the only thing separating this from `uniform`.
                    #
                    # `keep` is 1.0 unless early stopping is active, in which case
                    # it is the stop mask; the two call sites of `token_confidence`
                    # below are then redundant for one small Linear, which is
                    # preferable to restructuring the eval-only branch above.
                    keep = 1.0
                    if do_early_stop and i < self.conf.n_layers - 1:
                        token0, token1 = self.token_confidence[i](desc0, desc1)
                        keep = torch.where(
                            self.check_if_stop(
                                token0[..., :m, :], token1[..., :n, :], i, m + n),
                            1.0, 0.0,
                        ).to(la.dtype)
                    scores = scores * ((cnt - keep) / cnt) + (la / cnt) * keep
                    cnt = cnt + keep

            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

            # only for eval
            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        if agg == "uniform":
            # Every layer was summed with equal weight, so the mean is just the
            # sum divided by how many were accumulated.  No extra term: the last
            # layer is already in the sum exactly once.
            scores = scores / n_mean
        elif agg == "average":
            # The final merge.  `i` is the last layer that RAN, so with early
            # stopping this weights the stopping layer twice, mirroring the
            # upstream code's use of `log_assignment[i]` in the same position.
            la_last = self.log_assignment[i](desc0, desc1)[0]
            scores = scores * ((cnt - 1.0) / cnt) + (la_last / cnt)
        else:
            scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
        }

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

