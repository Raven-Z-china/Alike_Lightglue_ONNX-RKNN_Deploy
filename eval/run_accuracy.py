#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end accuracy: the two converted stages, chained, against the torch model.

WHY THIS IS NOT `scripts/check_accuracy.py`
-------------------------------------------
`check_accuracy.py` scores ONE stage against a torch reference on tensors.  That is
the right tool to localise a conversion defect, and it is what produced the
per-stage numbers in the README.  It is the wrong tool to answer the deployment
question, which is:

    does the C++ pipeline that will run on the board produce the same matches
    as the trained model?

That question needs the two stages CHAINED exactly as the C++ sample chains them -
stage-1 keypoints/descriptors straight into stage 2, no torch anywhere in the
loop - and then scored with a metric that is invariant to the things that are
allowed to differ (keypoint order, a handful of near-tie ranks flipping).

WHAT IS MEASURED
----------------
The **HPatches** benchmark (multiple public viewpoint sequences, 540 pairs,
homography ground truth) - an open-source dataset, the same one the training-side
reports use, so the numbers are directly comparable.

      mAA        : mean average accuracy of the estimated homography
      @3px / @5px: mAA restricted to a reprojection tolerance
      mprec@3px  : of the matches produced, the fraction whose endpoints agree
                   with the ground-truth homography within 3 px
      n_inl_gt   : mprec@3px x n_match, i.e. the deterministic inlier count
      n_inl      : RANSAC inlier count (what `prec@k` cannot express: a matcher
                   that emits fewer matches can win on precision and lose on the
                   number of usable correspondences)

WHY A HOMOGRAPHY ACCURACY, AND NOT A DETECTION+MATCHING SCORE ALONE
-------------------------------------------------------------------
`mAA` scores the whole chain through a geometric estimator, which is what the
deployment question is about; `mprec@3px` and `n_inl_gt` isolate the matcher from
the estimator.  A conversion that broke the descriptor lookup would move the
precision figure first; a conversion that broke the geometry would move both.
Reading only one of them is how a real regression gets missed.

THE THREE COLUMNS
-----------------
    torch  the trained checkpoint, through `model.load_*_stage` (self-contained)
    onnx   the exported graph(s), fp32, onnxruntime CPU
    rknn   the converted .rknn model(s), RKNN PC simulator, fp16

`--pipeline` selects which of the last two is run against torch; running all
three in one command is what makes the attribution possible, and it is the
default.

ON THE SIMULATOR
----------------
The RKNN "simulator" (`init_runtime(target=None)`) runs the converted graph on the
x86 host with the NPU's arithmetic.  It validates NUMERICS.  It says nothing about
latency - a simulator timing is the speed of the host CPU running an emulation of
the NPU's op sequence, which is not even the right order of magnitude.  No number
in the output of this script is a speed claim.

Usage -- the two arms live in different envs, so they are two runs: the `rknn`
arm has no omegaconf and cannot build the torch reference.  `--ref-from` reads
the reference arm's JSON so both halves score the SAME pairs.

    # in the export env
    python eval/run_accuracy.py \
        --checkpoint <d7 checkpoint, from the training checkout> \
        --pipeline torch onnx \
        --hpatches 540 --size 512 --keypoints 512 \
        --json <reports>/e2e_ta.json --report <reports>/e2e_ta.md
    # in the rknn env
    python eval/run_accuracy.py --pipeline rknn-fp16 \
        --ref-from <reports>/e2e_ta.json \
        --json <reports>/e2e_rknn.json --report <reports>/e2e_rknn.md
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import paths  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from portable_path import portable  # noqa: E402
from model import load_alike_stage, load_lightglue_stage  # noqa: E402


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #
class TorchPipeline:
    """The trained model, as trained - the reference everything is scored against.

    Every pipeline here exposes the deployment's two steps separately:
    `extract(image)` is per-image and can be cached, `match(f0, f1)` consumes two
    extractions.  `__call__` is the convenience path and is what the harness uses.
    """

    name = "torch"

    def __init__(self, checkpoint, size, keypoints):
        self.stage1 = load_alike_stage(checkpoint, top_k=keypoints,
                                       descriptor_interp="bilinear").eval()
        self.stage2 = load_lightglue_stage(checkpoint,
                                           image_size=(size, size)).eval()

    @torch.no_grad()
    def extract(self, rgb_u8):
        """ONE HWC uint8 colour image -> (k (1,K,2), d (1,K,128))."""
        img = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        k, d, _ = self.stage1(img)
        return k, d

    @torch.no_grad()
    def match(self, f0, f1):
        m0, ms0 = self.stage2(f0[0], f1[0], f0[1], f1[1])
        return m0[0], ms0[0]

    @torch.no_grad()
    def __call__(self, rgb0_u8, rgb1_u8):
        """Two HWC uint8 colour images -> (k0, k1, m0, ms0) in torch."""
        # ONE IMAGE AT A TIME: two extractions, then one match.
        (k0, d0), (k1, d1) = self.extract(rgb0_u8), self.extract(rgb1_u8)
        m0, ms0 = self.match((k0, d0), (k1, d1))
        return k0[0], k1[0], m0, ms0

    def release(self):
        pass


class OnnxPipeline:
    """The exported graphs, fp32, onnxruntime CPU.

    Deliberately NOT the CUDA provider: the RKNN simulator is an x86 CPU
    emulation of the NPU's arithmetic, and running the ONNX arm on the GPU would
    put a third arithmetic in the comparison for no benefit.
    """

    name = "onnx"

    def __init__(self, s1_path, s2_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s1 = ort.InferenceSession(str(s1_path), so,
                                       providers=["CPUExecutionProvider"])
        self.s2 = ort.InferenceSession(str(s2_path), so,
                                       providers=["CPUExecutionProvider"])

    def extract(self, rgb_u8):
        """ONE HWC uint8 colour image -> (k, d) as numpy (1,K,2)/(1,K,128)."""
        img = rgb_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        k, d, _ = self.s1.run(None, {"image": img})
        return k, d

    def match(self, f0, f1):
        return self.s2.run(None, {
            "keypoints0": f0[0], "keypoints1": f1[0],
            "descriptors0": f0[1], "descriptors1": f1[1]})

    def __call__(self, rgb0_u8, rgb1_u8):
        # ONE IMAGE AT A TIME: two extractions, then one match.
        (k0, d0) = self.extract(rgb0_u8)
        (k1, d1) = self.extract(rgb1_u8)
        m0, ms0 = self.match((k0, d0), (k1, d1))
        return (torch.from_numpy(k0[0]), torch.from_numpy(k1[0]),
                torch.from_numpy(m0[0]), torch.from_numpy(ms0[0]))

    def release(self):
        pass


class TrtPipeline:
    """The shipped TensorRT engines, fp16 by default.

    Why this arm exists at all: the engines under `weights/` were verified
    per-graph against a torch dump (`scripts/convert_to_trt.py --ref`), and a
    per-graph check is NOT an end-to-end number - it says nothing about inliers or
    match agreement on the benchmark.  This is the column that answers "what does
    the fp16 engine score on HPatches", on the same pairs, with the same staged
    error accumulation as the other arms.

    fp16 IS A DECISION, not an inherited setting: `--trt-precision` selects the
    engine, the value is recorded in the report, and an fp16 engine answering a
    question about fp32 accuracy would be worse than no column at all.
    """

    def __init__(self, s1_engine, s2_engine, name="trt-fp16", precision="fp16"):
        self.name = name
        self.precision = precision
        self.s1_path, self.s2_path = str(s1_engine), str(s2_engine)
        self.trt = self.torch = None
        self._engines, self._ctx, self._bufs = {}, {}, {}
        self.stream = None

    def _load(self, role, path):
        """Deserialise lazily: engine load is slow and the arm may not be reached."""
        import tensorrt as trt
        import torch

        if role in self._ctx:
            return
        if self.trt is None:
            self.trt, self.torch = trt, torch
        logger = trt.Logger(trt.Logger.ERROR)
        engine = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
        if engine is None:
            raise SystemExit(
                f"{path}: engine failed to deserialise.  An engine is compiled for "
                f"one GPU/TRT/CUDA combination, so this one either was built "
                f"elsewhere or the runtime here differs; rebuild it with "
                f"scripts/convert_to_trt.py --precision {self.precision}")
        self._engines[role] = engine
        self._ctx[role] = engine.create_execution_context()
        if self.stream is None:
            self.stream = torch.cuda.Stream()       # one stream, held for the run

    def _buffers(self, role):
        """Per-role device buffers, allocated once and OWNED BY THIS PIPELINE.

        Both of the things that can go wrong here are silent, and both did:

        * EVERY BUFFER IS ALLOCATED WITH THE DTYPE THE ENGINE DECLARES.  The first
          version hardcoded float32 for every input and output and produced 92 NaN
          entries in matches0 with index 0 for all 420 survivors, plus a match image
          whose lines all converged on one keypoint.  The cause was one tensor:
          `matches0` is **int64** in the engine, so TRT wrote 4096 bytes into a 2048
          byte float32 buffer and the result was read back as reinterpreted bits.
          `trt.nptype(engine.get_tensor_dtype(...))` is the only correct source for
          this - a buffer whose dtype merely "looks right" for the feature it holds
          is a memory error waiting to happen.
        * Buffers are held on the pipeline rather than in locals.  TRT keeps bare
          device ADDRESSES in the execution context, the context itself rejects
          `setattr` (it is a C++ object), and a freed tensor's memory returns to the
          allocator immediately - so anything released before the outputs are copied
          out can be handed to the next allocation and read back as garbage.
          Reuse across pairs is safe: every input is rewritten by the copy in `_run`
          and every output by `execute_async_v3`.
        """
        if role in self._bufs:
            return self._bufs[role]
        torch, trt = self.torch, self.trt
        engine = self._engines[role]
        dev_in, dev_out, np_dtype = {}, {}, {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            np_dtype[name] = trt.nptype(engine.get_tensor_dtype(name))
            # torch dtype straight from numpy's, so a kHALF / kINT64 tensor gets a
            # buffer of the right WIDTH - see the docstring for what a wrong width
            # did.
            tdtype = torch.from_numpy(np.empty(0, dtype=np_dtype[name])).dtype
            buf = torch.empty(shape, dtype=tdtype, device="cuda")
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                dev_in[name] = buf
            else:
                dev_out[name] = buf
        self._bufs[role] = (dev_in, dev_out, np_dtype)
        return dev_in, dev_out, np_dtype

    def _run(self, role, feeds):
        """Execute `role` on `feeds`; return outputs as {tensor_name: ndarray}."""
        torch, trt = self.torch, self.trt
        ctx = self._ctx[role]
        dev_in, dev_out, np_dtype = self._buffers(role)
        for name, arr in feeds.items():
            if name not in dev_in:
                raise SystemExit(f"{role}: engine has no input named {name!r}; it "
                                 f"takes {sorted(dev_in)}")
            # The input is cast to what the ENGINE declared, not to float32: the
            # graph's own dtype for `matches0` is int64, so a blanket float32 cast
            # is not a safe default even on the way in.
            want = np_dtype[name]
            dev_in[name].copy_(torch.from_numpy(
                np.ascontiguousarray(arr.astype(want))), non_blocking=True)
            ctx.set_tensor_address(name, dev_in[name].data_ptr())
        for name, dev in dev_out.items():
            ctx.set_tensor_address(name, dev.data_ptr())
        stream = self.stream
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise SystemExit("execute_async_v3 returned false")
        stream.synchronize()
        return {name: dev.cpu().numpy() for name, dev in dev_out.items()}

    def extract(self, rgb_u8):
        """ONE HWC uint8 colour image -> (k, d) as numpy (1,K,2)/(1,K,128).

        Selected BY THE ENGINE'S OWN OUTPUT NAMES, not by shape.  A shape heuristic
        was tried first and it is not safe here: it mixed up stage 1's outputs and
        produced keypoints that differed from the ONNX arm by up to 496 px, i.e. it
        silently scored the wrong tensor rather than failing.  `_pick` therefore
        prefers the names and only falls back to shape when they are absent.
        """
        self._load("stage1", self.s1_path)
        img = rgb_u8.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        got = self._run("stage1", {"image": img})
        k = self._pick(got, "keypoints", (1, 512, 2), "stage1")
        d = self._pick(got, "descriptors", (1, 512, 128), "stage1")
        return k, d

    @staticmethod
    def _pick(got, name, want_shape, role):
        """The named output, or - if the engine has no such name - the one matching
        `want_shape`, with a hard error when that is ambiguous."""
        if name in got:
            if got[name].shape != want_shape:
                raise SystemExit(f"{role}: output {name!r} has shape "
                                 f"{got[name].shape}, expected {want_shape}")
            return got[name]
        cands = [k for k, v in got.items() if v.shape == want_shape]
        if len(cands) != 1:
            raise SystemExit(
                f"{role}: no output named {name!r} and {len(cands)} outputs match "
                f"{want_shape} ({cands}); outputs are "
                f"{[(k, v.shape) for k, v in got.items()]}")
        warn_once(f"{role}: engine has no output named {name!r}; falling back to "
                  f"shape {want_shape} -> {cands[0]!r}")
        return got[cands[0]]

    def match(self, f0, f1):
        """The pair's two matches arrays.

        Both outputs are `(1, 512)` - one int64, one float32 - so SHAPE CANNOT TELL
        THEM APART, and a shape-keyed dict silently keeps only the last one.  That
        is how an early version of this arm reported "512 valid matches" for every
        pair: it was reading the engine's float output out of the matches slot, and
        the int64 one had never reached it at all.  The engine carries the ONNX
        output names, so those are used first; the fallback is the same content
        discriminator `cpp/infer.cpp` relies on (`matches0` holds keypoint indices).
        An unresolvable pair is an error, never a coin flip.
        """
        self._load("stage2", self.s2_path)
        feeds = {"keypoints0": f0[0], "keypoints1": f1[0],
                 "descriptors0": f0[1], "descriptors1": f1[1]}
        got = self._run("stage2", feeds)
        pairs = [(k, v) for k, v in got.items() if v.shape == (1, 512)]
        if len(pairs) != 2:
            raise SystemExit(f"stage2: expected two (1,512) outputs, got "
                             f"{[(k, v.shape) for k, v in got.items()]}")

        by_name = dict(pairs)
        if "matches0" in by_name and "mscores0" in by_name:
            return by_name["matches0"], by_name["mscores0"]

        def integral(v):
            f = v[v >= 0]
            return f.size > 0 and np.all(f == np.floor(f))

        matches = [kv for kv in pairs if integral(kv[1])]
        scores = [kv for kv in pairs if not integral(kv[1])]
        if len(matches) != 1 or len(scores) != 1:
            raise SystemExit(
                f"stage2: outputs are unlabelled {[k for k, _ in pairs]} and cannot "
                f"be told apart by content either; the engine or the graph changed, "
                f"so this is not a case to resolve by guessing")
        return matches[0][1], scores[0][1]

    def __call__(self, rgb0_u8, rgb1_u8):
        # ONE IMAGE AT A TIME: two extractions, then one match.
        (k0, d0) = self.extract(rgb0_u8)
        (k1, d1) = self.extract(rgb1_u8)
        m0, ms0 = self.match((k0, d0), (k1, d1))
        # The engine declares `matches0` as int64 with -1 as the "no match" sentinel,
        # so this should be a straight reinterpretation.  The checks below are for
        # the cases where that is NOT true - a differently exported graph, or a
        # float output cast straight to int64 (which turns -1.0 into INT64_MIN, and
        # NaN into index 0).  Both produced plausible-looking tables; neither is
        # allowed through without a warning.
        raw = np.asarray(m0[0])
        if raw.dtype.kind == "f":
            warn_once("stage2: matches0 arrived as float; converting with the -1 "
                      "sentinel handled explicitly")
            finite = np.isfinite(raw)
            if not finite.all():
                warn_once(f"stage2: {int((~finite).sum())} of {raw.size} matches0 "
                          f"entries are not finite; treated as 'no match'")
            raw = np.where(finite & (raw >= 0), raw, -1.0).astype(np.int64)
        matches = raw.astype(np.int64, copy=True)
        # Guard against an index the keypoints cannot be indexed with: a bad index
        # must not be able to reach the metric as a "match".
        over = matches > kKeypointsGlobal - 1
        if over.any():
            warn_once(f"stage2: {int(over.sum())} matches0 entries exceed the "
                      f"keypoint count ({kKeypointsGlobal}); treated as 'no match'")
            matches[over] = -1
        if (matches < -1).any():
            warn_once(f"stage2: {int((matches < -1).sum())} matches0 entries are "
                      f"below -1 (the sentinel); treated as 'no match'")
            matches[matches < -1] = -1
        return (torch.from_numpy(k0[0]), torch.from_numpy(k1[0]),
                torch.from_numpy(matches),
                torch.from_numpy(np.asarray(ms0[0], dtype=np.float32)))

    def release(self):
        self._engines.clear()
        self._ctx.clear()


class RknnPipeline:
    """The converted models, through the RKNN PC simulator.

    A fresh runtime is created per call rather than once: the simulator caches
    its context, and reusing one across hundreds of pairs grows host memory
    without bound.  The per-call cost is real but this is an accuracy harness,
    not a benchmark - correctness first, and the timing is reported separately by
    `scripts/bench_stages.py`.
    """

    def __init__(self, s1_path, s2_path, name="rknn-fp16",
                 s1_onnx=None, s2_onnx=None):
        if s1_onnx is None or s2_onnx is None:
            raise ValueError(
                "the simulator rebuilds from ONNX, so s1_onnx/s2_onnx are "
                "required - pass the pair the .rknn files were built from")
        self.name = name
        # ONNX + build, NOT `load_rknn`.  This is forced by the toolkit, and the
        # error is worth recording because the obvious route is the wrong one:
        #
        #     load_rknn(model.rknn) + init_runtime(target=None)
        #         -> "RKNN model that loaded by 'load_rknn' not support inference on
        #             the simulator, please set 'target' first"
        #
        # i.e. the PC simulator only exists on the load+build path.  So the
        # harness rebuilds the graph here.  The config below is therefore
        # load-bearing: it must be IDENTICAL to what `convert_to_rknn.py` used, or
        # this measures a different model than the one in `weights/`.  The fp16
        # graph build is deterministic - the `.rknn` file on disk and a fresh
        # build of the same ONNX give the same simulator numbers - which is why
        # this is a faithful measurement and not an approximation.
        #
        # The `.rknn` paths are still recorded, because they are what a reader
        # would go and look at, and a mismatch between the path on disk and the
        # ONNX actually built is the kind of thing that silently scores the wrong
        # model.  `--s2-onnx` defaults to the MatMul graph, so the default here
        # must follow it rather than name a file that may not be the shipped one.
        self.s1_path, self.s2_path = str(s1_path), str(s2_path)
        self.s1_onnx, self.s2_onnx = str(s1_onnx), str(s2_onnx)
        from rknn.api import RKNN
        self._RKNN = RKNN
        self._live = [None, None]

    def _runtime(self, onnx, norm, slot):
        if self._live[slot] is not None:
            return self._live[slot]
        rk = self._RKNN(verbose=False)
        cfg = {"target_platform": "rk3588", "float_dtype": "float16"}
        # Per-input-channel mean/std, exactly as `convert_to_rknn.py` sets them:
        # stage 1 normalises on the NPU so the C++ side can hand over uint8; the
        # matcher's inputs are keypoints and already-normalised descriptors and
        # must be left alone (passing them there fails outright, because RKNN
        # reads input 0 as a 512-"channel" tensor).
        if norm:
            cfg["mean_values"] = [[0, 0, 0]]
            cfg["std_values"] = [[255, 255, 255]]
        rk.config(**cfg)
        if rk.load_onnx(model=onnx) != 0:
            raise RuntimeError(f"load_onnx failed for {onnx}")
        if rk.build(do_quantization=False) != 0:
            raise RuntimeError(f"build failed for {onnx}")
        if rk.init_runtime(target=None) != 0:
            raise RuntimeError(f"init_runtime failed for {onnx}")
        self._live[slot] = rk
        return rk

    def extract(self, rgb_u8):
        """ONE HWC uint8 colour image -> (k, d) as numpy (1,K,2)/(1,K,128).

        The simulator does not guarantee output order; it is deterministic for a
        given graph, but identifying by shape removes the assumption entirely.
        """
        rk1 = self._runtime(self.s1_onnx, norm=True, slot=0)
        outs = rk1.inference(inputs=[rgb_u8[None].astype(np.uint8)],
                             data_format="nhwc")
        return self._pick_alike(outs)

    def match(self, f0, f1):
        # Stage 2 is fed the SIMULATOR's own stage-1 outputs, not torch's.  That is
        # the whole point of chaining rather than scoring the stages in isolation:
        # it is the error a deployment actually accumulates, and a per-stage
        # harness that hands the matcher clean inputs cannot see it.
        rk2 = self._runtime(self.s2_onnx, norm=False, slot=1)
        outs2 = rk2.inference(inputs=[f0[0].astype(np.float32),
                                      f1[0].astype(np.float32),
                                      f0[1].astype(np.float32),
                                      f1[1].astype(np.float32)])
        return self._pick_matcher(outs2)

    def __call__(self, rgb0_u8, rgb1_u8):
        # ONE IMAGE AT A TIME: two extractions, then one match.
        (k0, d0) = self.extract(rgb0_u8)
        (k1, d1) = self.extract(rgb1_u8)
        m0, ms0 = self.match((k0, d0), (k1, d1))
        return (torch.from_numpy(k0[0]), torch.from_numpy(k1[0]),
                torch.from_numpy(np.asarray(m0[0], dtype=np.int64)),
                torch.from_numpy(np.asarray(ms0[0], dtype=np.float32)))

    @staticmethod
    def _pick_alike(outs):
        o = [np.asarray(x) for x in outs]
        d = next(x for x in o if x.ndim == 3 and x.shape[-1] == 128)
        k = next(x for x in o if x.ndim == 3 and x.shape[-1] == 2)
        return k.astype(np.float32), d.astype(np.float32)

    @staticmethod
    def _pick_matcher(outs):
        o = [np.asarray(x) for x in outs]
        m = next((x for x in o if "int" in str(x.dtype)), None)
        if m is None:
            # the simulator may hand back float32 for an int64 graph output; the
            # match array is the ONLY integer-valued output, so `isclose` on the
            # rounded values is the tiebreak rather than a guess about dtypes.
            m = next(x for x in o if np.allclose(x, np.round(x)))
        ms = next(x for x in o if x is not m)
        return m.astype(np.int64), ms.astype(np.float32)

    def release(self):
        # Runtimes are held in `self._live` and released here rather than after
        # every call.  Building an fp16 graph takes minutes of host CPU - far more
        # than an inference - so rebuilding per pair would make a 540-pair judge
        # take hours for no reason.  Keeping one runtime per stage and closing it
        # at the end keeps host memory bounded, which is what the per-call rebuild
        # was protecting against.
        for i, rk in enumerate(self._live):
            try:
                rk.release()
            except Exception:                          # noqa: BLE001
                pass
            self._live[i] = None

    _live = None


# --------------------------------------------------------------------------- #
# ground truth / judges
# --------------------------------------------------------------------------- #
def load_hpatches(n, size):
    """Viewpoint sequences only: illumination pairs are too easy to be a judge."""
    out = []
    for seq in sorted(paths.hpatches().iterdir()):
        if not seq.is_dir() or seq.name[0] == "i":
            continue
        ref = seq / "1.ppm"
        for q in range(2, 7):
            qf, hf = seq / f"{q}.ppm", seq / f"H_1_{q}"
            if ref.is_file() and qf.is_file() and hf.is_file():
                a = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
                b = cv2.imread(str(qf), cv2.IMREAD_GRAYSCALE)
                if a is None or b is None:
                    continue
                sa = (a.shape[1], a.shape[0])
                a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
                b = cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR)
                # The homography must be transported with the resize, exactly as
                # `eval_hpatches_sources.py` does - scoring a resized pair against
                # the ORIGINAL H is a silent, plausible-looking error.
                H = np.loadtxt(hf)
                Hn = norm_to_resized(H, sa, (size, size))
                out.append((np.stack([a] * 3, -1), np.stack([b] * 3, -1), Hn))
                break
        if len(out) >= n:
            break
    return out


def norm_to_resized(H, src_wh, dst_wh):
    """H in original pixels -> H in resized pixels (`S @ H @ S^-1`)."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    S_i = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    Hn = S @ H @ S_i
    return Hn / Hn[2, 2]


def hpatches_metrics(k0, k1, m0, H, tol=(3, 5)):
    """Homography error + mAA from the matches, scored at fixed tolerances.

    The estimator is OpenCV RANSAC on the raw matches - the same estimator and
    the same thresholds as `eval_hpatches_sources.py`, so the numbers can be read
    next to the training-side table instead of only next to each other.

    `n_inl` IS ORDER-DEPENDENT; `n_inl_gt` IS NOT
    ---------------------------------------------
    See the `n_inl_gt` block below.  The RANSAC-based numbers (`n_inl`, `mAA`,
    `H_error`) are kept because they are the protocol the training-side tables
    use, but they carry an RNG-dependent offset that `cv2.setRNGSeed` does not
    remove (measured: the same graph scored 258.46 alone and 245.68 after another
    pipeline had run).  `n_inl_gt` is the number to compare arms on.
    """
    import cv2
    out = {"n_match": int((m0 >= 0).sum())}
    valid = (m0 >= 0).nonzero(as_tuple=True)[0]
    if valid.numel() < 4:
        out["H_error"] = float("nan")
        out["mAA"] = 0.0
        out["mprec@3px"] = float("nan")
        out["n_inl"] = 0
        out["n_inl_gt"] = 0
        for t in tol:
            out[f"@{t}px"] = 0.0
        return out
    p0 = k0[valid].numpy().astype(np.float64)
    p1 = k1[m0[valid].long()].numpy().astype(np.float64)

    # ------------------------------------------------------------------ #
    # DETERMINISTIC inlier count, measured against the GROUND-TRUTH homography.
    #
    # Why this exists next to the RANSAC one: `cv2.findHomography(..., RANSAC)`
    # draws from OpenCV's global RNG, so its inlier count for a given pair depends
    # on how many such calls happened BEFORE it.  Measured on this 59-pair set,
    # the SAME ONNX graph scored:
    #
    #     --pipeline onnx        alone      n_inl = 258.4576
    #     --pipeline torch onnx  onnx arm   n_inl = 245.6780
    #
    # and `cv2.setRNGSeed` does not fix it.  So the two arms in a two-pipeline run
    # are not measured on the same footing, and a 13-inlier gap was being read as
    # a model difference.
    #
    # Counting agreement with the KNOWN homography removes the estimator from the
    # measurement entirely: it is a pure function of the matches and the ground
    # truth, so it is identical no matter what ran before.  It is also the more
    # meaningful quantity for comparing two matchers - it says how many of the
    # correspondences are actually correct, rather than how many a particular
    # RANSAC run happened to agree with.
    # ------------------------------------------------------------------ #
    p0h_gt = np.concatenate([p0, np.ones((len(p0), 1))], 1) @ H.T
    p0h_gt = p0h_gt[:, :2] / p0h_gt[:, 2:3]
    err_gt = np.linalg.norm(p0h_gt - p1, axis=1)
    out["n_inl_gt"] = int((err_gt <= 3).sum())

    H_est, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    out["n_inl"] = int(inl.sum()) if inl is not None else 0
    if H_est is None:
        out["H_error"] = float("nan")
        out["mAA"] = 0.0
        out["mprec@3px"] = float("nan")
        for t in tol:
            out[f"@{t}px"] = 0.0
        return out
    # ground-truth reprojection error of the first image's corners
    h, w = 512, 512
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float64)
    def proj(Hm, pts):
        p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Hm.T
        return p[:, :2] / p[:, 2:3]
    err = np.linalg.norm(proj(H_est, corners) - proj(H, corners), axis=1).mean()
    out["H_error"] = float(err)
    out["mAA"] = float(np.clip(1.0 - err / 10.0, 0.0, 1.0))
    for t in tol:
        out[f"@{t}px"] = float(err <= t)
    # Precision, and `n_inl_gt` is its numerator.  Reusing `err_gt` rather than
    # recomputing the same projection guarantees the count and the fraction can
    # never disagree - they are the same measurement presented two ways, and a
    # reader who multiplies `mprec@3px` by `n_match` must get `n_inl_gt` back.
    out["mprec@3px"] = float((err_gt <= 3).mean())
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
# The keypoint count of the deployed graphs.  Used only to reject out-of-range
# match indices from a converted engine; any mismatch with the real count is caught
# by the shapes the pipeline already asserts.
kKeypointsGlobal = 512

_WARNED = set()


def warn_once(msg):
    """Print a diagnostic once per message.

    Used where a fallback is legitimate but should not be invisible: a per-pair
    warning would either flood the log or get ignored, and either way the fact that
    a fallback happened would be lost.
    """
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[w] {msg}")


def _want_trt(args):
    return any(w.startswith("trt") for w in args.pipeline)


def trt_engine_paths(args):
    """The two engines this arm will run, resolved and checked for existence.

    Defaults follow the shipped optimization: the gathered stage 1 and the 7-layer
    matcher, at the precision `--trt-precision` asks for.  Both files are named
    explicitly in the error if they are missing, because the usual cause is a
    checkout without engines built on this machine - and an engine is not portable.
    """
    repo = Path(__file__).resolve().parents[1]
    s1 = args.s1_engine or str(
        repo / f"weights/optimized/alike_stage_gath_{args.trt_precision}.engine")
    s2 = args.s2_engine or str(
        repo / f"weights/optimized/lightglue_stage_d7_{args.trt_precision}.engine")
    missing = [p for p in (s1, s2) if not Path(p).is_file()]
    if missing:
        raise SystemExit(
            "missing TensorRT engine(s):\n  " + "\n  ".join(missing) +
            "\nBuild them with scripts/convert_to_trt.py (see the README); engines "
            "are device-bound, so one built on another machine will not load here.")
    return s1, s2


def build_pipelines(which, args):
    out = {}
    for w in which:
        if w == "torch":
            out[w] = TorchPipeline(args.checkpoint, args.size, args.keypoints)
        elif w == "onnx":
            out[w] = OnnxPipeline(args.s1_onnx, args.s2_onnx)
        elif w.startswith("trt"):
            # The precision is part of the arm's name only as a label; the engine
            # directory is what decides, and `--trt-precision` picks it, so the two
            # cannot disagree silently.
            s1, s2 = trt_engine_paths(args)
            out[w] = TrtPipeline(s1, s2, name=w, precision=args.trt_precision)
        elif w.startswith("rknn"):
            # The simulator only exists on the load+build path, so the graph is
            # rebuilt from the SAME ONNX the `.rknn` on disk was built from.  Both
            # paths are passed through so the two cannot drift apart.
            out[w] = RknnPipeline(args.s1_rknn, args.s2_rknn, name=w,
                                  s1_onnx=args.s1_onnx, s2_onnx=args.s2_onnx)
        else:
            raise ValueError(f"unknown pipeline {w!r}")
    return out


def run_hpatches(pipe, pairs, verbose_every=60):
    rows = []
    t0 = time.time()
    for i, (a, b, H) in enumerate(pairs):
        try:
            k0, k1, m0, _ = pipe(a, b)
        except Exception as exc:                      # noqa: BLE001
            print(f"    !! pair {i} failed: {type(exc).__name__}: {exc}")
            continue
        r = hpatches_metrics(k0, k1, m0, H)
        rows.append(r)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    [{pipe.name}] {i+1}/{len(pairs)} "
                  f"({time.time() - t0:.0f}s)")
    return summarize(rows)


def summarize(rows):
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "per_pair"]
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None and not
             (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(v)) if v else float("nan")
    out["n_pairs"] = len(rows)
    return out


def compare(a, b, tol):
    """`b - a` where `a` is the reference (torch)."""
    if not a or not b:
        return {}
    return {k: b[k] - a[k] for k in a
            if isinstance(a.get(k), float) and isinstance(b.get(k), float)
            and not math.isnan(a[k]) and not math.isnan(b[k])
            and k != "n_pairs"}


def fmt(v, nd=4):
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return "  -  " if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    # Required only when the torch column is actually going to be measured: the
    # converted arms read their inputs from ONNX, and with `--ref-from` the
    # reference is loaded from a file.  Making it mandatory unconditionally would
    # force the simulator environment to carry the training dependencies it
    # deliberately does not have.
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pipeline", nargs="*",
                    default=["torch", "onnx", "rknn-fp16", "trt-fp16"],
                    choices=["torch", "onnx", "rknn-fp16", "trt-fp16", "trt-fp32"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--hpatches", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    # The stage-2 graph is selectable because there are two of them: upstream's
    # `Einsum` form and the `MatMul` rewrite that removes 63 off-NPU nodes.  A
    # hard-coded path would silently score whichever one happened to be in
    # `weights/`, which is how a report ends up describing a model that is not the
    # one being shipped.
    # These default to the SHIPPED pair under `weights/optimized/`.  Every
    # combination is reachable by overriding all four, and `weights/original/`
    # holds the pre-optimisation baseline (dense head + upstream `Einsum` attention).
    ap.add_argument("--s1-onnx", default=str(REPO / "weights/optimized/alike_stage_gath.onnx"))
    ap.add_argument("--s2-onnx", default=str(REPO / "weights/optimized/lightglue_stage_d7.onnx"))
    ap.add_argument("--s1-rknn", default=str(REPO / "weights/optimized/alike_stage_gath_fp.rknn"))
    ap.add_argument("--s2-rknn", default=str(REPO / "weights/optimized/lightglue_stage_d7_fp.rknn"))
    ap.add_argument("--trt-precision", default="fp16", choices=["fp16", "fp32"],
                    help="which engine the trt-* arms load; keep it consistent with "
                         "the arm name, since the name is only a label")
    ap.add_argument("--s1-engine", default=None,
                    help="override the stage-1 engine (default follows "
                         "--trt-precision under weights/optimized/)")
    ap.add_argument("--s2-engine", default=None, help="override the stage-2 engine")
    ap.add_argument("--ref-pipeline", default="torch",
                    help="which column the deltas are measured against")
    ap.add_argument("--ref-from", default=None,
                    help="reuse a previously measured reference row from this JSON "
                         "instead of recomputing it.  Needed because torch and the "
                         "RKNN toolkit live in different environments: the simulator "
                         "env has no `omegaconf` and should not need it.  The pair "
                         "counts and size are checked so a cached row cannot be "
                         "silently compared against a different sample.")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None, help="write the raw numbers here")
    args = ap.parse_args()

    if args.size % 32:
        ap.error("--size must be a multiple of 32 (the export only supports the "
                 "no-padding downsample path)")
    if "torch" in args.pipeline and not args.checkpoint:
        ap.error("--checkpoint is required to measure the torch column "
                 "(pass --ref-from <json> to reuse a previous measurement instead)")

    hp = load_hpatches(args.hpatches, args.size)
    print(f"[e2e] {len(hp)} HPatches pairs, "
          f"{args.size}px / {args.keypoints} kpts")
    print(f"[e2e] pipelines: {', '.join(args.pipeline)}")
    print("[e2e] NOTE the RKNN arm runs the PC simulator - it validates NUMERICS "
          "only.\n      No timing below is an on-device timing claim.\n")

    results = {}
    # The torch reference can be REUSED from an earlier run rather than
    # recomputed, and there are two reasons that matters beyond saving time:
    #
    # * It is deterministic. Same checkpoint, same pairs, same size - the row is
    #   reproducible, so recomputing it adds nothing and a small difference in a
    #   re-measurement would then have to be explained.
    # * The two stages live in different environments on purpose. Torch needs
    #   glue-factory (and therefore `omegaconf`); the RKNN simulator needs the
    #   toolkit. Nothing here should require one env to carry the other's
    #   dependencies, and `--ref-from` is what makes that possible.
    #
    # What this does NOT do is accept a reference measured on DIFFERENT pairs.
    # The pair count and size are checked below, because a cached row over 200
    # pairs silently compared against a fresh run over 150 is exactly the kind of
    # mismatch that looks like a model regression.
    if args.ref_from:
        cached = json.loads(Path(args.ref_from).read_text())
        cfg = cached.get("config", {})
        for key, want in (("hpatches", len(hp)), ("size", args.size)):
            got = cfg.get(key)
            if got is not None and len(hp) and got != want:
                ap.error(f"{args.ref_from} was measured with {key}={got}, this run "
                         f"has {key}={want}; the rows are not comparable")
        ref_name = args.ref_pipeline
        if ref_name not in cached["results"]:
            ap.error(f"{args.ref_from} has no '{ref_name}' row "
                     f"(has: {sorted(cached['results'])})")
        results[ref_name] = cached["results"][ref_name]
        print(f"[e2e] torch reference REUSED from {args.ref_from} "
              f"(pipeline '{ref_name}', {cfg.get('hpatches')} pairs)")

    for name in args.pipeline:
        pipe = build_pipelines([name], args)[name]
        print(f"[e2e] === {name} ===")
        entry = {}
        if hp:
            print(f"  HPatches ({len(hp)} pairs)")
            entry["hpatches"] = run_hpatches(pipe, hp)
        pipe.release()
        results[name] = entry
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # The baseline column is the reference pipeline, whether it was just measured
    # or loaded - so the deltas are always against torch, not against whatever
    # happened to be measured first in this invocation.
    ref = results.get(args.ref_pipeline, results.get(args.pipeline[0], {}))
    cols = list(results) if args.ref_pipeline not in args.pipeline else args.pipeline

    # ---- console table -----------------------------------------------------
    # `n_inl_gt` is the deterministic inlier count and the one to compare arms
    # on; `n_inl` (RANSAC) is kept for continuity with the training-side tables
    # but is order-dependent, so it is printed after it with `H_error`.
    hp_keys = ["mAA", "@3px", "@5px", "mprec@3px", "n_inl_gt", "n_inl",
               "H_error", "n_match"]
    print("\n" + "=" * 88)
    print(f"END-TO-END ACCURACY  ({args.size}px, {args.keypoints} kpts)")
    print("=" * 88)
    if ref.get("hpatches"):
        print("\n-- hpatches --")
        print(f"{'metric':14s}" + "".join(f"{p:>14s}" for p in cols)
              + f"   delta(vs {args.ref_pipeline})")
        for k in hp_keys:
            if k not in ref["hpatches"]:
                continue
            line = f"{k:14s}" + "".join(
                f"{fmt(results[p]['hpatches'].get(k), 4):>14s}" for p in cols)
            if len(cols) > 1:
                d = compare(ref["hpatches"], results[cols[-1]]["hpatches"], 0).get(k)
                line += f"   {fmt(d, 4)}"
            print(line)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "config": {"size": args.size, "keypoints": args.keypoints,
                       "hpatches": len(hp), "seed": args.seed,
                       # Repo-relative, never absolute: these reports are committed,
                       # and an absolute path both leaks the author's filesystem
                       # layout and cannot be resolved by anyone reading later.
                       "checkpoint": portable(args.checkpoint, "checkpoint"),
                       "s1_onnx": portable(args.s1_onnx, "onnx"),
                       "s2_onnx": portable(args.s2_onnx, "onnx"),
                       # An engine is identified by the precision it was built at
                       # as much as by its path, and the trt-* arm's name is only a
                       # label - so the precision the arm actually loaded is what
                       # gets recorded.
                       "trt_precision": args.trt_precision if _want_trt(args) else None,
                       "reference_from": portable(args.ref_from, "onnx")},
            "results": results}, indent=2))
        print(f"\n[e2e] raw numbers -> {args.json}")

    if args.report:
        write_report(Path(args.report), args, results, hp_keys, cols)
        print(f"[e2e] report -> {args.report}")
    return 0


def write_report(path, args, results, hp_keys, cols):
    p = cols
    n_pairs = (results[p[0]]["hpatches"].get("n_pairs", 0)
               if "hpatches" in results[p[0]] else 0)
    lines = [
        "# End-to-end accuracy, converted pipeline vs torch",
        "",
        f"`{args.size}px`, `{args.keypoints}` keypoints, "
        f"`alike_native_gl_s1`, seed {args.seed}.",
        f"Open-source HPatches viewpoint sequences: {n_pairs} pairs.",
        "",
        "The RKNN arm runs the **PC simulator**. It validates numerics; it does not "
        "measure latency.",
        "",
    ]
    if results[p[0]].get("hpatches"):
        lines += ["## hpatches", "",
                  "| metric | " + " | ".join(p) + " | delta |",
                  "|---|" + "---|" * (len(p) + 1)]
        for k in hp_keys:
            if k not in results[p[0]]["hpatches"]:
                continue
            row = [fmt(results[q]["hpatches"].get(k), 4) for q in p]
            d = compare(results[p[0]]["hpatches"], results[p[-1]]["hpatches"], 0).get(k)
            lines.append(f"| `{k}` | " + " | ".join(row) + f" | {fmt(d, 4)} |")
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "* `mAA` / `@3px` / `@5px` are HPatches homography accuracy; "
        "`mprec@3px` is the fraction of matches consistent with the ground-truth "
        "homography and `n_inl` the RANSAC inlier count. Precision alone is not "
        "enough: a matcher that emits fewer matches can win on precision and "
        "deliver fewer usable correspondences.",
        "* Compare arms on `n_inl_gt`, not `n_inl`: the RANSAC count depends on "
        "OpenCV's global RNG and therefore on call order.",
        "* A delta is only meaningful next to the pair count - see the line above "
        "the table.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
