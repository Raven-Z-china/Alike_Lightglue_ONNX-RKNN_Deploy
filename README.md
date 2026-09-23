# ALIKE + LightGlue — RKNN/TensorRT 

Keypoint detection and matching on RKNN and TensorRT.

The full pipeline — ALIKE backbone, NMS, top-k, soft-argmax, descriptor sampling, and LightGlue matcher — runs **in-graph**. No operators fall back to CPU in the released configuration.

## Repository layout

| path | contents |
|---|---|
| `weights/` | Deployable artifacts: checkpoints, ONNX, RKNN (see below) |
| `model/` | Torch modules for ONNX export (source of truth) |
| `model/matchers/` | Vendored LightGlue implementation (no training checkout needed) |
| `scripts/` | Model conversion (RKNN + TensorRT), validation, benchmark, ablation and probing tools |
| `pruning/` | Checkpoint truncation + accuracy recovery fine-tuning |
| `eval/` | End-to-end evaluation harness (HPatches) |
| `refs/` | Scratch (git-ignored): reference tensor dumps + calibration lists |
| `paths.py` | Centralized management of all external paths |
| `cpp/` | On-board C++ examples (RKNN C API): one pair, or one image against a folder |
| `image/` | Demo images: two stills and a 16-frame sequence |


## Environments

Two independent Conda environments. They do not need to be active simultaneously; the required environment is marked for each workflow step.

| env | python | purpose | spec |
|---|---|---|---|
| `alike` | 3.10 | ONNX export, accuracy evaluation, pruning, ablation, fine-tuning | [`requirements-alike.txt`](requirements-alike.txt) |
| `rknn` | 3.8 | ONNX to RKNN conversion, simulator validation | [`requirements-rknn.txt`](requirements-rknn.txt) |

```bash
conda activate alike   # Export & validate
conda activate rknn    # Convert & simulator test
```

### Setup

Requires x86-64 Linux with CUDA driver. PyTorch builds support Turing ~ Ampere GPUs (`sm_75` ~ `sm_86`). GPU is optional.

```bash
# --- alike env: graph construction & evaluation ---
conda create -n alike python=3.10 -y && conda activate alike
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-alike.txt

# --- rknn env: model conversion & simulator only ---
conda create -n rknn python=3.8 -y && conda activate rknn
# Install torch first: rknn-toolkit2 enforces torch<=2.4.0
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install rknn-toolkit2==2.3.2 omegaconf==2.3.1
```

Version pin notes:

- `rknn-toolkit2==2.3.2` only provides cp38 wheel, so Python 3.8 is mandatory for `rknn` env.
- `omegaconf` is not a toolkit dependency; it is only used for checkpoint inspection scripts.
- The `torch<=2.4.0` pin cannot be met in the `alike` env: torch 2.4 cannot export stage 2 at any opset, and its stage-1 graph is 22 % larger — that is what forces two environments rather than one.
- Other pinned versions are validated combinations; minor version drift usually has negligible impact.

No extra training code checkout required. Run sanity check (no arguments / dataset needed):

```bash
conda activate alike && python scripts/check_selfcontained.py   # Expected output: 0 blocked
```

## ONNX → RKNN Conversion Workflow

```bash
conda activate alike

# 1) Stage 1 (ALIKE). --batch 1: one image per call, so each reference dump holds
#    ONE image. Run it twice (different --seed) to get the two views stage 2 needs.
python scripts/export_onnx.py --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar \
    --size 512 --keypoints 512 --batch 1 --seed 0 \
    --out weights/optimized/alike_stage_gath.onnx --ref refs/s1_a.npz
python scripts/export_onnx.py --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar \
    --size 512 --keypoints 512 --batch 1 --seed 1 \
    --out weights/optimized/alike_stage_gath.onnx --ref refs/s1_b.npz
# Baseline: add --no-gathered-head, output to weights/original/

# 2) Stage 2 (LightGlue). --stage1-ref uses real stage1 outputs instead of random
#    tensors for a valid reference dump; it takes the TWO single-image dumps,
#    view0 first.
# Baseline: --attention einsum + 9-layer checkpoint
# <d7 checkpoint>: the 9->7 fine-tune is an output of the training checkout. The
# graph in weights/optimized/ is shipped; the checkpoint it came from is not.
python scripts/export_onnx.py --stage lightglue --attention matmul \
    --checkpoint <d7 checkpoint> --keypoints 512 \
    --stage1-ref refs/s1_a.npz,refs/s1_b.npz --ref refs/s2_d7.npz \
    --out weights/optimized/lightglue_stage_d7.onnx

conda activate rknn

# 3) Convert to FP16 RKNN. Normalization differs by stage:
# Stage1: NPU performs normalization (C++ passes raw uint8). LightGlue uses no normalization.
python scripts/convert_to_rknn.py weights/optimized/alike_stage_gath.onnx \
    --stage alike --norm 0,0,0:255,255,255 --ref refs/s1_a.npz \
    --report reports/rknn_alike_gath.json
python scripts/convert_to_rknn.py weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --norm none --ref refs/s2_d7.npz \
    --report reports/rknn_lg_d7.json

# 4) Off-NPU audit (ground-truth operator count).
# Audit inside convert_to_rknn.py is for convenience only and incomplete; it misses C++ stream operators.
python scripts/build_log_probe.py --onnx weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --dump /tmp/d7.log

# 5) End-to-end accuracy test (two separate environments).
# Reference checkpoint must match matcher weights.
conda activate alike
python eval/run_accuracy.py --checkpoint <d7 checkpoint> \
    --pipeline torch onnx --json reports/e2e.json
conda activate rknn
python eval/run_accuracy.py --pipeline rknn-fp16 \
    --ref-from reports/e2e.json --json reports/rknn.json
```

## TensorRT (optional, x86 GPU)

The same ONNX graphs also convert to TensorRT, for host-side inference on a discrete
GPU. This produces a second set of engines under `weights/` — one per stage, plus an
fp32 pair kept as the exactness reference.

```bash
conda activate alike

# 6) Convert to TensorRT. --ref is a torch dump from step 1/2, so the engine is
#    checked against the same reference the RKNN path is checked against.
python scripts/convert_to_trt.py weights/optimized/alike_stage_gath.onnx \
    --stage alike --precision fp16 --ref refs/s1_a.npz
python scripts/convert_to_trt.py weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --precision fp16 --ref refs/s2_d7.npz
# Exactness reference (fp32), and the baselines under weights/original/ likewise:
#   ... --precision fp32 --ref refs/s1_a.npz
```

Three things to know about the engines:

- **An engine is not portable.** It is compiled for one GPU architecture, one
  TensorRT version and one CUDA version, and it will refuse to load elsewhere. Each
  report records the provenance (`device`, `tensorrt`, `cuda`) next to the file, and
  the tool prints it after every build. Treat `.engine` as a build output, not as a
  distributable model — the `.onnx` stays the artifact that travels.
- **`--precision fp16` is a request, not a description.** TensorRT keeps fp32 where a
  layer cannot be reduced safely, so the report carries the engine's actual
  per-tensor dtype histogram instead of restating the flag. No plugins are used in
  any of the shipped engines, including the baseline's 27 `Einsum` nodes, which the
  RKNN path leaves off-NPU.
- **Builds are not bit-reproducible.** Tactic selection varies between builds (same
  graph, 3.12 vs 3.31 MB; 497 vs 505 identical match columns), so a rebuilt engine
  should be re-verified with `--ref` rather than assumed equal to the shipped one.

For end-to-end numbers rather than per-graph ones, the accuracy harness takes the
engines as a fourth arm (needs only the `alike` env and the GPU):

```bash
conda activate alike
python eval/run_accuracy.py --pipeline trt-fp16 --ref-from <reports>/e2e.json \
    --json <reports>/trt_e2e_fp16.json
```

`--trt-precision` picks `_fp16`/`_fp32` engines, and `--s1-engine`/`--s2-engine` override
the paths entirely. To reproduce the sequence demo, matching `image/image0.png` against
every other image in a folder and drawing the match lines:

```bash
python ../alike_lightglue_outputs/run_sequence_trt.py \
    --images image --out sequence_out --ext .png
```

## Accuracy (HPatches)

Tested on HPatches viewpoint sequences, 512×512 input, 512 keypoints, seed=0.

- `L2`: PyTorch → ONNX FP32
- `L3`: ONNX → RKNN FP16

Raw reports in `reports/` (git-ignored). Evaluated on 59 viewpoint pairs with shipped model (`weights/optimized/`):

| metric | torch | ONNX (L2) | RKNN fp16 (L3) | TRT fp16 (GPU) |
|---|---|---|---|---|
| `n_inl_gt` (deterministic inliers) | 225.5932 | **225.5932** | 225.2203 (−0.17 %) | 225.3220 (−0.12 %) |
| `mAA` | 0.7476 | 0.7485 | 0.7343 | 0.7405 |
| `@3px` | 0.8136 | 0.8136 | 0.7966 | 0.7966 |
| `@5px` | 0.8136 | 0.8136 | 0.8136 | 0.8136 |
| `mprec@3px` | 0.6535 | 0.6536 | **0.7335** | 0.7345 |
| `n_match` | 329.54 | 329.51 | 295.24 (−10.4 %) | 294.92 (−10.5 %) |

### Interpretation

1. Export from PyTorch to ONNX is exact: `n_inl_gt` matches to four decimal places; `mAA` differs only by 0.0009.
2. Both fp16 backends retain ~99.8 % of valid correspondences. They output ~10 % fewer matches than fp32, discarding low-confidence pairs (confidence <1e-4). This improves precision while keeping the inlier count nearly unchanged — the same effect, and of the same size, on NPU and GPU.
3. Per-point alignment (8 image pairs, one image each): ONNX keypoints are bit-exact (0.000 px). RKNN median error 0.043 px — but that median hides the shape of the error: it is ~0.15 px on most keypoints with a handful of near-tie rank flips on top (per-pair maxima 0.15–11.4 px; **3 keypoints of 4096** beyond 1 px, on 3 of the 8 pairs). Median descriptor cosine similarity: 0.99971. With identical inputs, matcher agrees on 502 out of 512 match indices. Same-protocol comparison of both fp16 backends: `reports/fp16_tail.json`.
4. Use `n_inl_gt` for comparison, not raw RANSAC `n_inl`. RANSAC inlier count is order-dependent. Identity: `mprec@3px × n_match = n_inl_gt`.

## ONNX Inference Speed

| model | unit | CPU ms | GPU ms | TRT fp16 ms | CPU rel. |
|---|---|---|---|---|---|
| stage 1, original (dense head) | one image | 72 | 4.0 | 1.2 | 1.00× |
| stage 1, optimized (gathered head) | one image | **31** | 3.3 | **1.1** | **2.3×** |
| stage 2, original (`Einsum`, 9 layers) | one pair | 222 | 6.3 | 1.2 | 1.00× |
| stage 2, `MatMul` 9 layers (graph rewrite only) | one pair | 105 | 5.7 | — | 2.1× |
| stage 2, optimized (`MatMul`, 7 layers) | one pair | **78** | 5.6 | **0.9** | **2.8×** |
| **full pipeline** | one pair | 365 → **140** | 14.3 → **12.2** | **3.1** | 2.6× |
| each further image, reusing a cached extraction | one image | — → **31** | — → **3.3** | — → **1.1** | |




## License

Apache License 2.0 - see [`LICENSE`](LICENSE). The two models keep their own terms:
this repository's license covers the conversion, deployment and tooling code here,
while ALIKE and LightGlue remain under their upstream licenses (see *Acknowledgements*).


## C++ On-board Inference Sample

Two samples build from the same CMake project; both read uint8 RGB images, with
normalization on the NPU (mean=0, std=255; no float preprocessing on CPU).

```bash
cd cpp && mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=<aarch64-toolchain> \
      -DRKNN_API_ROOT=<path-to-librknn_api> \
      -DOPENCV_ROOT=<opencv-for-board> ..
make

# 1) One image pair: stage 1 twice, stage 2 once.
./infer ../../weights/optimized/alike_stage_gath_fp.rknn \
        ../../weights/optimized/lightglue_stage_d7_fp.rknn \
        left.jpg right.jpg [--dump DIR]

# 2) One reference image against a folder of others: the reference is extracted
#    ONCE and reused, so each further frame costs one extraction plus one match.
mkdir -p out
./infer_sequence ../../weights/optimized/alike_stage_gath_fp.rknn \
        ../../weights/optimized/lightglue_stage_d7_fp.rknn \
        ../../image/freiburg_sequence out --ext .png [--vis] [--dump DIR]
```


## Acknowledgements

This deployment is assembled from two models and two toolchains - what this
repository takes from each:

| project | taken from it |
|---|---|
| [ALIKE](https://github.com/Shiaoming/ALIKE) (Zhao et al., 2022) | the keypoint detector/descriptor: the network in `model/` follows its architecture, the shipped checkpoints descend from training on its code, and its DKD sampling was re-implemented for graph export |
| [LightGlue](https://github.com/cvg/LightGlue) (Lindenberger et al., ICCV 2023) | the matcher: `model/matchers/` is a vendored copy, trimmed to what the deployment executes |
| [SuperPoint-LightGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT) | the two-stage shape of `cpp/` (extract per image, then match) and the demo images under `image/` |
| [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) | the converter, the accuracy simulator, and the on-board runtime behind the RKNN path |

If you build on this repository, please cite the two papers behind the models -
the conversion work here changes where they run, not what they are.