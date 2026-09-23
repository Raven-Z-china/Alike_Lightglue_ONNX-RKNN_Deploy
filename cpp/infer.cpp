// RKNN C API sample: ALIKE + LightGlue, two stages, image pair -> matches.
//
// Design notes that matter on the board
// -------------------------------------
// * NO float preprocessing.  The stage-1 RKNN model is configured with
//   mean=0 / std=255, so the NPU divides by 255 itself and the sample feeds raw
//   uint8 RGB.  Doing `x/255` in C++ would be both slower and wrong (it would be
//   applied a second time by the NPU).
// * Image size is FIXED at 512x512 with 512 keypoints - the models are exported
//   with static shapes and RKNN cannot resize them.  Resizing happens once, on
//   the host, before the first inference.
// * ONE IMAGE PER STAGE-1 CALL, deliberately.  `extract()` is the unit of work, so
//   a single extraction can be cached and reused (one image against N others), and
//   the peak activation footprint is halved - the ONNX graph is exported at
//   batch 1 and RKNN cannot resize it.  The cost is real and is paid per pair: two
//   `rknn_run` calls, each with its own input copy and dispatch, instead of one
//   call for both views.  See `bench_stages.py`, which reports both the per-image
//   and the per-pair figure rather than hiding the difference.
// * Outputs are read with `want_float = 1`.  RKNN may keep intermediate buffers
//   in fp16; letting it convert on read is one place a silent precision bug would
//   hide, so every output is requested as float32.
//
// Build: see CMakeLists.txt.  Requires librknn_api (aarch64) and OpenCV.
//
// For the many-to-one direction - one reference image matched against a folder of
// others, with the reference's stage-1 output reused - see `infer_sequence.cpp`,
// which also reports timings.
//
// Usage:
//   ./infer alike_stage.rknn lightglue_stage.rknn left.jpg right.jpg [--dump DIR]
//
// With --dump, every stage output is written as raw float32 (shape in a sidecar
// .txt) so the board result can be diffed against the Python parity harness.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

#define CHECK(expr, msg)                                                    \
  do {                                                                      \
    int _r = (expr);                                                        \
    if (_r != 0) {                                                          \
      fprintf(stderr, "[E] %s failed (ret=%d)\n", (msg), _r);               \
      return 1;                                                             \
    }                                                                       \
  } while (0)

namespace {

constexpr int kSize = 512;       // network input side (must stay in sync with the export)
constexpr int kKeypoints = 512;  // fixed keypoint count (static shapes)

struct Model {
  rknn_context ctx = 0;
  std::vector<rknn_tensor_attr> in_attrs, out_attrs;

  ~Model() {
    if (ctx) rknn_destroy(ctx);
  }

  bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      fprintf(stderr, "[E] cannot open %s\n", path.c_str());
      return false;
    }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != sz) {
      fclose(f);
      fprintf(stderr, "[E] short read on %s\n", path.c_str());
      return false;
    }
    fclose(f);
    if (rknn_init(&ctx, buf.data(), sz, 0, nullptr) != 0) {
      fprintf(stderr, "[E] rknn_init failed for %s\n", path.c_str());
      return false;
    }
    uint32_t n_in = 0, n_out = 0;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n_in, sizeof(n_in));  // n_in reused below
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n_out, sizeof(n_out));
    in_attrs.resize(n_in);
    out_attrs.resize(n_out);
    for (uint32_t i = 0; i < n_in; ++i) {
      in_attrs[i] = {};
      in_attrs[i].index = i;
      rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < n_out; ++i) {
      out_attrs[i] = {};
      out_attrs[i].index = i;
      rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
    }
    printf("[i] %s: %u input(s), %u output(s)\n", path.c_str(), n_in, n_out);
    return true;
  }
};

void write_dump(const std::string& dir, const std::string& name,
                const float* data, size_t n, const std::string& shape) {
  if (dir.empty()) return;
  std::string base = dir + "/" + name;
  FILE* f = fopen((base + ".bin").c_str(), "wb");
  if (!f) return;
  fwrite(data, sizeof(float), n, f);
  fclose(f);
  f = fopen((base + ".shape.txt").c_str(), "w");
  if (f) {
    fprintf(f, "%s\n", shape.c_str());
    fclose(f);
  }
}

// Resize to 512x512 exactly as the Python reference does (bilinear, no crop).
cv::Mat prepare(const cv::Mat& bgr) {
  cv::Mat rgb, resized;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(kSize, kSize), 0, 0, cv::INTER_LINEAR);
  return resized;
}

// One image's stage-1 outputs.  `outs` owns the runtime buffers, so the raw
// pointers stay valid until `release()` - which is why the features are released
// only after the matcher has consumed them.
struct Features {
  std::vector<rknn_output> outs;
  const float* kpts = nullptr;
  const float* desc = nullptr;
  const float* scores = nullptr;

  void release(rknn_context ctx) {
    if (!outs.empty()) {
      rknn_outputs_release(ctx, static_cast<uint32_t>(outs.size()), outs.data());
      outs.clear();
    }
  }
};

// Stage 1 on ONE image: (1, 512, 512, 3) uint8 NHWC -> keypoints/descriptors/scores.
// A pair is two of these; see the design note at the top of the file for what that
// costs and what it buys.
bool extract(Model& m, const cv::Mat& rgb, Features& f) {
  std::vector<rknn_input> in(1);
  in[0].index = 0;
  in[0].type = RKNN_TENSOR_UINT8;
  in[0].fmt = RKNN_TENSOR_NHWC;
  in[0].size = kSize * kSize * 3;
  in[0].buf = rgb.data;
  if (rknn_inputs_set(m.ctx, 1, in.data()) != 0) {
    fprintf(stderr, "[E] rknn_inputs_set(stage1) failed\n");
    return false;
  }
  if (rknn_run(m.ctx, nullptr) != 0) {
    fprintf(stderr, "[E] rknn_run(stage1) failed\n");
    return false;
  }

  const uint32_t n = m.out_attrs.size();
  f.outs.assign(n, rknn_output{});
  for (uint32_t i = 0; i < n; ++i) {
    f.outs[i].index = i;
    f.outs[i].want_float = 1;
  }
  if (rknn_outputs_get(m.ctx, n, f.outs.data(), nullptr) != 0) {
    fprintf(stderr, "[E] rknn_outputs_get(stage1) failed\n");
    return false;
  }

  // Identify outputs by element count rather than by position: the runtime does
  // not guarantee the ONNX output order.
  const size_t n_kpts = static_cast<size_t>(kKeypoints) * 2;
  const size_t n_desc = static_cast<size_t>(kKeypoints) * 128;
  const size_t n_scores = static_cast<size_t>(kKeypoints);
  for (uint32_t i = 0; i < n; ++i) {
    size_t c = f.outs[i].size / sizeof(float);
    if (c == n_desc) f.desc = static_cast<float*>(f.outs[i].buf);
    else if (c == n_kpts) f.kpts = static_cast<float*>(f.outs[i].buf);
    else if (c == n_scores) f.scores = static_cast<float*>(f.outs[i].buf);
  }
  if (!f.kpts || !f.desc || !f.scores) {
    fprintf(stderr, "[E] stage-1 outputs not identified (element counts seen:");
    for (uint32_t i = 0; i < n; ++i) {
      fprintf(stderr, " %u", f.outs[i].size / 4u);
    }
    fprintf(stderr, ")\n");
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s alike_stage.rknn lightglue_stage.rknn left.jpg right.jpg "
            "[--dump DIR]\n",
            argv[0]);
    return 1;
  }
  std::string dump_dir;
  for (int i = 5; i < argc; ++i) {
    if (std::string(argv[i]) == "--dump" && i + 1 < argc) dump_dir = argv[++i];
  }

  Model stage1, stage2;
  if (!stage1.load(argv[1]) || !stage2.load(argv[2])) return 1;

  // ---- host-side image preparation (the only non-NPU step) ----------------
  cv::Mat l = cv::imread(argv[3]), r = cv::imread(argv[4]);
  if (l.empty() || r.empty()) {
    fprintf(stderr, "[E] could not read one of the images\n");
    return 1;
  }
  cv::Mat lr = prepare(l), rr = prepare(r);

  // ---- stage 1: two calls, ONE IMAGE EACH ---------------------------------
  Features f0, f1;
  if (!extract(stage1, lr, f0) || !extract(stage1, rr, f1)) {
    f0.release(stage1.ctx);
    f1.release(stage1.ctx);
    return 1;
  }
  const size_t n_kp_f = static_cast<size_t>(kKeypoints) * 2;
  const size_t n_ds_f = static_cast<size_t>(kKeypoints) * 128;
  const size_t n_sc_f = static_cast<size_t>(kKeypoints);
  write_dump(dump_dir, "keypoints0", f0.kpts, n_kp_f, "1,512,2");
  write_dump(dump_dir, "descriptors0", f0.desc, n_ds_f, "1,512,128");
  write_dump(dump_dir, "scores0", f0.scores, n_sc_f, "1,512");
  write_dump(dump_dir, "keypoints1", f1.kpts, n_kp_f, "1,512,2");
  write_dump(dump_dir, "descriptors1", f1.desc, n_ds_f, "1,512,128");
  write_dump(dump_dir, "scores1", f1.scores, n_sc_f, "1,512");

  // ---- stage 2: ONE call, the pair's four tensors --------------------------
  // The matcher still consumes both views together - that part was never batched
  // in this sense: its inputs are two separate (1,K,...) feature sets.
  std::vector<rknn_input> in2(4);
  const size_t kp_bytes = sizeof(float) * kKeypoints * 2;
  const size_t ds_bytes = sizeof(float) * kKeypoints * 128;
  const float* ptrs[4] = {f0.kpts, f1.kpts, f0.desc, f1.desc};
  const size_t sizes[4] = {kp_bytes, kp_bytes, ds_bytes, ds_bytes};
  for (int i = 0; i < 4; ++i) {
    in2[i] = {};
    in2[i].index = i;
    in2[i].type = RKNN_TENSOR_FLOAT32;
    in2[i].fmt = RKNN_TENSOR_UNDEFINED;  // keep the model's own layout
    in2[i].size = sizes[i];
    in2[i].buf = const_cast<float*>(ptrs[i]);
  }
  CHECK(rknn_inputs_set(stage2.ctx, 4, in2.data()), "rknn_inputs_set(stage2)");
  CHECK(rknn_run(stage2.ctx, nullptr), "rknn_run(stage2)");

  const uint32_t n2 = stage2.out_attrs.size();
  std::vector<rknn_output> o2(n2);
  for (uint32_t i = 0; i < n2; ++i) {
    o2[i] = {};
    o2[i].index = i;
    o2[i].want_float = 1;
  }
  CHECK(rknn_outputs_get(stage2.ctx, n2, o2.data(), nullptr),
        "rknn_outputs_get(stage2)");

  const float* matches = nullptr;
  const float* mscores = nullptr;
  for (uint32_t i = 0; i < n2; ++i) {
    size_t n = o2[i].size / sizeof(float);
    if (n == kKeypoints) {
      if (!matches) matches = static_cast<float*>(o2[i].buf);
      else mscores = static_cast<float*>(o2[i].buf);
    }
  }
  if (!matches || !mscores) {
    fprintf(stderr, "[E] stage-2 outputs not identified\n");
    f0.release(stage1.ctx);
    f1.release(stage1.ctx);
    rknn_outputs_release(stage2.ctx, n2, o2.data());
    return 1;
  }
  write_dump(dump_dir, "matches0", matches, kKeypoints, "1,512");
  write_dump(dump_dir, "mscores0", mscores, kKeypoints, "1,512");

  // ---- report -------------------------------------------------------------
  // matches0 is -1 where a keypoint is unmatched.
  int n_match = 0;
  double sum_score = 0.0;
  for (int i = 0; i < kKeypoints; ++i) {
    if (matches[i] >= 0.0f) {
      ++n_match;
      sum_score += mscores[i];
    }
  }
  printf("[i] matches: %d / %d\n", n_match, kKeypoints);
  if (n_match) printf("[i] mean match score: %.4f\n", sum_score / n_match);

  // stdout is the deliverable: one match per line, `idx0 idx1 score`.
  for (int i = 0; i < kKeypoints; ++i) {
    if (matches[i] >= 0.0f) {
      int j = static_cast<int>(matches[i] + 0.5f);
      printf("%d %d %.6f\n", i, j, mscores[i]);
    }
  }

  f0.release(stage1.ctx);
  f1.release(stage1.ctx);
  rknn_outputs_release(stage2.ctx, n2, o2.data());
  return 0;
}
