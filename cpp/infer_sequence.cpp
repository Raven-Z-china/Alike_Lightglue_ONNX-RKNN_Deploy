// RKNN C API sample: one reference image matched against N others.
//
// This is the workflow the per-image stage 1 exists for.  `infer` (single pair)
// extracts both views every time it runs; here the REFERENCE is extracted ONCE and
// then reused for every other image, so matching a frame costs one extraction plus
// one match instead of two extractions plus one match.  On the board that is the
// difference between ~2x and ~1x stage-1 work per frame, which is why the design
// note in `infer.cpp` accepts the extra dispatch per pair.
//
// What the numbers mean
// ---------------------
// Three times are reported per frame and they are NOT interchangeable:
//   extract_ms  stage 1 on the current image - the part that is skipped for a
//               cached reference, and the only part that scales with image count
//   match_ms    stage 2 on the pair, which every frame pays
//   total_ms    frame average including the one-off reference extraction and the
//               host-side resize, i.e. throughput as measured end to end
// `infer` prints only per-pair work and no timer at all; this sample exists to make
// the cached-extraction win visible, so it prints all three.
//
// Assumptions, stated rather than guessed
// ---------------------------------------
// * The output directory must exist - this sample does not create it, and does not
//   shell out to `mkdir`.
// * --ext ".png" restricts the folder scan to one suffix; without it every regular
//   file in the folder is attempted and anything unreadable is reported and skipped.
// * The reference is the FIRST image in sorted order (the same rule the Python
//   harness uses), or the file named by --sort none's natural directory order when
//   --sort none is passed.
//
// Build: see CMakeLists.txt.  Requires librknn_api (aarch64) and OpenCV.
//
// Usage:
//   ./infer_sequence alike_stage.rknn lightglue_stage.rknn image_dir/ out_dir/
//       [--ext .png] [--sort name|none] [--vis] [--dump DIR]
//
// stdout is one block per frame:
//   # <path> extract_ms match_ms matches mean_score
//   idx0 idx1 score
//   ...
// With --vis, `out_dir` also gets a side-by-side match image per frame.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

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
constexpr int kDescDim = 128;

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

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
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n_in, sizeof(n_in));
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

struct Features {
  std::vector<rknn_output> outs;
  const float* kpts = nullptr;
  const float* desc = nullptr;
  const float* scores = nullptr;
  double extract_ms = 0.0;   // stage-1 time for THIS image (0 for a cached one)

  void release(rknn_context ctx) {
    if (!outs.empty()) {
      rknn_outputs_release(ctx, static_cast<uint32_t>(outs.size()), outs.data());
      outs.clear();
    }
  }
};

// Stage 1 on ONE image: (1, 512, 512, 3) uint8 NHWC -> keypoints/descriptors/scores.
bool extract(Model& m, const cv::Mat& rgb, Features& f) {
  std::vector<rknn_input> in(1);
  in[0].index = 0;
  in[0].type = RKNN_TENSOR_UINT8;
  in[0].fmt = RKNN_TENSOR_NHWC;
  in[0].size = kSize * kSize * 3;
  in[0].buf = rgb.data;
  auto t0 = Clock::now();
  if (rknn_inputs_set(m.ctx, 1, in.data()) != 0) {
    fprintf(stderr, "[E] rknn_inputs_set(stage1) failed\n");
    return false;
  }
  if (rknn_run(m.ctx, nullptr) != 0) {
    fprintf(stderr, "[E] rknn_run(stage1) failed\n");
    return false;
  }
  f.extract_ms = ms_since(t0);

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
  const size_t n_desc = static_cast<size_t>(kKeypoints) * kDescDim;
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

// Stage 2 on a pair, into caller-owned buffers: the matches are kept after the
// runtime buffers are released, so the two views can be freed as soon as they are
// no longer needed by the NPU (the reference lives for the whole sequence).
bool match_pair(Model& m, const Features& a, const Features& b,
                std::vector<float>& matches, std::vector<float>& mscores,
                double& match_ms) {
  const size_t kp_bytes = sizeof(float) * kKeypoints * 2;
  const size_t ds_bytes = sizeof(float) * kKeypoints * kDescDim;
  const float* ptrs[4] = {a.kpts, b.kpts, a.desc, b.desc};
  const size_t sizes[4] = {kp_bytes, kp_bytes, ds_bytes, ds_bytes};

  std::vector<rknn_input> in(4);
  for (int i = 0; i < 4; ++i) {
    in[i] = {};
    in[i].index = i;
    in[i].type = RKNN_TENSOR_FLOAT32;
    in[i].fmt = RKNN_TENSOR_UNDEFINED;  // keep the model's own layout
    in[i].size = sizes[i];
    in[i].buf = const_cast<float*>(ptrs[i]);
  }

  auto t0 = Clock::now();
  if (rknn_inputs_set(m.ctx, 4, in.data()) != 0) {
    fprintf(stderr, "[E] rknn_inputs_set(stage2) failed\n");
    return false;
  }
  if (rknn_run(m.ctx, nullptr) != 0) {
    fprintf(stderr, "[E] rknn_run(stage2) failed\n");
    return false;
  }

  const uint32_t n = m.out_attrs.size();
  std::vector<rknn_output> outs(n);
  for (uint32_t i = 0; i < n; ++i) {
    outs[i] = {};
    outs[i].index = i;
    outs[i].want_float = 1;
  }
  if (rknn_outputs_get(m.ctx, n, outs.data(), nullptr) != 0) {
    fprintf(stderr, "[E] rknn_outputs_get(stage2) failed\n");
    return false;
  }
  match_ms = ms_since(t0);

  // Both outputs are (1, 512) float32, so shape cannot separate them: the
  // distractor is that `matches0` holds keypoint INDICES, which are integers, so
  // it is the one with no fractional part.  Ties are broken by order only as a
  // last resort, and reported, because a silent swap here would look like a
  // plausible - but wrong - result.
  const float* first = nullptr;
  const float* second = nullptr;
  for (uint32_t i = 0; i < n; ++i) {
    if (outs[i].size / sizeof(float) != static_cast<size_t>(kKeypoints)) continue;
    if (!first) first = static_cast<float*>(outs[i].buf);
    else second = static_cast<float*>(outs[i].buf);
  }
  if (!first || !second) {
    fprintf(stderr, "[E] stage-2 outputs not identified (sizes:");
    for (uint32_t i = 0; i < n; ++i) {
      fprintf(stderr, " %u", outs[i].size / 4u);
    }
    fprintf(stderr, ")\n");
    rknn_outputs_release(m.ctx, n, outs.data());
    return false;
  }
  auto integral = [](const float* p) {
    for (int i = 0; i < kKeypoints; ++i) {
      if (p[i] >= 0.0f && p[i] != std::floor(p[i])) return false;
    }
    return true;
  };
  const float* idx = integral(first) ? first : (integral(second) ? second : first);
  const float* sc = (idx == first) ? second : first;
  matches.assign(idx, idx + kKeypoints);
  mscores.assign(sc, sc + kKeypoints);
  rknn_outputs_release(m.ctx, n, outs.data());
  return true;
}

std::vector<std::string> list_images(const std::string& dir, const std::string& ext,
                                    bool sort_names) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) {
    fprintf(stderr, "[E] cannot open directory %s\n", dir.c_str());
    return out;
  }
  while (struct dirent* e = readdir(d)) {
    std::string name = e->d_name;
    if (name == "." || name == "..") continue;
    if (!ext.empty() && name.size() >= ext.size() &&
        name.compare(name.size() - ext.size(), ext.size(), ext) != 0) {
      continue;
    }
    struct stat st;
    std::string path = dir + "/" + name;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
    out.push_back(path);
  }
  closedir(d);
  if (sort_names) std::sort(out.begin(), out.end());
  return out;
}

void visualize(const cv::Mat& a_rgb, const Features& fa, const cv::Mat& b_rgb,
               const Features& fb, const std::vector<float>& matches,
               const std::vector<float>& mscores, const std::string& title,
               cv::Mat& out) {
  std::vector<cv::KeyPoint> ka, kb;
  ka.reserve(kKeypoints);
  kb.reserve(kKeypoints);
  for (int i = 0; i < kKeypoints; ++i) {
    ka.emplace_back(fa.kpts[i * 2], fa.kpts[i * 2 + 1], 4.f, -1.f, fa.scores[i]);
    kb.emplace_back(fb.kpts[i * 2], fb.kpts[i * 2 + 1], 4.f, -1.f, fb.scores[i]);
  }
  std::vector<cv::DMatch> dm;
  for (int i = 0; i < kKeypoints; ++i) {
    if (matches[i] >= 0.0f) {
      dm.emplace_back(i, static_cast<int>(matches[i] + 0.5f), mscores[i]);
    }
  }
  cv::Mat a_bgr, b_bgr;
  cv::cvtColor(a_rgb, a_bgr, cv::COLOR_RGB2BGR);
  cv::cvtColor(b_rgb, b_bgr, cv::COLOR_RGB2BGR);
  cv::drawMatches(a_bgr, ka, b_bgr, kb, dm, out, cv::Scalar::all(-1),
                  cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::putText(out, title, cv::Point(8, 26), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(0, 0, 255), 2);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s alike_stage.rknn lightglue_stage.rknn image_dir out_dir "
            "[--ext .png] [--sort name|none] [--vis] [--dump DIR]\n",
            argv[0]);
    return 1;
  }
  const std::string alike_path = argv[1], lg_path = argv[2];
  const std::string image_dir = argv[3], out_dir = argv[4];
  std::string ext, dump_dir, sort_mode = "name";
  bool vis = false;
  for (int i = 5; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ext" && i + 1 < argc) ext = argv[++i];
    else if (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
    else if (a == "--sort" && i + 1 < argc) sort_mode = argv[++i];
    else if (a == "--vis") vis = true;
    else {
      fprintf(stderr, "[E] unknown argument: %s\n", a.c_str());
      return 1;
    }
  }

  struct stat st;
  if (stat(out_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
    fprintf(stderr, "[E] output directory does not exist: %s\n"
                    "    create it first; this sample does not create it for you\n",
            out_dir.c_str());
    return 1;
  }

  std::vector<std::string> images = list_images(image_dir, ext, sort_mode != "none");
  if (images.size() < 2) {
    fprintf(stderr, "[E] need at least 2 images in %s, found %zu\n",
            image_dir.c_str(), images.size());
    return 1;
  }
  printf("[i] %zu images from %s (sort=%s%s)\n", images.size(), image_dir.c_str(),
         sort_mode.c_str(), ext.empty() ? "" : (", ext=" + ext).c_str());

  Model stage1, stage2;
  if (!stage1.load(alike_path) || !stage2.load(lg_path)) return 1;

  auto t_start = Clock::now();

  // ---- reference: extracted ONCE, reused for every other image -------------
  cv::Mat ref_bgr = cv::imread(images[0]);
  if (ref_bgr.empty()) {
    fprintf(stderr, "[E] cannot read reference image %s\n", images[0].c_str());
    return 1;
  }
  cv::Mat ref_rgb = prepare(ref_bgr);
  Features ref;
  auto t_ref = Clock::now();
  if (!extract(stage1, ref_rgb, ref)) {
    ref.release(stage1.ctx);
    return 1;
  }
  const double ref_setup_ms = ms_since(t_ref);
  printf("[i] reference %s: extracted once in %.1f ms (%.1f ms incl. decode+resize)\n",
         images[0].c_str(), ref.extract_ms, ref_setup_ms);

  int frames = 0, failures = 0;
  double sum_extract = 0, sum_match = 0;
  long long total_matches = 0;

  for (size_t i = 1; i < images.size(); ++i) {
    cv::Mat other = cv::imread(images[i]);
    if (other.empty()) {
      fprintf(stderr, "[W] skip unreadable image %s\n", images[i].c_str());
      ++failures;
      continue;
    }
    cv::Mat other_rgb = prepare(other);

    Features cur;
    if (!extract(stage1, other_rgb, cur)) {
      cur.release(stage1.ctx);
      ++failures;
      continue;
    }
    std::vector<float> matches, mscores;
    double match_ms = 0;
    bool ok = match_pair(stage2, ref, cur, matches, mscores, match_ms);

    // The reference is extracted once and reused; zeroing its timing here keeps a
    // future per-frame line from accidentally claiming that cost again.
    ref.extract_ms = 0.0;

    if (!ok) {
      cur.release(stage1.ctx);
      ++failures;
      continue;
    }

    int n_valid = 0;
    double score_sum = 0;
    for (int k = 0; k < kKeypoints; ++k) {
      if (matches[k] >= 0.0f) {
        ++n_valid;
        score_sum += mscores[k];
      }
    }
    const double mean_score = n_valid ? score_sum / n_valid : 0.0;
    printf("# %s extract_ms %.2f match_ms %.2f matches %d mean_score %.4f\n",
           images[i].c_str(), cur.extract_ms, match_ms, n_valid, mean_score);
    for (int k = 0; k < kKeypoints; ++k) {
      if (matches[k] >= 0.0f) {
        printf("%d %d %.6f\n", k, static_cast<int>(matches[k] + 0.5f), mscores[k]);
      }
    }
    fflush(stdout);

    sum_extract += cur.extract_ms;
    sum_match += match_ms;
    total_matches += n_valid;
    ++frames;

    if (vis) {
      cv::Mat canvas;
      char title[128];
      snprintf(title, sizeof(title), "extract %.1f ms  match %.1f ms  hits %d",
               cur.extract_ms, match_ms, n_valid);
      visualize(ref_rgb, ref, other_rgb, cur, matches, mscores, title, canvas);
      size_t slash = images[i].find_last_of('/');
      std::string base = (slash == std::string::npos) ? images[i] : images[i].substr(slash + 1);
      cv::imwrite(out_dir + "/" + base + ".matches.png", canvas);
    }

    if (i == 1) {  // dump the first pair, symmetric with infer.cpp's --dump
      write_dump(dump_dir, "keypoints0", ref.kpts, kKeypoints * 2, "1,512,2");
      write_dump(dump_dir, "descriptors0", ref.desc, kKeypoints * kDescDim, "1,512,128");
      write_dump(dump_dir, "scores0", ref.scores, kKeypoints, "1,512");
      write_dump(dump_dir, "keypoints1", cur.kpts, kKeypoints * 2, "1,512,2");
      write_dump(dump_dir, "descriptors1", cur.desc, kKeypoints * kDescDim, "1,512,128");
      write_dump(dump_dir, "scores1", cur.scores, kKeypoints, "1,512");
      write_dump(dump_dir, "matches0", matches.data(), kKeypoints, "1,512");
      write_dump(dump_dir, "mscores0", mscores.data(), kKeypoints, "1,512");
    }

    cur.release(stage1.ctx);
  }

  ref.release(stage1.ctx);
  const int processed = frames ? frames : 1;
  printf("\n[i] %d frame(s), %d failure(s)\n", frames, failures);
  printf("[i] per frame: extract %.1f ms (mean), match %.1f ms (mean)\n",
         sum_extract / processed, sum_match / processed);
  printf("[i] total %.1f ms -> %.2f ms/frame end to end incl. the one-off reference\n",
         ms_since(t_start), ms_since(t_start) / processed);
  printf("[i] matches total %lld (mean %.1f/frame)\n", total_matches,
         static_cast<double>(total_matches) / processed);
  return failures ? 1 : 0;
}
