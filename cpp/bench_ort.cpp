// C++ ONNX Runtime check + latency for the exported policy, same protocol as export_onnx.py:
// 1 intra-op thread, batch 1, 50 warm-up runs, n timed runs cycling over rollout observations.
// Inputs from cpp/dump_obs.py: obs.bin (N x 48 float32) and ref.bin (N x 14, Python ORT output).
// Usage: bench_ort <model.onnx> <obs.bin> <ref.bin> [n]
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

constexpr int64_t kObs = 48, kAct = 14;

static std::vector<float> read_f32(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  std::vector<float> v(static_cast<size_t>(f.tellg()) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  return v;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: %s model.onnx obs.bin ref.bin [n]\n", argv[0]); return 2; }
  const int n = argc > 4 ? std::atoi(argv[4]) : 2000;
  std::vector<float> obs = read_f32(argv[2]), ref = read_f32(argv[3]);
  const size_t rows = obs.size() / kObs;
  if (rows == 0 || ref.size() != rows * kAct) { std::fprintf(stderr, "obs/ref size mismatch\n"); return 2; }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(1);
  so.SetInterOpNumThreads(1);
  Ort::Session s(env, argv[1], so);
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t ishape[2] = {1, kObs}, oshape[2] = {1, kAct};
  std::vector<Ort::Value> in;
  for (size_t i = 0; i < rows; ++i)
    in.push_back(Ort::Value::CreateTensor<float>(mem, obs.data() + i * kObs, kObs, ishape, 2));
  std::vector<float> act(kAct);
  Ort::Value out = Ort::Value::CreateTensor<float>(mem, act.data(), act.size(), oshape, 2);
  const char* iname[] = {"obs"};
  Ort::AllocatedStringPtr on = s.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  const char* oname[] = {on.get()};
  Ort::RunOptions ro;

  double diff = 0;
  for (size_t i = 0; i < rows; ++i) {
    s.Run(ro, iname, &in[i], 1, oname, &out, 1);
    for (int64_t j = 0; j < kAct; ++j) diff = std::max(diff, double(std::fabs(act[j] - ref[i * kAct + j])));
  }
  for (int i = 0; i < 50; ++i) s.Run(ro, iname, &in[i % rows], 1, oname, &out, 1);
  std::vector<double> ms(n);
  for (int i = 0; i < n; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    s.Run(ro, iname, &in[i % rows], 1, oname, &out, 1);
    ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::sort(ms.begin(), ms.end());
  std::printf("C++ ONNX Runtime %s, 1 thread, batch 1: max |C++ - Python| %.2e over %zu obs; "
              "n %d p50 %.4f ms p99 %.4f ms\n",
              OrtGetApiBase()->GetVersionString(), diff, rows, n, ms[n / 2], ms[(n * 99) / 100]);
  return diff < 1e-5 ? 0 : 1;
}
