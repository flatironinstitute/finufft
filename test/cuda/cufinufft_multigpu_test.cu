// Exercise opts.gpu_device_id and a caller-owned opts.gpu_stream.
//
// The same 2D type-1 problem is run on every visible device, with and without a
// user stream; every result must match the device-0 default-stream one to a
// tight relative L2 error. The call is always made with device 0 current, so it
// also checks that cufinufft restores the caller's device. An unusable device id
// must come back as an error code. With one visible device this still covers the
// stream path, but the cross-device part needs two (a MIG pod exposes only one
// instance per process), so it then exits 77 and ctest reports a SKIP.

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cufinufft.h>
#include <finufft_common/common.h>
#include <finufft_errors.h>

#include "../utils/norms.hpp"

using ::finufft::common::PI;

namespace {

constexpr int64_t M = 10000, N1 = 64, N2 = 32;
constexpr size_t NK = size_t(N1) * N2;
constexpr double REQ_TOL = 1e-9, CHECK_RTOL = 1e-9;

// Run the transform on `device`, optionally through a caller-owned stream on
// that device, with device 0 current across the call. Returns the modes, or an
// empty vector on failure.
// ponytail: device buffers are not freed - the process exits right after.
std::vector<std::complex<double>> run_on_device(
    int device, bool use_stream, const std::vector<double> &x,
    const std::vector<double> &y, const std::vector<std::complex<double>> &c) {
  double *dx, *dy;
  cuDoubleComplex *dc, *dfk;
  cudaStream_t stream{};
  if (cudaSetDevice(device) || cudaMalloc(&dx, M * sizeof(*dx)) ||
      cudaMalloc(&dy, M * sizeof(*dy)) || cudaMalloc(&dc, M * sizeof(*dc)) ||
      cudaMalloc(&dfk, NK * sizeof(*dfk)) || (use_stream && cudaStreamCreate(&stream)) ||
      cudaMemcpy(dx, x.data(), M * sizeof(*dx), cudaMemcpyHostToDevice) ||
      cudaMemcpy(dy, y.data(), M * sizeof(*dy), cudaMemcpyHostToDevice) ||
      cudaMemcpy(dc, c.data(), M * sizeof(*dc), cudaMemcpyHostToDevice)) {
    std::fprintf(stderr, "setup failed on device %d: %s\n", device,
                 cudaGetErrorString(cudaGetLastError()));
    return {};
  }

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_device_id = device;
  opts.gpu_stream = use_stream ? (void *)stream : nullptr;

  cudaSetDevice(0); // the library must cope with, and restore, this
  const int ier = cufinufft2d1(M, dx, dy, dc, 1, REQ_TOL, N1, N2, dfk, &opts);
  int current = -1;
  cudaGetDevice(&current);
  if (ier || current != 0) {
    std::fprintf(stderr, "device %d: ier %d, current device %d (expected 0)\n", device,
                 ier, current);
    return {};
  }

  cudaSetDevice(device);
  if (use_stream) cudaStreamSynchronize(stream);
  std::vector<std::complex<double>> fk(NK);
  if (cudaMemcpy(fk.data(), dfk, NK * sizeof(*dfk), cudaMemcpyDeviceToHost)) {
    std::fprintf(stderr, "readback failed on device %d: %s\n", device,
                 cudaGetErrorString(cudaGetLastError()));
    return {};
  }
  return fk;
}

// An unusable gpu_device_id must come back as an error code, not a crash: the
// plan ctor's DeviceSwitcher throws and the C boundary maps it.
bool bad_device_id_is_rejected(int ndev) {
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_device_id = ndev; // one past the last valid device
  cufinufft_plan plan;
  const int64_t nmodes[3] = {N1, N2, 1};
  const int ier = cufinufft_makeplan(1, 2, nmodes, 1, 1, REQ_TOL, &plan, &opts);
  if (ier != FINUFFT_ERR_CUDA_FAILURE) {
    std::fprintf(stderr, "gpu_device_id=%d gave ier %d, expected %d\n", ndev, ier,
                 FINUFFT_ERR_CUDA_FAILURE);
    return false;
  }
  cudaGetLastError(); // clear the sticky invalid-device error
  return true;
}

} // namespace

int main() {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess) return 1;
  std::printf("%d device(s) visible\n", ndev);

  // Same inputs for every device: a fixed lattice-like point set, no RNG needed.
  std::vector<double> x(M), y(M);
  std::vector<std::complex<double>> c(M);
  for (int64_t j = 0; j < M; ++j) {
    x[j] = -PI + 2 * PI * double(j % 97) / 97.0;
    y[j] = -PI + 2 * PI * double(j % 89) / 89.0;
    c[j] = {std::cos(0.1 * double(j)), std::sin(0.07 * double(j))};
  }

  const auto ref = run_on_device(0, false, x, y, c);
  if (ref.empty()) return 1;

  int ret = 0;
  for (int device = 0; device < ndev; ++device) {
    for (const bool use_stream : {false, true}) {
      if (device == 0 && !use_stream) continue; // that is the reference run
      const auto fk = run_on_device(device, use_stream, x, y, c);
      if (fk.empty()) return 1;

      const double err = relerrtwonorm(int64_t(NK), ref.data(), fk.data());
      std::printf("device %d (stream %s): rel l2 err vs device 0 = %.3g\n", device,
                  use_stream ? "user" : "default", err);
      if (!(err < CHECK_RTOL)) {
        std::fprintf(stderr, "device %d: mismatch vs device 0\n", device);
        ret = 1;
      }
    }
  }

  if (!bad_device_id_is_rejected(ndev)) ret = 1;
  if (ret) return ret;

  // The stream path ran, but with one device nothing above ever left device 0,
  // so the cross-device part of this test did not run. Report that as a ctest
  // SKIP (see SKIP_RETURN_CODE) rather than a green that overstates coverage.
  if (ndev < 2) {
    std::printf("only 1 device visible: cross-device coverage skipped\n");
    return 77;
  }
  return 0;
}
