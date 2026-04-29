#ifndef CUFINUFFT_PLAN_T_H
#define CUFINUFFT_PLAN_T_H

#include <cuda/std/array>
#include <cufft.h>
#include <cufinufft/defs.h>
#include <cufinufft/types.h>
#include <cufinufft_opts.h>
#include <finufft_common/spread_opts.h>
#include <finufft_errors.h>
#include <optional>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <type_traits>

#include <cstddef>
#include <cuComplex.h>
#include <memory>

/* This header file contains the internal plan class of cufinufft,
   as well as other types and functions which are exclusively
   needed in combination with that class. */

template<typename T> inline T *dethrust(thrust::device_ptr<T> ptr) {
  return thrust::raw_pointer_cast(ptr);
}
template<typename T> inline thrust::device_ptr<T> enthrust(T *ptr) {
  return thrust::device_pointer_cast(ptr);
}

class DeviceSwitcher {
private:
  int orig_device;

  static int get_orig_device() noexcept {
    int device{};
    cudaGetDevice(&device);
    return device;
  }

public:
  explicit DeviceSwitcher(int newDevice) : orig_device{get_orig_device()} {
    if (cudaSetDevice(newDevice) != cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
  }

  ~DeviceSwitcher() {
    if (cudaSetDevice(orig_device) != cudaSuccess) {
      std::cerr << "failure reverting to original CUDA device; Exiting..." << std::endl;
      std::terminate();
    }
  }
};

template<typename T>
struct ThrustAllocatorAsync : public thrust::device_malloc_allocator<T> {
public:
  using Base      = thrust::device_malloc_allocator<T>;
  using pointer   = typename Base::pointer;
  using size_type = typename Base::size_type;

private:
  cudaStream_t stream;
  int deviceID;
  bool pool;

public:
  // Prefer explicit stream; no default ctor needed if you always pass alloc to
  // device_vector
  template<typename U> friend struct ThrustAllocatorAsync;

  explicit ThrustAllocatorAsync(cudaStream_t s, int ID, bool supports_pools)
      : stream(s), deviceID(ID), pool(supports_pools) {}

  template<typename U>
  ThrustAllocatorAsync(const ThrustAllocatorAsync<U> &o)
      : stream(o.stream), deviceID(o.deviceID), pool(o.pool) {}

  pointer allocate(size_type n) {
    DeviceSwitcher switcher(deviceID);
    T *p = nullptr;
    auto err =
        pool ? cudaMallocAsync(&p, n * sizeof(T), stream) : cudaMalloc(&p, n * sizeof(T));
    if (err != cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
    return enthrust(p);
  }

  void deallocate(pointer p, size_type) {
    DeviceSwitcher switcher(deviceID);
    auto err = pool ? cudaFreeAsync(dethrust(p), stream) : cudaFree(dethrust(p));
    if (err != cudaSuccess) { // something went really, really wrong, memory is corrupted
      std::cerr << "error while deallocating GPU memory! Exiting ..." << std::endl;
      std::terminate();
    }
  }
};

template<typename T> using gpu_array = thrust::device_vector<T, ThrustAllocatorAsync<T>>;

template<typename T> inline T *dethrust(gpu_array<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
}
template<typename T> inline const T *dethrust(const gpu_array<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
}
template<typename T>
inline cuda::std::array<T *, 3> dethrust(cuda::std::array<gpu_array<T>, 3> &arr) {
  cuda::std::array<T *, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = dethrust(arr[i]);
  return res;
}
template<typename T>
inline cuda::std::array<const T *, 3> dethrust(
    const cuda::std::array<gpu_array<T>, 3> &arr) {
  cuda::std::array<const T *, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = dethrust(arr[i]);
  return res;
}

template<typename T> struct cufinufft_plan_t {
  // FIXME: we want to make data members private in the future.
  // Not yet possible at the moment, since not all functions working
  // on plans have been converted to members.

  cufinufft_opts opts;
  bool supports_pools = false;
  finufft_spread_opts spopts;

  ThrustAllocatorAsync<std::byte> alloc{(cudaStream_t)opts.gpu_stream, opts.gpu_device_id,
                                        supports_pools};

  int type                                    = 0;
  int dim                                     = 0;
  CUFINUFFT_BIGINT M                          = 0;
  cuda::std::array<CUFINUFFT_BIGINT, 3> nf123 = {0, 0, 0};
  cuda::std::array<CUFINUFFT_BIGINT, 3> mstu  = {0, 0, 0};
  int ntransf                                 = 0;
  int batchsize                               = 0;
  int iflag                                   = 0;

  int totalnumsubprob                         = 0;
  cuda::std::array<gpu_array<T>, 3> fwkerhalf = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<const T *, 3> kxyz     = {nullptr, nullptr, nullptr};
  cuda::std::array<gpu_array<T>, 3> kxyzp = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};

  // Type 3 specific
  struct {
    cuda::std::array<T, 3> X = {0, 0, 0}, C = {0, 0, 0}, S = {0, 0, 0}, D = {0, 0, 0},
                           h = {0, 0, 0}, gam = {0, 0, 0};
  } type3_params;
  int N                                  = 0; // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf                    = 0;
  cuda::std::array<const T *, 3> STU     = {nullptr, nullptr, nullptr};
  cuda::std::array<gpu_array<T>, 3> STUp = {
      gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}, gpu_array<T>{0, alloc}};
  T tol = 0;
  // inner type 2 plan for type 3
  std::unique_ptr<cufinufft_plan_t<T>> t2_plan;

  gpu_array<cuda_complex<T>> prephase{0, alloc}; // pre-phase, for all input NU pts
  gpu_array<cuda_complex<T>> deconv{0, alloc};   // reciprocal of kernel FT, phase, all
                                                 // output NU pts

  // Arrays that used in subprob method
  gpu_array<int> idxnupts{0, alloc};   // length: #nupts, index of the nupts in the
                                       // bin-sorted order
  gpu_array<int> sortidx{0, alloc};    // length: #nupts, order inside the bin the nupt
                                       // belongs to
  gpu_array<int> numsubprob{0, alloc}; // length: #bins,  number of subproblems in each
                                       // bin
  gpu_array<int> binsize{0, alloc}; // length: #bins, number of nonuniform ponits in each
                                    // bin
  gpu_array<int> binstartpts{0, alloc}; // length: #bins, exclusive scan of array binsize
  gpu_array<int> subprob_to_bin{0, alloc}; // length: #subproblems, the bin the subproblem
                                           // works on
  gpu_array<int> subprobstartpts{0, alloc}; // length: #bins, exclusive scan of array
                                            // numsubprob

  // Arrays for 3d (need to sort out)
  gpu_array<int> numnupts{0, alloc};
  gpu_array<int> subprob_to_nupts{0, alloc};

  cufftHandle fftplan = 0;
  cudaStream_t stream = 0;

  bool eps_too_small = false;

  cufinufft_plan_t() = delete;
  cufinufft_plan_t(int type_, int dim_, const int *nmodes, int iflag_, int ntransf_,
                   T tol_, const cufinufft_opts &opts_);
  cufinufft_plan_t &operator=(cufinufft_plan_t &) = delete;

  ~cufinufft_plan_t() {
    DeviceSwitcher switcher(opts.gpu_device_id);
    if (fftplan) cufftDestroy(fftplan);
  }

private:
  void exec1(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const;
  // The "ntransf_override" parameter is only needed when a type 3 plan calls
  // its inner type 2 plan. Leave at default in all other circumstances!
  void exec2(cuda_complex<T> *d_c, cuda_complex<T> *d_fk,
             std::optional<int> ntransf_override = std::optional<int>()) const;
  void exec3(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const;

  void deconvolve(cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const;
  template<int modeord, int ndim>
  void deconvolve_nd(cuda_complex<T> *fw, cuda_complex<T> *fk, int blksize) const;

  void setpts_12(int M_, const T *d_kx, const T *d_ky, const T *d_kz);
  void allocate();
  void allocate_nupts();

public:
  void setpts(int M_, const T *d_kx, const T *d_ky, const T *d_kz, int N_, const T *d_s,
              const T *d_t, const T *d_u);
  // FIXME: we want to make this "const" in the future
  void exec(cuda_complex<T> *d_c, cuda_complex<T> *d_fk) const;
};

// This class contains a subset of the information stored in
// cufinufft_plan_t, in a shape that can be copied to GPU.
// Its only intended use is for (implicit) on-the-fly construction
// from a full cufinufft_plan_t object when launching a kernel, such
// that the kernel has convenient access to the relevant parameters
// and arrays. Do not cache such objects or its members!
template<typename T> struct cufinufft_gpu_data {
  cufinufft_opts opts;
  finufft_spread_opts spopts;

  int type                                    = 0;
  int dim                                     = 0;
  CUFINUFFT_BIGINT M                          = 0;
  cuda::std::array<CUFINUFFT_BIGINT, 3> nf123 = {0, 0, 0};
  cuda::std::array<CUFINUFFT_BIGINT, 3> mstu  = {0, 0, 0};
  int ntransf                                 = 0;
  int batchsize                               = 0;
  int iflag                                   = 0;

  int totalnumsubprob = 0;

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<const T *, 3> xyz = {nullptr, nullptr, nullptr};

  int N                              = 0; // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf                = 0;
  cuda::std::array<const T *, 3> STU = {nullptr, nullptr, nullptr};
  T tol                              = 0;

  const cuda_complex<T> *prephase = nullptr; // pre-phase, for all input NU pts
  const cuda_complex<T> *deconv   = nullptr; // reciprocal of kernel FT, phase, all
                                             // output NU pts

  // Arrays that used in subprob method
  const int *idxnupts = nullptr;   // length: #nupts, index of the nupts in the
                                   // bin-sorted order
  const int *sortidx = nullptr;    // length: #nupts, order inside the bin the nupt
                                   // belongs to
  const int *numsubprob = nullptr; // length: #bins,  number of subproblems in each
                                   // bin
  const int *binsize = nullptr;    // length: #bins, number of nonuniform ponits in each
                                   // bin
  const int *binstartpts    = nullptr;  // length: #bins, exclusive scan of array binsize
  const int *subprob_to_bin = nullptr;  // length: #subproblems, the bin the subproblem
                                        // works on
  const int *subprobstartpts = nullptr; // length: #bins, exclusive scan of array
                                        // numsubprob

  // Arrays for 3d (need to sort out)
  const int *numnupts         = nullptr;
  const int *subprob_to_nupts = nullptr;

  cufinufft_gpu_data() = delete;
  cufinufft_gpu_data(const cufinufft_plan_t<T> &orig)
      : opts(orig.opts), spopts(orig.spopts), type(orig.type), dim(orig.dim), M(orig.M),
        nf123(orig.nf123), mstu(orig.mstu), ntransf(orig.ntransf),
        batchsize(orig.batchsize), iflag(orig.iflag),
        totalnumsubprob(orig.totalnumsubprob), xyz(orig.kxyz), N(orig.N), nf(orig.nf),
        STU(orig.STU), tol(orig.tol), prephase(dethrust(orig.prephase)),
        deconv(dethrust(orig.deconv)), idxnupts(dethrust(orig.idxnupts)),
        sortidx(dethrust(orig.sortidx)), numsubprob(dethrust(orig.numsubprob)),
        binsize(dethrust(orig.binsize)), binstartpts(dethrust(orig.binstartpts)),
        subprob_to_bin(dethrust(orig.subprob_to_bin)),
        subprobstartpts(dethrust(orig.subprobstartpts)),
        numnupts(dethrust(orig.numnupts)),
        subprob_to_nupts(dethrust(orig.subprob_to_nupts)) {}
};

#endif
