#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <cufft.h>
#include <cuda/std/array>

#include <cufinufft/defs.h>
#include <cufinufft_opts.h>
#include <finufft_common/spread_opts.h>
#include <type_traits>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>

#include <cuComplex.h>

#define CUFINUFFT_BIGINT int

// Marco Barbone 8/5/2024, replaced the ugly trick with std::conditional
// to define cuda_complex
// by using std::conditional and std::is_same, we can define cuda_complex
// if T is float, cuda_complex<T> is cuFloatComplex
// if T is double, cuda_complex<T> is cuDoubleComplex
// where cuFloatComplex and cuDoubleComplex are defined in cuComplex.h
// TODO: migrate to cuda/std/complex and remove this
//       Issue: cufft seems not to support cuda::std::complex
//       A reinterpret_cast should be enough
template<typename T>
using cuda_complex = typename std::conditional<
    std::is_same<T, float>::value, cuFloatComplex,
    typename std::conditional<std::is_same<T, double>::value, cuDoubleComplex,
                              void>::type>::type;

template<typename T> inline T* dethrust(thrust::device_ptr<T> ptr) {
  return thrust::raw_pointer_cast(ptr);
  }
template<typename T> inline thrust::device_ptr<T> enthrust(T *ptr) {
  return thrust::device_pointer_cast(ptr);
  }

template <typename T>
struct ThrustAllocatorAsync : public thrust::device_malloc_allocator<T> {
  public:
    using Base      = thrust::device_malloc_allocator<T>;
    using pointer   = typename Base::pointer;
    using size_type = typename Base::size_type;

  private:
    cudaStream_t stream;
    bool pool;

  public:
    // We need a default constructor as long as we do not fully C++ify
    // the NUFFT plan struct.
    ThrustAllocatorAsync() : stream(0), pool (false) {}
    // Prefer explicit stream; no default ctor needed if you always pass alloc to device_vector
    explicit ThrustAllocatorAsync(cudaStream_t s, bool supports_pools) : stream(s), pool (supports_pools) {}

    pointer allocate(size_type n) {
      T* p = nullptr;
      auto err = pool ? cudaMallocAsync(&p, n*sizeof(T), stream)
                      : cudaMalloc(&p, n*sizeof(T));
      if (err!=cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
      return enthrust(p);
    }
  
    void deallocate(pointer p, size_type) {
      auto err = pool ? cudaFreeAsync(dethrust(p), stream)
                      : cudaFree(dethrust(p));
      if (err!=cudaSuccess) throw int(FINUFFT_ERR_CUDA_FAILURE);
    }
};

template<typename T> using gpuArray = thrust::device_vector<T, ThrustAllocatorAsync<T>>;

template<typename T> inline T* dethrust(gpuArray<T> &arr) {
  return thrust::raw_pointer_cast(arr.data());
  }

template<typename T> struct cufinufft_plan_t {
  cufinufft_opts opts;
  finufft_spread_opts spopts;

  ThrustAllocatorAsync<int> ialloc;
  ThrustAllocatorAsync<T> alloc;
  ThrustAllocatorAsync<cuda_complex<T>> calloc;

  int type;
  int dim;
  CUFINUFFT_BIGINT M;
  cuda::std::array<CUFINUFFT_BIGINT,3> nf123;
  cuda::std::array<CUFINUFFT_BIGINT,3> mstu;
  int ntransf;
  int batchsize;
  int iflag;
  int supports_pools;

  int totalnumsubprob;
  cuda::std::array<gpuArray<T>,3> fwkerhalf;

  // for type 1,2 it is a pointer to kx, ky, kz (no new allocs), for type 3 it
  // for t3: allocated as "primed" (scaled) src pts x'_j, etc
  cuda::std::array<T *,3> kxyz;
  cuda::std::array<gpuArray<T>,3> kxyzp;
  gpuArray<cuda_complex<T>> CpBatch; // working array of prephased strengths

  // no allocs here
  cuda_complex<T> *c;
  gpuArray<cuda_complex<T>> fwp;
  cuda_complex<T> *fw;
  cuda_complex<T> *fk;

  // Type 3 specific
  struct {
    cuda::std::array<T,3> X, C, S, D, h, gam;
  } type3_params;
  int N;                        // number of NU freq pts (type 3 only)
  CUFINUFFT_BIGINT nf;
  cuda::std::array<T *,3> STU;
  cuda::std::array<gpuArray<T>,3> STUp;
  T tol;
  // inner type 2 plan for type 3
  cufinufft_plan_t<T> *t2_plan;
  // new allocs.
  // FIXME: convert to device vectors to use resize
  gpuArray<cuda_complex<T>> prephase; // pre-phase, for all input NU pts
  gpuArray<cuda_complex<T>> deconv;   // reciprocal of kernel FT, phase, all output NU pts

  // Arrays that used in subprob method
  gpuArray<int> idxnupts;        // length: #nupts, index of the nupts in the bin-sorted order
  gpuArray<int> sortidx;         // length: #nupts, order inside the bin the nupt belongs to
  gpuArray<int> numsubprob;      // length: #bins,  number of subproblems in each bin
  gpuArray<int> binsize;         // length: #bins, number of nonuniform ponits in each bin
  gpuArray<int> binstartpts;     // length: #bins, exclusive scan of array binsize
  gpuArray<int> subprob_to_bin;  // length: #subproblems, the bin the subproblem works on
  gpuArray<int> subprobstartpts; // length: #bins, exclusive scan of array numsubprob

  // Arrays for 3d (need to sort out)
  gpuArray<int> numnupts;
  gpuArray<int> subprob_to_nupts;

  cufftHandle fftplan;
  cudaStream_t stream;

  cufinufft_plan_t() : ialloc(0,false), alloc(0,false), calloc(0,false),
    fwkerhalf({gpuArray<T>{0, alloc},gpuArray<T>{0,alloc},gpuArray<T>{0,alloc}}),
    kxyzp({gpuArray<T>{0, alloc},gpuArray<T>{0,alloc},gpuArray<T>{0,alloc}}),
    CpBatch(0,calloc), fwp(0, calloc),
    STUp({gpuArray<T>{0, alloc},gpuArray<T>{0,alloc},gpuArray<T>{0,alloc}}),
    prephase(0, calloc), deconv(0, calloc), idxnupts(0, ialloc), sortidx(0, ialloc),
    numsubprob(0, ialloc), binsize(0, ialloc), binstartpts(0, ialloc), subprob_to_bin(0, ialloc),
    subprobstartpts(0, ialloc) {}
};

template<typename T> static inline constexpr cufftType_t cufft_type();
template<> inline constexpr cufftType_t cufft_type<float>() { return CUFFT_C2C; }

template<> inline constexpr cufftType_t cufft_type<double>() { return CUFFT_Z2Z; }

static inline cufftResult cufft_ex(cufftHandle plan, cufftComplex *idata,
                                   cufftComplex *odata, int direction) {
  return cufftExecC2C(plan, idata, odata, direction);
}
static inline cufftResult cufft_ex(cufftHandle plan, cufftDoubleComplex *idata,
                                   cufftDoubleComplex *odata, int direction) {
  return cufftExecZ2Z(plan, idata, odata, direction);
}

#endif
