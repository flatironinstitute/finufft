#include <array>
#include <cstdio>
#include <iostream>

namespace finufft::test {

// sphinx tag (don't remove): @CudaStatusInfo_start
// ``CudaStatusInfo_`` is used as argument to the function
// ``print_cuda_status_``.
// The following fields are used:
//
struct CudaStatusInfo_ {

  int type;            // Type of transformation being reported.
  int method;          // Gpu method being tested.
  int count_NU;        // Main count for Non Uniform points.
                       // Used for either input or output for type 2.
  int count_secondary; // Holds the uniform grid count for types 1 and 2;
                       // for type 3 (no grid) it is a second NU point count.
  float totaltime;     // Total time the tests ran for.
  float exectime;      // Total time used on execution.

  constexpr int total_points() const; // The function ``total_points`` returns the
                                      // number of points the algorithm processes.
};
// sphinx tag (don't remove): @CudaStatusInfo_end

inline constexpr int CudaStatusInfo_::total_points() const {
  return type == 3 ? (count_NU + count_secondary) : count_NU;
}

// sphinx tag (don't remove): @print_cuda_status_start

inline void print_cuda_status_(CudaStatusInfo_ status);
//
// The function ``print_cuda_status_`` prints the status for the
// ``test/cuda/cufinufft?d_test.cu`` family of fft cuda tests (1D, 2D, 3D).
// The status argument is used to parametrize the printout.
//
// The status message follows the following formatting:
//   "[Method <<method number>>] <<input count>> <<data type>> pts to <<output count>>
//   <<data type>> pts in <<time>> s: <<point/s>> NU pts/s"
//
// All the positions and types are selected according to the type being tested.
//
// sphinx tag (don't remove): @print_cuda_status_end

inline void print_cuda_status_(CudaStatusInfo_ status) {
  if (status.type < 1 || status.type > 3) {
    std::cerr << "Invalid test type (" << status.type << ").";
    return;
  }

  static const std::array<const char *, 3> format = {
      "[Method %d] %d NU pts to %d U pts in %.3g s:\t%.3g NU pts/s",
      "[Method %1$d] %3$d U pts to %2$d NU pts in %4$.3g s:\t%5$.3g NU pts/s",
      "[Method %d] %d NU pts to %d NU pts in %.3g s:\t%.3g NU pts/s"};

  ::printf(format.at(status.type - 1), status.method, status.count_NU,
           status.count_secondary, status.totaltime / 1000,
           status.total_points() / status.totaltime * 1000);

  ::printf("\t\t\t\t\t(exec-only throughput: %.3g NU pts/s)\n",
           status.total_points() / status.exectime * 1000);
}

} // namespace finufft::test
