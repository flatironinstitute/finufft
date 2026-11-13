#include <algorithm>
#include <array>
#include <cstdio>

#include <thrust/complex.h>
#include <thrust/host_vector.h>

inline void print_cuda_status_(int type, int method, int count_NU, int count_secondary,
                               float totaltime, float exectime) {

  static constexpr std::array<const char *, 3> format = {
      "[Method %d] %d NU pts to %d U pts in %.3g s:\t%.3g NU pts/s",
      "[Method %1$d] %3$d U pts to %2$d NU pts in %4$.3g s:\t%5$.3g NU pts/s",
      "[Method %d] %d NU pts to %d NU pts in %.3g s:\t%.3g NU pts/s"};

  int total_points = type == 3 ? (count_NU + count_secondary) : count_NU;

  ::printf(format.at(type - 1), method, count_NU, count_secondary, totaltime / 1000,
           total_points / totaltime * 1000);

  ::printf("\t\t\t\t\t(exec-only throughput: %.3g NU pts/s)\n",
           total_points / exectime * 1000);
}

// M=0 or an empty mode set: there is no input to compare against, so the output
// must be all zero; any nonzero value is a failure.
template<typename T>
int check_empty_output_(const thrust::host_vector<thrust::complex<T>> &out) {
  const bool allzero = std::all_of(
      out.begin(), out.end(),
      [](const thrust::complex<T> &v) { return v == thrust::complex<T>(0, 0); });
  ::printf("[gpu   ] empty input or mode set: output %s all zero\n",
           allzero ? "is" : "is NOT");
  return allzero ? 0 : 1;
}
