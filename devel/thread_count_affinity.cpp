/* Manual check that the default thread count follows the CPU affinity mask.
   Devel-only: it needs an externally applied mask, so it is not a CI test.

   Usage (Linux; 6 P-cores with SMT + 8 E-cores + 2 LP-E shown):

     cmake -S . -B build -DFINUFFT_BUILD_DEVEL=ON && cmake --build build -j
     ./build/devel/thread_count_affinity              # whole machine: nthr=16
     taskset -c 0-11 ./build/devel/thread_count_affinity   # 6 cores, SMT: expect 6
     taskset -c 0,5  ./build/devel/thread_count_affinity   # 1 core, both siblings: 1
     taskset -c 12-19 ./build/devel/thread_count_affinity  # 8 E-cores: 8

   Before the mask was honoured, `taskset -c 0-11` reported 12: physical cores of the
   whole machine were compared against logical CPUs in the mask, so an SMT pair counted
   twice and 12 threads shared 6 cores. The interesting masks are the ones holding both
   siblings of a core - a mask of distinct cores gives the right answer either way.

   opts.nthreads is left 0, so makeplan takes the detected count and prints it with
   debug=1 as "nthr=".
*/

#include <complex>
#include <cstdint>
#include <cstdio>
#include <finufft.h>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

int main() {
#ifdef __linux__
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
    printf("affinity mask: ");
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &mask)) printf("%d ", c);
    printf("(%d CPUs)\n", CPU_COUNT(&mask));
  }
#else
  printf("no affinity API on this platform; reporting the unmasked count\n");
#endif

  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.debug = 1; // prints "nthr=" - the count under test
  finufft_plan plan;
  int64_t N[3] = {32, 32, 32};
  if (finufft_makeplan(1, 3, N, 1, 1, 1e-6, &plan, &opts)) return 1;
  std::vector<double> x(1000, 0.0);
  std::vector<std::complex<double>> c(1000), F(32 * 32 * 32);
  finufft_setpts(plan, 1000, x.data(), x.data(), x.data(), 0, nullptr, nullptr, nullptr);
  finufft_execute(plan, c.data(), F.data());
  finufft_destroy(plan);
  return 0;
}
