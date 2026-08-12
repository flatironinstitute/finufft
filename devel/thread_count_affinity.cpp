/* Manual check that the default thread count follows the CPU affinity mask; needs an
   externally applied mask, so it is not a CI test. opts.nthreads stays 0, so debug=1
   prints the detected count as "nthr=".

     taskset -c $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list) \
         ./build/devel/thread_count_affinity     # one core, both siblings: expect nthr=1

   Masks holding both SMT siblings of a core are the interesting ones; a mask of distinct
   cores was already right before the fix.
*/

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
  // nthr is chosen when setpts sizes the grid, so the transform itself is not needed
  std::vector<double> x(1000, 0.0);
  finufft_setpts(plan, 1000, x.data(), x.data(), x.data(), 0, nullptr, nullptr, nullptr);
  finufft_destroy(plan);
  return 0;
}
