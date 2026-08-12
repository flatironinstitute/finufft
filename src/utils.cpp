// Low-level array manipulations, quadrature, timer, and OMP helpers,
// that are precision-independent (no FLT allowed in argument lists).

// For self-test see ../test/testutils.cpp

#include <finufft/plan.hpp>
#include <finufft/utils.hpp>

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include <chrono>
#include <iostream>

#if defined(_WIN32)
#include <random> // for the rand_r shim at the bottom
#include <vector>
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // Enable GNU extensions for sched_getaffinity
#endif
#include <fstream>
#include <sched.h>
#endif

namespace finufft::utils {

// ----------------------- helpers for timing (always stay double prec) ------

void CNTime::start() {
  initial = double(std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count()) *
            1e-6;
}

double CNTime::restart()
// Barnett changed to returning in sec
{
  double delta = elapsedsec();
  start();
  return delta;
}

double CNTime::elapsedsec() const
// returns answers as double, in seconds, to microsec accuracy. Barnett 5/22/18
{
  std::uint64_t now   = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
  const double nowsec = double(now) * 1e-6;
  return nowsec - initial;
}

#ifdef _OPENMP
namespace { // helpers local to this TU

// Physical cores this process may run on (0 if unknown): one thread per core within the
// affinity mask, counting an SMT pair once. A cgroup CPU quota (docker --cpus) is not
// visible here.

#if defined(_WIN32)

unsigned getAllowedPhysicalCoreCount() {
  // both APIs see only our own processor group, so >64 logical CPUs counts within one
  // group, as before
  DWORD_PTR processMask, systemMask;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) return 0;

  // one RelationProcessorCore entry per core: count those owning an allowed CPU
  DWORD bufferSize = 0;
  if (GetLogicalProcessorInformation(nullptr, &bufferSize) == FALSE &&
      GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> procInfo(
        bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (GetLogicalProcessorInformation(procInfo.data(), &bufferSize) != FALSE) {
      unsigned cores = 0;
      for (const auto &info : procInfo)
        cores += (info.Relationship == RelationProcessorCore &&
                  (info.ProcessorMask & processMask));
      if (cores) return cores;
    }
  }
  // no topology: count logical CPUs
  unsigned count = 0;
  for (DWORD_PTR m = processMask; m; m >>= 1) count += unsigned(m & 1U);
  return count;
}

#elif defined(__APPLE__)

unsigned getAllowedPhysicalCoreCount() {
  // no affinity API on macOS: the whole machine is always allowed
  int cores = 0;
  size_t size = sizeof(cores);
  if (sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) != 0) return 0;
  return cores > 0 ? unsigned(cores) : 0; // never let a bogus value wrap
}

#elif defined(__linux__)

unsigned getAllowedPhysicalCoreCount() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return 0;

  // sysfs prints a CPU mask in ascending order, so the first sibling id names the core:
  // both SMT siblings map to the same one and the core is counted once. A CPU with no
  // such file (no sysfs topology at all) is its own core.
  // (core_id is unusable: arm64/riscv without firmware topology report -1 for every CPU,
  // which would collapse the whole machine to one core.)
  cpu_set_t cores;
  CPU_ZERO(&cores);
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) continue;
    char path[64];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    std::ifstream f(path);
    int first; // stops at the ',' or '-' of a multi-sibling list
    if (!(f >> first) || first < 0 || first >= CPU_SETSIZE) first = cpu;
    CPU_SET(first, &cores);
  }
  return unsigned(CPU_COUNT(&cores));
}

#else

#warning "Unknown platform. Impossible to detect the number of physical cores."
unsigned getAllowedPhysicalCoreCount() { return 0; }

#endif
} // anonymous namespace

unsigned getOptimalThreadCount() {
  static const auto cached_threads = []() -> unsigned {
    // an explicit OMP_NUM_THREADS wins over anything we detect
    if (const auto env = std::getenv("OMP_NUM_THREADS")) {
      char *end;
      const auto nthr = std::strtoll(env, &end, 10); // end==env if it does not parse
      // 0 and negatives are not usable: 0 aborts makeplan, a negative wraps to a huge
      // unsigned and crashes the FFT setup. Warn and detect instead. (INT_MAX, not
      // numeric_limits: windows.h makes max() a macro.)
      if (end != env && nthr > 0 && nthr <= INT_MAX) return unsigned(nthr);
      std::cerr << "Invalid OMP_NUM_THREADS value: " << env
                << ". using default thread count." << std::endl;
    }
    try {
      if (const auto cores = getAllowedPhysicalCoreCount()) return cores;
    } catch (const std::exception &e) {
      std::cerr << "Error determining optimal thread count: " << e.what()
                << ". Using OpenMP default thread count." << std::endl;
    }
    return MY_OMP_GET_MAX_THREADS(); // detection failed
  }();

  return cached_threads;
}

#endif // _OPENMP
// ---------- thread-safe rand number generator for Windows platform ---------
// (note this is used by macros in test_defs.h, and supplied in linux/macosx)
#ifdef _WIN32
int rand_r(unsigned int * /*seedp*/)
// Libin Lu, 6/18/20
{
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_int_distribution<int> distribution(0, RAND_MAX);
  return distribution(generator);
}
#endif

} // namespace finufft::utils

namespace finufft::spreadinterp {

void print_subgrid_info(int ndims, BIGINT offset1, BIGINT offset2, BIGINT offset3,
                        UBIGINT padded_size1, UBIGINT size1, UBIGINT size2, UBIGINT size3,
                        UBIGINT M0) {
  printf("size1 %" PRIu64 ", padded_size1 %" PRIu64 "\n", size1, padded_size1);
  switch (ndims) {
  case 1:
    printf("\tsubgrid: off %" PRId64 "\t siz %" PRIu64 "\t #NU %" PRIu64 "\n", offset1,
           padded_size1, M0);
    break;
  case 2:
    printf("\tsubgrid: off %" PRId64 ",%" PRId64 "\t siz %" PRIu64 ",%" PRIu64
           "\t #NU %" PRIu64 "\n",
           offset1, offset2, padded_size1, size2, M0);
    break;
  case 3:
    printf("\tsubgrid: off %" PRId64 ",%" PRId64 ",%" PRId64 "\t siz %" PRIu64 ",%" PRIu64
           ",%" PRIu64 "\t #NU %" PRIu64 "\n",
           offset1, offset2, offset3, padded_size1, size2, size3, M0);
    break;
  default:
    printf("Invalid number of dimensions: %d\n", ndims);
    break;
  }
}

int report_invalid_kernel_params(int ns, int nc) {
  fprintf(stderr,
          "FINUFFT error: invalid kernel params selected at runtime (ns=%d, nc=%d).\n",
          ns, nc);
  return 1;
}

} // namespace finufft::spreadinterp
