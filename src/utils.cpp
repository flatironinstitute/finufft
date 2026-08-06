// Low-level array manipulations, quadrature, timer, and OMP helpers,
// that are precision-independent (no FLT allowed in argument lists).

// For self-test see ../test/testutils.cpp

#include <finufft/plan.hpp>
#include <finufft/utils.hpp>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <vector>
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <vector>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // Enable GNU extensions for sched_getaffinity
#endif
#include <fstream>
#include <sched.h>
#include <set>
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
#if defined(_WIN32)
// Returns the number of physical CPU cores on Windows (excluding hyper-threaded cores)
unsigned getPhysicalCoreCount() {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
  int physicalCoreCount = 0;

  // Determine the required buffer size.
  DWORD bufferSize = 0;
  if (GetLogicalProcessorInformation(nullptr, &bufferSize) == FALSE &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return physicalCoreCount;
  }

  // Calculate the number of entries and allocate a vector.
  size_t entryCount = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> procInfo(entryCount);
  if (GetLogicalProcessorInformation(procInfo.data(), &bufferSize) != FALSE) {
    for (const auto &info : procInfo) {
      if (info.Relationship == RelationProcessorCore) ++physicalCoreCount;
    }
  }

  if (physicalCoreCount == 0) {
    return MY_OMP_GET_MAX_THREADS();
  }
  return physicalCoreCount;
#else
  // On non-x86 architectures, there should be no hyper-threading
  return MY_OMP_GET_MAX_THREADS();
#endif
}

// Physical cores this process may run on: cores with at least one logical CPU in the
// affinity mask, so an SMT sibling pair counts once and the count is comparable with
// getPhysicalCoreCount(). Counting mask bits instead would count that pair twice, and the
// min() of the two then picks the SMT number, which oversubscribes.
// Only the process's own processor group is visible through GetProcessAffinityMask, so
// above 64 logical CPUs this describes one group; that limit is unchanged from before.
unsigned getAllowedCoreCount() {
  DWORD_PTR processMask, systemMask;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
    return 0; // API call failed (should rarely happen for the current process)
  }
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
  DWORD bufferSize = 0;
  if (GetLogicalProcessorInformation(nullptr, &bufferSize) == FALSE &&
      GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> procInfo(
        bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (GetLogicalProcessorInformation(procInfo.data(), &bufferSize) != FALSE) {
      unsigned cores = 0;
      for (const auto &info : procInfo) {
        if (info.Relationship == RelationProcessorCore &&
            (info.ProcessorMask & processMask))
          ++cores;
      }
      if (cores) return cores;
    }
  }
#endif
  // no topology available: fall back to counting the logical CPUs in the mask
  unsigned count = 0;
  for (DWORD_PTR m = processMask; m; m >>= 1) count += static_cast<unsigned>(m & 1U);
  return count;
}

#elif defined(__APPLE__)

// Returns the number of physical CPU cores on macOS (excluding hyper-threaded cores)
unsigned getPhysicalCoreCount() {
  int physicalCoreCount = 0;
  int cores             = 0;
  size_t size           = sizeof(cores);
  if (sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0) == 0) {
    physicalCoreCount = cores;
  }

  if (physicalCoreCount == 0) {
    return MY_OMP_GET_MAX_THREADS();
  }
  return physicalCoreCount;
}

unsigned getAllowedCoreCount() {
  // MacOS does not support CPU affinity, so we return the maximum number of threads.
  return MY_OMP_GET_MAX_THREADS();
}

#elif defined(__linux__)
// Highest CPU index the kernel knows about, from /sys/devices/system/cpu/present (e.g.
// "0-21", or "0-3,8-11" when indices are sparse - the last number is what matters). -1 if
// sysfs is unavailable. Only the bound is needed, so there is no list to parse: the
// per-CPU topology read below skips indices that do not exist.
int maxCpuIndex() {
  std::ifstream f("/sys/devices/system/cpu/present");
  std::string present;
  if (!(f >> present)) return -1;
  const auto last = present.find_last_of("-,");
  return std::atoi(present.c_str() + (last == std::string::npos ? 0 : last + 1));
}

// Which physical core a CPU belongs to, as (package, core). {-1, -1} when sysfs has no
// topology for it, which is also how absent CPU indices are skipped.
// Compatibility:
// - Linux kernels 2.6 and later (provides /sys/devices/system/cpu topology interface);
//   older kernels or architectures without the interface make countCores() return 0 and
//   the callers fall back
// - Any CPU architecture supported by the kernel (Intel, AMD, ARM, POWER, etc.)
std::pair<int, int> coreOfCpu(int cpu) {
  const std::string topo =
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
  std::ifstream pkgF(topo + "physical_package_id"), coreF(topo + "core_id");
  int pkg = -1, core = -1;
  if (pkgF >> pkg && coreF >> core) return {pkg, core};
  return {-1, -1};
}

// CPU index -> its (package, core), for every index the kernel reports; {-1, -1} for
// indices that do not exist. Read once: the two counts below differ only by which CPUs
// they look at, so re-reading sysfs per count would double the file opens at plan
// creation.
const std::vector<std::pair<int, int>> &cpuTopology() {
  static const std::vector<std::pair<int, int>> topo = [] {
    std::vector<std::pair<int, int>> t;
    const int maxCpu = maxCpuIndex();
    for (int cpu = 0; cpu <= maxCpu && cpu < CPU_SETSIZE; ++cpu)
      t.push_back(coreOfCpu(cpu));
    return t;
  }();
  return topo;
}

// How many distinct physical cores the CPUs in the mask cover (all CPUs if it is null),
// so an SMT sibling pair counts once. 0 if the topology cannot be read at all.
unsigned countCores(const cpu_set_t *mask) {
  const auto &topo = cpuTopology();
  std::set<std::pair<int, int>> cores;
  for (int cpu = 0; cpu < int(topo.size()); ++cpu) {
    if (mask && !CPU_ISSET(cpu, mask)) continue;
    if (topo[cpu].first >= 0) cores.insert(topo[cpu]);
  }
  return static_cast<unsigned>(cores.size());
}

// Physical cores on the machine, ignoring any affinity restriction.
unsigned getPhysicalCoreCount() {
  // only x86_64 and x86_32 architectures support HT (hyper-threading)
  // in all other cases, we assume no HT and return MY_OMP_GET_MAX_THREADS()
#if defined(__i386__) || defined(__x86_64__)
  if (const unsigned n = countCores(nullptr)) return n;
#endif
  // in ARM and RISKV we only need this
  return MY_OMP_GET_MAX_THREADS();
}

// Physical cores this process may actually run on. Counted in cores, not logical CPUs, so
// that it is comparable with getPhysicalCoreCount(): counting CPUs lets an SMT sibling
// pair count twice, and the min() of the two then picks the SMT number, which
// oversubscribes (taskset -c 0-11 on a 6P+8E machine asked 12 threads to share 6 cores).
// Covers taskset and cgroup/cpuset pinning, which all land in the mask. NOT cgroup CPU
// quota (cpu.max, i.e. docker --cpus=2): that throttles bandwidth without restricting the
// mask, so a quota-limited container still reports every core here.
unsigned getAllowedCoreCount() {
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  if (sched_getaffinity(0, sizeof(cpuSet), &cpuSet) != 0) {
    return 0; // Error (e.g., not supported or failed)
  }
#if defined(__i386__) || defined(__x86_64__)
  if (const unsigned n = countCores(&cpuSet)) return n;
#endif
  return static_cast<unsigned>(CPU_COUNT(&cpuSet)); // no topology: logical CPUs
}

#else

#warning "Unknown platform. Impossible to detect the number of physical cores."
// Fallback version if none of the above platforms is detected.
unsigned getPhysicalCoreCount() { return MY_OMP_GET_MAX_THREADS(); }
unsigned getAllowedCoreCount() { return MY_OMP_GET_MAX_THREADS(); }

#endif
} // anonymous namespace

unsigned getOptimalThreadCount() {
  // if the user has set the OMP_NUM_THREADS environment variable, use that value
  static const auto cached_threads = []() -> unsigned {
    const auto OMP_THREADS = std::getenv("OMP_NUM_THREADS");
    if (OMP_THREADS) {
      try {
        return std::stoi(OMP_THREADS);
      } catch (...) {
        std::cerr << "Invalid OMP_NUM_THREADS value: " << OMP_THREADS
                  << ". using default thread count." << std::endl;
      }
    }
    // otherwise, use the min between number of physical cores or the number of allowed
    // cores (e.g. by taskset). Both are counted in cores, so the min is meaningful; 0
    // means that detection failed, and must not become the thread count.
    try {
      const auto physicalCores = getPhysicalCoreCount();
      const auto allowedCores  = getAllowedCoreCount();
      if (allowedCores && allowedCores < physicalCores) return allowedCores;
      if (physicalCores) return physicalCores;
    } catch (const std::exception &e) {
      std::cerr << "Error determining optimal thread count: " << e.what()
                << ". Using OpenMP default thread count." << std::endl;
    }
    return MY_OMP_GET_MAX_THREADS();
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
