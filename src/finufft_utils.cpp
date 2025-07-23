// Low-level array manipulations, quadrature, timer, and OMP helpers,
// that are precision-independent (no FLT allowed in argument lists).

// For self-test see ../test/testutils.cpp

#include <finufft/finufft_utils.hpp>

#include <cstdint>
#include <iostream>
#include <string>

#if defined(_WIN32)
#include <vector>
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE // Enable GNU extensions for sched_getaffinity
#endif
#include <cpuid.h>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <set>
#endif

namespace finufft::utils {

BIGINT next235even(BIGINT n)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth"). Adapted from fortran in hellskitchen.  Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
{
  if (n <= 2) return 2;
  if (n % 2 == 1) n += 1;                // even
  BIGINT nplus  = n - 2;                 // to cancel out the +=2 at start of loop
  BIGINT numdiv = 2;                     // a dummy that is >1
  while (numdiv > 1) {
    nplus += 2;                          // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0) numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0) numdiv /= 3;
    while (numdiv % 5 == 0) numdiv /= 5;
  }
  return nplus;
}

void gaussquad(int n, double *xgl, double *wgl) {
  // n-node Gauss-Legendre quadrature, adapted from a code by Jason Kaye (2022-2023),
  // from the utils file of https://github.com/flatironinstitute/cppdlr version 1.2,
  // which is Apache-2 licensed. It uses Newton iteration from Chebyshev points.
  // Double-precision only.
  // Adapted by Barnett 6/8/25 to write nodes (xgl) and weights (wgl) into arrays
  // that the user must pre-allocate to length at least n.

  double x = 0, dx = 0;
  int convcount = 0;

  // Get Gauss-Legendre nodes
  xgl[n / 2] = 0;                   // If odd number of nodes, middle node is 0
  for (int i = 0; i < n / 2; i++) { // Loop through nodes
    convcount = 0;
    x         = cos((2 * i + 1) * PI / (2 * n)); // Initial guess: Chebyshev node
    while (true) {                               // Newton iteration
      auto [p, dp] = leg_eval(n, x);
      dx           = -p / dp;
      x += dx; // Newton step
      if (std::abs(dx) < 1e-14) {
        convcount++;
      }
      if (convcount == 3) {
        break;
      } // If convergence tol hit 3 times, stop
    }
    xgl[i]         = -x;
    xgl[n - i - 1] = x; // Symmetric nodes
  }

  // Get Gauss-Legendre weights from formula
  // w_i = -2 / ((n+1)*P_n'(x_i)*P_{n+1}(x_i)) (Atkinson '89, pg. 276)
  for (int i = 0; i < n / 2 + 1; i++) {
    auto [junk1, dp] = leg_eval(n, xgl[i]);
    auto [p, junk2]  = leg_eval(n + 1, xgl[i]); // This is a bit inefficient, but who
                                                // cares...
    wgl[i]         = -2 / ((n + 1) * dp * p);
    wgl[n - i - 1] = wgl[i];
  }
}

std::tuple<double, double> leg_eval(int n, double x) {
  // return Legendre polynomial P_n(x) and its derivative P'_n(x).
  // Uses Legendre three-term recurrence.
  // Used by gaussquad above, with which it shares the same authorship and source.

  if (n == 0) {
    return {1.0, 0.0};
  }
  if (n == 1) {
    return {x, 1.0};
  }
  // Three-term recurrence and formula for derivative
  double p0 = 0.0, p1 = 1.0, p2 = x;
  for (int i = 1; i < n; i++) {
    p0 = p1;
    p1 = p2;
    p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1);
  }
  return {p2, n * (x * p2 - p1) / (x * x - 1)};
}

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
  std::uint64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  const double nowsec = double(now) * 1e-6;
  return nowsec - initial;
}

#ifdef _OPENMP
namespace {
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

unsigned getAllowedCoreCount() {
  DWORD_PTR processMask, systemMask;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
    return 0; // API call failed (should rarely happen for the current process)
  }
  // Count bits in processMask
  int count = 0;
  while (processMask) {
    count += static_cast<int>(processMask & 1U);
    processMask >>= 1;
  }
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
// Returns the number of physical CPU cores on Linux (excluding hyper-threaded cores)
// Compatibility:
// - Linux kernels 2.6 and later (provides /sys/devices/system/cpu topology interface)
// - Any CPU architecture supported by the kernel (Intel, AMD, ARM, POWER, etc.)
// - Works in containers or cgroups (reflects host topology)
unsigned getPhysicalCoreCount() {
  // only x86_64 and x86_32 architectures support HT (hyper-threading)
  // in all other cases, we assume no HT and return MY_OMP_GET_MAX_THREADS()
#if defined(__i386__) || defined(__x86_64__)
  // Parse strings like "0-3,5,7-9" → {0,1,2,3,5,7,8,9}
  auto parseCpuList = [](const std::string &s) {
    std::vector<int> cpus;
    std::istringstream iss(s);
    for (std::string tok; std::getline(iss, tok, ',');) {
      auto dash = tok.find('-');
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(tok));
      } else {
        int start = std::stoi(tok.substr(0, dash));
        int end   = std::stoi(tok.substr(dash + 1));
        for (int i = start; i <= end; ++i) cpus.push_back(i);
      }
    }
    return cpus;
  };

  // Read a single integer from the given sysfs file
  auto readInt = [&](const std::string &path, int &out) -> bool {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    f >> out;
    return !f.fail();
  };

  // 1) Read list of present CPUs
  std::ifstream presentF("/sys/devices/system/cpu/present");
  if (!presentF.is_open()) {
    return MY_OMP_GET_MAX_THREADS();
  }
  std::string presentLine;
  std::getline(presentF, presentLine);
  auto cpus = parseCpuList(presentLine);

  // 2) For each CPU, read its package & core IDs
  std::set<std::pair<int, int>> physicalCores;
  for (int cpu : cpus) {
    std::string topoBase =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    int pkg = -1, core = -1;
    if (!readInt(topoBase + "physical_package_id", pkg)) continue;
    if (!readInt(topoBase + "core_id", core)) continue;
    physicalCores.emplace(pkg, core);
  }

  // 3) Return count if successful, else fallback
  if (!physicalCores.empty()) {
    return static_cast<unsigned>(physicalCores.size());
  }
#endif
  // in ARM and RISKV we only need this
  return MY_OMP_GET_MAX_THREADS();
}

unsigned getAllowedCoreCount() {
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuSet) != 0) {
    return 0; // Error (e.g., not supported or failed)
  }
  int count = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &cpuSet)) {
      ++count;
    }
  }
  return count;
}

#else

#warning "Unknown platform. Impossible to detect the number of physical cores."
// Fallback version if none of the above platforms is detected.
unsigned getPhysicalCoreCount() { return MY_OMP_GET_MAX_THREADS(); }
unsigned getAllowedCoreCount() { return MY_OMP_GET_MAX_THREADS(); }

#endif

} // namespace

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
    // cores (e.g. by taskset)
    try {
      const auto physicalCores = getPhysicalCoreCount();
      const auto allowedCores  = getAllowedCoreCount();
      if (physicalCores < allowedCores) {
        return physicalCores;
      }
      return allowedCores;
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
