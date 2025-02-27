// Low-level array manipulations, timer, and OMP helpers, that are precision-
// independent (no FLT allowed in argument lists). Others are in utils.cpp

// For self-test see ../test/testutils.cpp.      Barnett 2017-2020.

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
#include <fstream>
#include <sched.h>
#include <set>
#endif

#include <finufft/finufft_utils.hpp>

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

#if defined(_WIN32)
// Returns the number of physical CPU cores on Windows (excluding hyper-threaded cores)
static int getPhysicalCoreCount() {
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
}

static int getAllowedCoreCount() {
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
static int getPhysicalCoreCount() {
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

static int getAllowedCoreCount() {
  // MacOS does not support CPU affinity, so we return the maximum number of threads.
  return MY_OMP_GET_MAX_THREADS();
}

#elif defined(__linux__)
// Returns the number of physical CPU cores on Linux (excluding hyper-threaded cores)
static int getPhysicalCoreCount() {
  int physicalCoreCount = 0;
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo.is_open()) return MY_OMP_GET_MAX_THREADS();

  std::set<std::string> coreSet;
  std::string line;
  int physicalId = -1, coreId = -1;
  bool foundPhysical = false, foundCore = false;

  while (std::getline(cpuinfo, line)) {
    // An empty line indicates the end of a processor block.
    if (line.empty()) {
      if (foundPhysical && foundCore)
        coreSet.insert(std::to_string(physicalId) + "-" + std::to_string(coreId));
      // Reset for the next processor block.
      foundPhysical = foundCore = false;
      physicalId = coreId = -1;
    } else {
      auto colonPos = line.find(':');
      if (colonPos == std::string::npos) continue;
      std::string key   = line.substr(0, colonPos);
      std::string value = line.substr(colonPos + 1);
      // Trim whitespace.
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));

      if (key == "physical id") {
        physicalId    = std::stoi(value);
        foundPhysical = true;
      } else if (key == "core id") {
        coreId    = std::stoi(value);
        foundCore = true;
      }
    }
  }
  // In case the file doesn't end with an empty line.
  if (foundPhysical && foundCore)
    coreSet.insert(std::to_string(physicalId) + "-" + std::to_string(coreId));

  if (!coreSet.empty()) {
    physicalCoreCount = static_cast<int>(coreSet.size());
  } else {
    // Fallback: try reading "cpu cores" from the first processor block.
    cpuinfo.clear();
    cpuinfo.seekg(0, std::ios::beg);
    while (std::getline(cpuinfo, line)) {
      auto colonPos = line.find(':');
      if (colonPos != std::string::npos) {
        std::string key   = line.substr(0, colonPos);
        std::string value = line.substr(colonPos + 1);
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        if (key == "cpu cores") {
          physicalCoreCount = std::stoi(value);
          break;
        }
      }
    }
  }

  if (physicalCoreCount == 0) {
    return MY_OMP_GET_MAX_THREADS();
  }
  return physicalCoreCount;
}

static int getAllowedCoreCount() {
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
static int getPhysicalCoreCount() { return MY_OMP_GET_MAX_THREADS(); }
static int getAllowedCoreCount() { return MY_OMP_GET_MAX_THREADS(); }

#endif

int getOptimalThreadCount() {
  // if the user has set the OMP_NUM_THREADS environment variable, use that value
  const auto OMP_THREADS = std::getenv("OMP_NUM_THREADS");
  if (OMP_THREADS) {
    return std::stoi(OMP_THREADS);
  }
  // otherwise, use the min between number of physical cores or the number of allowed
  // cores (e.g. by taskset)
  return std::min(getPhysicalCoreCount(), getAllowedCoreCount());
}

// -------------------------- openmp helpers -------------------------------
int get_num_threads_parallel_block()
// return how many threads an omp parallel block would use.
// omp_get_max_threads() does not report this; consider case of NESTED=0.
// Why is there no such routine?   Barnett 5/22/20
{
  int nth_used;
#pragma omp parallel
  {
#pragma omp single
    nth_used = MY_OMP_GET_NUM_THREADS();
  }
  return nth_used;
}

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
