#pragma once

// Cross-platform RAII wrapper for large temporary buffers.
//
// Uses mmap/VirtualAlloc for large allocations with two key features:
// 1. allocation keeps a stable virtual address range for reuse
// 2. MADV_FREE / MEM_RESET marks pages as reclaimable by the OS without
//    releasing the virtual address range, so subsequent reuse is free
//    unless the OS reclaimed the pages under memory pressure.

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#else
#include <cstring> // memset fallback
#endif

namespace finufft {

// Large-buffer allocator that returns page-aligned memory and supports
// marking pages as reclaimable between uses.
class ReclaimableMemory {
public:
  ReclaimableMemory() = default;

  // Non-copyable, movable
  ReclaimableMemory(const ReclaimableMemory &)            = delete;
  ReclaimableMemory &operator=(const ReclaimableMemory &) = delete;
  ReclaimableMemory(ReclaimableMemory &&o) noexcept : ptr_(o.ptr_), nbytes_(o.nbytes_) {
    o.ptr_    = nullptr;
    o.nbytes_ = 0;
  }
  ReclaimableMemory &operator=(ReclaimableMemory &&o) noexcept {
    if (this != &o) {
      deallocate();
      ptr_      = o.ptr_;
      nbytes_   = o.nbytes_;
      o.ptr_    = nullptr;
      o.nbytes_ = 0;
    }
    return *this;
  }

  ~ReclaimableMemory() { deallocate(); }

  // Allocate nbytes of memory while leaving physical pages to be faulted in on
  // first use. Returns true on success.
  bool allocate(size_t nbytes) {
    if (nbytes == 0) return true;
    if (ptr_ && nbytes_ == nbytes) return true; // already the right size
    deallocate();
    nbytes_ = nbytes;
#if defined(_WIN32)
    ptr_ = VirtualAlloc(nullptr, nbytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ptr_) {
      nbytes_ = 0;
      return false;
    }
#elif defined(__linux__)
    ptr_ =
        mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_    = nullptr;
      nbytes_ = 0;
      return false;
    }
#elif defined(__APPLE__) || defined(__unix__)
    // macOS and other Unix: no MAP_POPULATE, use plain mmap
    ptr_ =
        mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr_ == MAP_FAILED) {
      ptr_    = nullptr;
      nbytes_ = 0;
      return false;
    }
#else
    // Fallback: aligned allocation
    ptr_ = std::aligned_alloc(4096, ((nbytes + 4095) / 4096) * 4096);
    if (!ptr_) {
      nbytes_ = 0;
      return false;
    }
    std::memset(ptr_, 0, nbytes);
#endif
    return true;
  }

  // Mark pages as reclaimable by the OS. The virtual address range is kept,
  // and pages may remain resident if there is no memory pressure.
  // After this call, the contents are undefined until the next write.
  void mark_reclaimable() {
    if (!ptr_ || !nbytes_) return;
#if defined(_WIN32)
    // MEM_RESET tells Windows the pages are no longer needed.
    // Pages remain committed but can be discarded under pressure.
    VirtualAlloc(ptr_, nbytes_, MEM_RESET, PAGE_READWRITE);
#elif defined(__linux__) || defined(__APPLE__)
    madvise(ptr_, nbytes_, MADV_FREE);
#endif
    // Other platforms: no-op, pages stay resident
  }

  void *data() const { return ptr_; }
  size_t size() const { return nbytes_; }

private:
  void deallocate() {
    if (!ptr_) return;
#if defined(_WIN32)
    VirtualFree(ptr_, 0, MEM_RELEASE);
#elif defined(__unix__) || defined(__APPLE__)
    munmap(ptr_, nbytes_);
#else
    std::free(ptr_);
#endif
    ptr_    = nullptr;
    nbytes_ = 0;
  }

  void *ptr_     = nullptr;
  size_t nbytes_ = 0;
};

} // namespace finufft
