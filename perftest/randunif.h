#pragma once

// Index i is one of the CHUNK draws of the mt19937_64 seeded for chunk
// i / CHUNK, so fill() parallelizes over chunks without changing the data:
// every invocation and either arm see the same points.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace perftest_rand {

// The arrays are filled from end to end before anything reads them, so the
// zero fill of a default vector is a second pass over up to several GB for
// nothing. construct() is left empty; resize/construct then only maps pages,
// and the parallel fill faults them in across every socket at once.
template<typename T> struct noinit_alloc : std::allocator<T> {
  using std::allocator<T>::allocator;
  template<typename U> struct rebind {
    using other = noinit_alloc<U>;
  };
  template<typename U> void construct(U *) {}
};
template<typename T> using noinit_vector = std::vector<T, noinit_alloc<T>>;

constexpr std::int64_t CHUNK = 1 << 16;

// One seed per stream, so a coordinate's and a strength's stream never align.
enum : std::uint64_t {
  X = 0xA0761D6478BD642Full,
  Y = 0xE7037ED1A0B428DBull,
  Z = 0x8EBC6AF09C88C6E3ull,
  C = 0x589965CC75374CC3ull,
  FK = 0xEB44ACCAB455D165ull,
  S = 0x9E3779B97F4A7C15ull,
  T = 0xC2B2AE3D27D4EB4Full,
  U = 0x165667B19E3779F9ull,
};

// out[i] = scale * (shift + u), u uniform over (-1,1).
template<typename T>
inline void fill(T *out, std::int64_t n, std::uint64_t stream, T scale, T shift) {
  const std::int64_t nchunks = (n + CHUNK - 1) / CHUNK;
  const auto chunk = [&](std::int64_t ci) {
    std::mt19937_64 eng(stream + static_cast<std::uint64_t>(ci));
    std::uniform_real_distribution<T> dist11(T(-1), T(1));
    const std::int64_t hi = std::min(n, (ci + 1) * CHUNK);
    for (std::int64_t i = ci * CHUNK; i < hi; ++i) out[i] = scale * (shift + dist11(eng));
  };
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
  for (std::int64_t ci = 0; ci < nchunks; ++ci) chunk(ci);
#else
  // cuperftest compiles without OpenMP; fan the chunks over std::threads
  // instead. The assignment is a fixed stride, so the thread count changes the
  // wall time and never the data.
  const unsigned nt = std::min<unsigned>(std::thread::hardware_concurrency(), nchunks);
  if (nt <= 1) {
    for (std::int64_t ci = 0; ci < nchunks; ++ci) chunk(ci);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (std::int64_t p = 0; p < nt; ++p)
      pool.emplace_back([&, p] {
        for (std::int64_t ci = p; ci < nchunks; ci += nt) chunk(ci);
      });
    for (auto &t : pool) t.join();
  }
#endif
}

} // namespace perftest_rand
