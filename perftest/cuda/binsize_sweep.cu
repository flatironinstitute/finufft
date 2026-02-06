// Benchmark to find optimal bin size and np for cufinufft methods
// Tests all methods, dimensions, tolerances with various bin sizes
// Designed by Marco Barbone but printing and refactoring by
// Claude Sonnet 4.5

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <cufinufft.h>
#include <cufinufft/types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

static size_t shared_memory_per_point(int dim, int ns) {
  return ns * sizeof(double) * dim  // kernel evaluations
         + sizeof(int) * dim        // indexes
         + sizeof(cuDoubleComplex); // strength
}

static size_t shared_memory_required(int dim, int ns, int bin_size_x, int bin_size_y,
                                     int bin_size_z, int np) {
  const auto shmem_per_point = shared_memory_per_point(dim, ns);
  const int ns_2             = (ns + 1) / 2;
  size_t grid_size           = bin_size_x + 2 * ns_2;
  if (dim > 1) grid_size *= bin_size_y + 2 * ns_2;
  if (dim > 2) grid_size *= bin_size_z + 2 * ns_2;
  return grid_size * sizeof(cuDoubleComplex) + shmem_per_point * np;
}

static int find_bin_size(size_t mem_size, int dim, int ns) {
  const auto elements        = mem_size / sizeof(cuDoubleComplex);
  const auto padded_bin_size = int(floor(pow(elements, 1.0 / dim)));
  const auto bin_size        = padded_bin_size - (2 * (ns + 1) / 2);
  return bin_size;
}

static int binsize_from_percent(int dim, int ns, int shmem_limit, int percent, int np) {
  const auto shmem_per_point = shared_memory_per_point(dim, ns);
  const auto target_shmem    = (shmem_limit * percent) / 100;
  const auto available       = int(target_shmem - shmem_per_point * np);
  if (available <= 0) return 0;
  return find_bin_size(static_cast<size_t>(available), dim, ns);
}

struct BenchResult {
  int method;
  int dim;
  int type;
  double tol;
  int ns;
  int binsize_x, binsize_y, binsize_z;
  int np;
  int shmem_used;
  int shmem_limit;
  int shmem_percent;
  int shmem_target_percent;
  double exec_time_ms;
  double throughput;
  bool valid;
  string error_msg;
};

BenchResult run_benchmark(int dim, int type, int method, double tol, int64_t N, int64_t M,
                          int binsize_override, int np_override, int shmem_target_percent,
                          vector<BenchResult> &results) {

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_method = method;

  // Override bin sizes if requested
  if (binsize_override > 0) {
    opts.gpu_binsizex = binsize_override;
    opts.gpu_binsizey = binsize_override;
    opts.gpu_binsizez = binsize_override;
  }

  // Override np for method 3
  if (method == 3 && np_override > 0) {
    opts.gpu_np = np_override;
  }

  // Allocate host and device arrays
  int64_t N_total = (dim == 1) ? N : (dim == 2) ? N * N : N * N * N;

  thrust::host_vector<double> h_x(M), h_y(M), h_z(M);
  thrust::host_vector<cuDoubleComplex> h_c(M), h_fk(N_total);

  // Initialize random points
  for (int64_t i = 0; i < M; i++) {
    h_x[i]   = M_PI * (2.0 * rand() / RAND_MAX - 1.0);
    h_y[i]   = M_PI * (2.0 * rand() / RAND_MAX - 1.0);
    h_z[i]   = M_PI * (2.0 * rand() / RAND_MAX - 1.0);
    h_c[i].x = 2.0 * rand() / RAND_MAX - 1.0;
    h_c[i].y = 2.0 * rand() / RAND_MAX - 1.0;
  }

  // Copy to device
  thrust::device_vector<double> d_x          = h_x;
  thrust::device_vector<double> d_y          = h_y;
  thrust::device_vector<double> d_z          = h_z;
  thrust::device_vector<cuDoubleComplex> d_c = h_c;
  thrust::device_vector<cuDoubleComplex> d_fk(N_total);

  // Create plan
  cufinufft_plan plan;
  int ier = 0;

  int64_t nmodes[3] = {N, (dim >= 2) ? N : 1, (dim >= 3) ? N : 1};

  ier = cufinufft_makeplan(type, dim, nmodes, 1, 1, tol, &plan, &opts);

  BenchResult result          = {};
  result.method               = method;
  result.dim                  = dim;
  result.type                 = type;
  result.tol                  = tol;
  result.shmem_target_percent = shmem_target_percent;
  cudaDeviceGetAttribute(&result.shmem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         opts.gpu_device_id);

  if (ier != 0) {
    result.method    = method;
    result.dim       = dim;
    result.tol       = tol;
    result.valid     = false;
    result.error_msg = "makeplan failed: " + to_string(ier);
    results.push_back(result);
    return result;
  }

  // Cast to internal type to access fields
  auto *plan_internal = reinterpret_cast<cufinufft_plan_t<double> *>(plan);

  // Set points (pass device pointers)
  ier = cufinufft_setpts(
      plan, M, thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_z.data()), 0, nullptr, nullptr, nullptr);

  if (ier != 0) {
    result.ns        = plan_internal->spopts.nspread;
    result.binsize_x = plan_internal->opts.gpu_binsizex;
    result.binsize_y = plan_internal->opts.gpu_binsizey;
    result.binsize_z = plan_internal->opts.gpu_binsizez;
    result.np        = plan_internal->opts.gpu_np;
    result.valid     = false;
    result.error_msg = "setpts failed: " + to_string(ier);
    results.push_back(result);
    cufinufft_destroy(plan);
    return result;
  }

  // Warmup
  ier = cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                          thrust::raw_pointer_cast(d_fk.data()));
  if (ier != 0) {
    result.ns        = plan_internal->spopts.nspread;
    result.binsize_x = plan_internal->opts.gpu_binsizex;
    result.binsize_y = plan_internal->opts.gpu_binsizey;
    result.binsize_z = plan_internal->opts.gpu_binsizez;
    result.np        = plan_internal->opts.gpu_np;
    result.valid     = false;
    result.error_msg = "execute failed: " + to_string(ier);
    results.push_back(result);
    cufinufft_destroy(plan);
    return result;
  }

  // Benchmark (10 runs)
  const int n_runs = 10;
  cudaDeviceSynchronize();
  auto start = chrono::high_resolution_clock::now();

  for (int i = 0; i < n_runs; i++) {
    cufinufft_execute(plan, thrust::raw_pointer_cast(d_c.data()),
                      thrust::raw_pointer_cast(d_fk.data()));
  }

  cudaDeviceSynchronize();
  auto end          = chrono::high_resolution_clock::now();
  double elapsed_ms = chrono::duration<double, milli>(end - start).count() / n_runs;
  double throughput = M / (elapsed_ms / 1000.0);

  // Record result
  result.ns           = plan_internal->spopts.nspread;
  result.binsize_x    = plan_internal->opts.gpu_binsizex;
  result.binsize_y    = plan_internal->opts.gpu_binsizey;
  result.binsize_z    = plan_internal->opts.gpu_binsizez;
  result.np           = plan_internal->opts.gpu_np;
  result.exec_time_ms = elapsed_ms;
  result.throughput   = throughput;
  result.valid        = true;

  cudaDeviceGetAttribute(&result.shmem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         plan_internal->opts.gpu_device_id);
  result.shmem_used = static_cast<int>(shared_memory_required(
      dim, result.ns, result.binsize_x, result.binsize_y, result.binsize_z, result.np));
  result.shmem_percent =
      result.shmem_limit > 0 ? int(100.0 * result.shmem_used / result.shmem_limit) : 0;

  results.push_back(result);

  // Cleanup
  cufinufft_destroy(plan);
  // Device vectors automatically freed
  return result;
}

void print_csv(const vector<BenchResult> &results) {
  cout << "\n========== CSV OUTPUT ==========\n" << endl;
  cout << "method,dim,tol,ns,binsize_x,binsize_y,binsize_z,np,shmem_used,shmem_limit,"
       << "shmem_used_pct,shmem_target_pct,exec_ms,throughput,valid,error" << endl;

  for (const auto &r : results) {
    if (!r.valid) {
      cout << r.method << "," << r.dim << "," << scientific << r.tol << "," << r.ns << ","
           << r.binsize_x << "," << r.binsize_y << "," << r.binsize_z << "," << r.np
           << ",0,0,0," << r.shmem_target_percent << ",0,0,false," << r.error_msg << endl;
    } else {
      cout << r.method << "," << r.dim << "," << scientific << r.tol << "," << r.ns << ","
           << r.binsize_x << "," << r.binsize_y << "," << r.binsize_z << "," << r.np
           << "," << r.shmem_used << "," << r.shmem_limit << "," << r.shmem_percent << ","
           << r.shmem_target_percent << "," << fixed << setprecision(3) << r.exec_time_ms
           << "," << scientific << r.throughput << ",true," << endl;
    }
  }
}

void print_summary_by_method_dim(const vector<BenchResult> &results) {
  cout << "\n========== PERFORMANCE BY METHOD & DIMENSION ==========\n" << endl;

  for (int method = 1; method <= 3; method++) {
    for (int dim = 1; dim <= 3; dim++) {
      // Get results for this method/dim
      vector<BenchResult> group;
      for (const auto &r : results) {
        if (r.valid && r.method == method && r.dim == dim) {
          group.push_back(r);
        }
      }

      if (group.empty()) continue;

      cout << "\nMethod " << method << ", " << dim << "D:" << endl;
      cout << string(80, '-') << endl;
      cout << setw(6) << "Tol" << setw(4) << "ns" << setw(10) << "BinSize" << setw(5)
           << "np" << setw(8) << "Shmem" << setw(12) << "Time(ms)" << setw(14)
           << "Throughput" << setw(10) << "Speedup" << endl;
      cout << string(80, '-') << endl;

      // Find baseline (first entry with natural bin size for each tolerance)
      map<int, double> baseline_times; // ns -> baseline_time
      for (const auto &r : group) {
        if (baseline_times.find(r.ns) == baseline_times.end()) {
          baseline_times[r.ns] = r.exec_time_ms;
        }
      }

      for (const auto &r : group) {
        string binsize = to_string(r.binsize_x);
        if (dim > 1) binsize += "x" + to_string(r.binsize_y);
        if (dim > 2) binsize += "x" + to_string(r.binsize_z);

        double baseline = baseline_times[r.ns];
        double speedup  = baseline / r.exec_time_ms;

        cout << scientific << setprecision(0) << setw(6) << r.tol << fixed << setw(4)
             << r.ns << setw(10) << binsize << setw(5) << r.np << setw(7)
             << r.shmem_percent << "%" << setw(12) << fixed << setprecision(2)
             << r.exec_time_ms << setw(14) << scientific << setprecision(2)
             << r.throughput << setw(9) << fixed << setprecision(3) << speedup << "x";

        if (speedup > 1.05) cout << " ★";
        if (speedup < 0.95) cout << " ⚠";
        cout << endl;
      }
    }
  }
}

void print_best_shmem_percent(const vector<BenchResult> &results) {
  cout << "\n========== BEST SHMEM PERCENT BY METHOD/DIM/TOL ==========\n" << endl;

  for (int method = 1; method <= 3; method++) {
    for (int dim = 1; dim <= 3; dim++) {
      map<double, BenchResult> best_by_tol;
      for (const auto &r : results) {
        if (!r.valid || r.method != method || r.dim != dim) continue;
        auto it = best_by_tol.find(r.tol);
        if (it == best_by_tol.end() || r.throughput > it->second.throughput) {
          best_by_tol[r.tol] = r;
        }
      }

      if (best_by_tol.empty()) continue;

      cout << "\nMethod " << method << ", " << dim << "D:" << endl;
      cout << string(70, '-') << endl;
      cout << setw(6) << "Tol" << setw(4) << "ns" << setw(10) << "BinSize" << setw(5)
           << "np" << setw(8) << "Shmem" << setw(12) << "Time(ms)" << setw(14)
           << "Throughput" << endl;
      cout << string(70, '-') << endl;

      for (const auto &[tol, r] : best_by_tol) {
        string binsize = to_string(r.binsize_x);
        if (dim > 1) binsize += "x" + to_string(r.binsize_y);
        if (dim > 2) binsize += "x" + to_string(r.binsize_z);

        cout << scientific << setprecision(0) << setw(6) << tol << fixed << setw(4)
             << r.ns << setw(10) << binsize << setw(5) << r.np << setw(7)
             << r.shmem_percent << "%" << setw(12) << fixed << setprecision(2)
             << r.exec_time_ms << setw(14) << scientific << setprecision(2)
             << r.throughput << endl;
      }
    }
  }
}

int main(int argc, char **argv) {
  cout << "========================================" << endl;
  cout << "  CUFINUFFT BIN SIZE PERFORMANCE SWEEP  " << endl;
  cout << "========================================\n" << endl;

  srand(42); // Reproducible
  vector<BenchResult> results;

  int type = 1; // Type 1 transform

  // Quick debug mode: test only one config
  bool debug_mode = (argc > 1 && string(argv[1]) == "--debug");

  vector<double> tolerances =
      debug_mode ? vector<double>{1e-6} : vector<double>{1e-3, 1e-6, 1e-9, 1e-12};

  vector<tuple<int, int64_t, int64_t>> configs =
      debug_mode ? vector<tuple<int, int64_t, int64_t>>{{3, 128, 100000}} : // Quick 3D
                                                                            // test
          vector<tuple<int, int64_t, int64_t>>{
              {1, 1 << 20, 1e7}, {2, 2048, 1e7}, {3, 256, 1e7}};

  for (auto [dim, N, M] : configs) {
    cout << "\n" << dim << "D case (";
    if (dim == 1)
      cout << N;
    else if (dim == 2)
      cout << N << "x" << N;
    else
      cout << N << "x" << N << "x" << N;
    cout << " modes, " << M << " points)" << endl;
    cout << string(60, '=') << endl;

    for (double tol : tolerances) {
      cout << "\nTolerance " << scientific << tol << ":" << endl;

      for (int method : {1, 2, 3}) {
        cout << "  Method " << method << ": natural";
        BenchResult baseline =
            run_benchmark(dim, type, method, tol, N, M, 0, 0, 0, results);
        if (!baseline.valid) {
          cout << " (failed) ✓" << endl;
          continue;
        }

        const int ns                     = baseline.ns;
        const int shmem_limit            = baseline.shmem_limit;
        const vector<int> shmem_percents = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

        if (method != 3) {
          for (int percent : shmem_percents) {
            int binsize = binsize_from_percent(dim, ns, shmem_limit, percent, 0);
            if (binsize < 1) continue;
            cout << " " << percent << "%";
            run_benchmark(dim, type, method, tol, N, M, binsize, 0, percent, results);
          }
        } else {
          const auto shmem_per_point = shared_memory_per_point(dim, ns);
          const int max_np           = (shmem_limit / shmem_per_point) & ~15;
          const int np_step          = max(16, (max_np / 16) & ~15);
          vector<int> np_candidates;
          for (int np = 16; np <= max_np; np += np_step) {
            np_candidates.push_back(np & ~15);
          }
          if (!np_candidates.empty() && np_candidates.back() != max_np && max_np >= 16) {
            np_candidates.push_back(max_np);
          }
          np_candidates.erase(unique(np_candidates.begin(), np_candidates.end()),
                              np_candidates.end());

          for (int percent : shmem_percents) {
            cout << " [p=" << percent << "%";
            for (int np : np_candidates) {
              int binsize = binsize_from_percent(dim, ns, shmem_limit, percent, np);
              if (binsize < 1) continue;
              cout << " np=" << np;
              run_benchmark(dim, type, method, tol, N, M, binsize, np, percent, results);
            }
            cout << "]";
          }
        }

        cout << " ✓" << endl;
      }
    }
  }

  // Print results
  print_summary_by_method_dim(results);
  print_best_shmem_percent(results);
  print_csv(results);

  cout << "\n========================================" << endl;
  cout << "LEGEND:" << endl;
  cout << "  ★ = >5% faster than baseline" << endl;
  cout << "  ⚠ = >5% slower than baseline" << endl;
  cout << "\nRECOMMENDATION:" << endl;
  cout << "  Analyze speedup column to find optimal bin sizes." << endl;
  cout << "  CSV output above can be imported for further analysis." << endl;
  cout << "========================================\n" << endl;

  return 0;
}
