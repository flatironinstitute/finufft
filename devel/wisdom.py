import numpy as np
import finufft
import timeit
import statistics
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import multiprocessing as mp

sys.stdout.reconfigure(line_buffering=True)  # Ensures auto-flushing


# Global list for collecting detailed benchmark rows
all_results = []

def benchmark_function(func, *args, runs=5, **kwargs):
    """Runs the function multiple times and returns average & std dev."""
    runtimes = [
        timeit.timeit(lambda: func(*args, **kwargs), number=1)
        for _ in range(runs)
    ]
    avg_runtime = statistics.mean(runtimes)
    stdev_runtime = statistics.stdev(runtimes) if runs > 1 else 0.0
    return avg_runtime, stdev_runtime

import numpy as np
import threading

def generate_random_data_chunk(nufft_type, nufft_sizes, num_pts, dtype, seed, x_out, c_out, f_out, d_out, start_idx):
    """Generates a chunk of NUFFT input data and writes it directly to preallocated arrays."""
    rng = np.random.Generator(np.random.SFC64(seed))
    dim = len(nufft_sizes)
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128

    # Generate nonuniform points
    x_chunk = [2 * np.pi * rng.random(num_pts, dtype=dtype) - np.pi for _ in range(dim)]

    # Generate complex coefficients correctly
    c_chunk = rng.random(num_pts, dtype=dtype) + 1j * rng.random(num_pts, dtype=dtype)  # Separate real & imag parts

    f_chunk = (np.random.standard_normal(nufft_sizes) + 1j * np.random.standard_normal(nufft_sizes)).astype(complex_dtype) \
        if nufft_type == 2 else None
    d_chunk = [2 * np.pi * rng.random(num_pts, dtype=dtype) - np.pi for _ in range(dim)] if nufft_type == 3 else None

    # Write directly into preallocated memory
    end_idx = start_idx + num_pts
    for d in range(dim):
        x_out[d][start_idx:end_idx] = x_chunk[d]

    c_out[start_idx:end_idx] = c_chunk  # Corrected complex array assignment

    if nufft_type == 2:
        f_out[start_idx:end_idx] = f_chunk

    if nufft_type == 3:
        for d in range(dim):
            d_out[d][start_idx:end_idx] = d_chunk[d]

def fast_multi_threaded_generate(nufft_type, nufft_sizes, num_pts, dtype, seed=42, num_workers=32):
    """Multi-threaded batch generation of NUFFT data with preallocated arrays."""

    dim = len(nufft_sizes)
    total_pts = num_pts * num_workers  # Total points to generate

    # Preallocate memory
    x_out = [np.empty(total_pts, dtype=dtype) for _ in range(dim)]
    c_out = np.empty(total_pts, dtype=np.complex64 if dtype == np.float32 else np.complex128)
    f_out = np.empty((total_pts, *nufft_sizes), dtype=np.complex64 if dtype == np.float32 else np.complex128) if nufft_type == 2 else None
    d_out = [np.empty(total_pts, dtype=dtype) for _ in range(dim)] if nufft_type == 3 else None

    # Create and start threads
    threads = []
    for i in range(num_workers):
        start_idx = i * num_pts
        t = threading.Thread(target=generate_random_data_chunk, args=(nufft_type, nufft_sizes, num_pts, dtype, seed + i, x_out, c_out, f_out, d_out, start_idx))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    return x_out, c_out, f_out, d_out


def run_nufft(nufft_type, nufft_sizes, epsilon, n_threads, upsampfac, x, c, f, d):
    """Runs NUFFT with the correct dtype and parameters."""
    opts = {'nthreads': n_threads, 'upsampfac': upsampfac, 'debug': 2}
    dim = len(nufft_sizes)

    if nufft_type == 1:
        if dim == 1:
            return finufft.nufft1d1(x[0], c, nufft_sizes[0], eps=epsilon, **opts)
        elif dim == 2:
            return finufft.nufft2d1(x[0], x[1], c, nufft_sizes, eps=epsilon, **opts)
        elif dim == 3:
            return finufft.nufft3d1(x[0], x[1], x[2], c, nufft_sizes, eps=epsilon, **opts)
    elif nufft_type == 2:
        if dim == 1:
            return finufft.nufft1d2(x[0], f, eps=epsilon, **opts)
        elif dim == 2:
            return finufft.nufft2d2(x[0], x[1], f, eps=epsilon, **opts)
        elif dim == 3:
            return finufft.nufft3d2(x[0], x[1], x[2], f, eps=epsilon, **opts)

    elif nufft_type == 3:
        if dim == 1:
            return finufft.nufft1d3(x[0], c, d[0], eps=epsilon, **opts)
        elif dim == 2:
            return finufft.nufft2d3(x[0], x[1], c, d[0], d[1], eps=epsilon, **opts)
        elif dim == 3:
            return finufft.nufft3d3(x[0], x[1], x[2], c, d[0], d[1], d[2], eps=epsilon, **opts)

    else:
        raise ValueError("Invalid NUFFT type. Use 1, 2, or 3.")

def benchmark_nufft_collection(nufft_type, nufft_sizes, num_pts, epsilons, n_threads, upsampfacs, dtype, runs=5,
                               description=""):
    """Runs benchmarks while measuring performance across densities and records detailed results."""
    results = {upsamp: [] for upsamp in upsampfacs}
    x, c, f, d = fast_multi_threaded_generate(nufft_type, nufft_sizes, num_pts, dtype)

    # Compute density
    size_product = np.prod(nufft_sizes)
    density = num_pts / size_product

    title = (f"NUFFT Type {nufft_type}, {len(nufft_sizes)}D, {dtype.__name__} "
             f"(Size: {'x'.join(map(str, nufft_sizes))}, Num Pts: {num_pts}, Density: {density:.2f}, Threads: {n_threads})")
    print(f"\n=== DEBUG: {title} ===")
    print(f"{'Epsilon':<10} | {'1.25s':<12} | {'2.0s':<12} | % Diff")

    for epsilon in epsilons:
        runtimes = {}
        for upsamp in upsampfacs:
            avg_runtime, _ = benchmark_function(
                run_nufft, nufft_type, nufft_sizes, epsilon, n_threads, upsamp, x, c, f, d, runs=runs
            )
            runtimes[upsamp] = avg_runtime
            results[upsamp].append(avg_runtime)

        diff = ((runtimes[2.0] - runtimes[1.25]) / runtimes[1.25]) * 100
        print(f"{epsilon:<10.1e} | {runtimes[1.25]:<12.6f} | {runtimes[2.0]:<12.6f} | {diff:+.1f}%")

        # Append a row of results for this epsilon value
        all_results.append({
            'Epsilon': epsilon,
            'Time_1.25': runtimes[1.25],
            'Time_2.0': runtimes[2.0],
            '% Diff': diff,
            'NUFFT_type': nufft_type,
            'Data_type': dtype.__name__,
            'Size': 'x'.join(map(str, nufft_sizes)),
            'Num_pts': num_pts,
            'Density': density,
            'n_threads': n_threads
        })

    plot_benchmark_results(results, epsilons, upsampfacs, title, density)
    return results

def plot_benchmark_results(results, epsilons, upsampfacs, title, density):
    """Plots performance results while including density information."""
    x_axis = np.arange(len(epsilons))
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange']

    for i, upsamp in enumerate(upsampfacs):
        ax.bar(x_axis + (i - 0.5) * width, results[upsamp], width, label=f"upsampfac = {upsamp}", color=colors[i])

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Average Runtime (s)")
    ax.set_title(f"{title} - Density {density:.2f}")
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f"{eps:.0e}" for eps in epsilons])
    ax.set_yscale("log")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{title.replace(' ', '_').replace(',', '').replace(':', '').replace('.', '_')}.png"
    plt.savefig(filename)
    plt.show()

# Define parameters for benchmarking
upsampfacs = [2.0, 1.25]
runs = 5


# Define memory size (approximately 10 GB of RAM)
memory_size_gb = .5
bytes_per_element = 8  # Assuming complex64 (8 bytes per complex number)
total_elements = (memory_size_gb * 1024**3) // bytes_per_element

# Select test dimensions for 1D, 2D, and 3D
size_1d = (int(total_elements),)
size_2d = (int(np.sqrt(total_elements)), int(np.sqrt(total_elements)))
size_3d = (int(np.cbrt(total_elements)), int(np.cbrt(total_elements)), int(np.cbrt(total_elements)))

# Define num_pts range: starts with 1/16th of volume, ends with 1024x the volume
volume_1d = np.prod(size_1d)
volume_2d = np.prod(size_2d)
volume_3d = np.prod(size_3d)

num_pts_range = lambda volume: [volume // 16 * (2**i) for i in range(8)]


test_cases = []
for nufft_type in [1, 2]:
    for n_threads in reversed([1, 16]):
        for size, desc in [(size_1d, "1D"), (size_2d, "2D"), (size_3d, "3D")]:
            for num_pts in reversed(num_pts_range(np.prod(size))):
                test_cases.append({
                    "nufft_type": nufft_type,
                    "nufft_sizes": size,
                    "num_pts": num_pts,
                    "n_threads": n_threads,
                    "description": f"NUFFT Type {nufft_type}, {desc}, Threads {n_threads}, Size {'x'.join(map(str, size))}, Num Pts {num_pts}"
                })

# Run benchmarks and generate plots for each test case and for both float32 and float64.
for case in test_cases:
    for dtype in [np.float32, np.float64]:
        epsilons = np.logspace(-1, -6, num=6) if dtype == np.float32 else np.logspace(-1, -9, num=9)

        print(f'RUNNING TEST CASE : {case["description"]} with dtype : {dtype.__name__} epsilons : {epsilons}')
        benchmark_nufft_collection(
            case["nufft_type"],
            case["nufft_sizes"],
            case["num_pts"],
            epsilons,
            case["n_threads"],
            upsampfacs,
            dtype,
            runs=runs,
            description=case["description"]
        )

# After all benchmarks are done, build and print the final results table.
df = pd.DataFrame(all_results)
# Reorder columns as desired.
df = df[['Epsilon', 'Time_1.25', 'Time_2.0', '% Diff', 'NUFFT_type', 'Data_type', 'Size', 'Num_pts', 'Density', 'n_threads']]
print("\nFinal Benchmark Results:")
print(df.to_string(index=False))
df.to_csv('wisdom.csv')
df.to_latex('wisdom.tex')
df.to_markdown('wisdom.md')
