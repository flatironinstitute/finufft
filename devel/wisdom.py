import numpy as np
import finufft
import timeit
import statistics
import matplotlib.pyplot as plt
import os
import pandas as pd

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

def generate_random_data(nufft_type, nufft_sizes, num_pts, dtype):
    """Generates random NUFFT input data in the correct dtype."""
    dim = len(nufft_sizes)

    # Set the correct complex dtype
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128

    # Generate data for nonuniform points and coefficients
    x = [np.array(2 * np.pi * np.random.rand(num_pts) - np.pi, dtype=dtype) for _ in range(dim)]
    c = np.array(np.random.rand(num_pts) + 1j * np.random.rand(num_pts), dtype=complex_dtype)

    f = (np.random.standard_normal(nufft_sizes) + 1j * np.random.standard_normal(nufft_sizes)).astype(complex_dtype) \
        if nufft_type == 2 else None
    d = [np.array(2 * np.pi * np.random.rand(num_pts) - np.pi, dtype=dtype) for _ in range(dim)] \
        if nufft_type == 3 else None
    return x, c, f, d

def run_nufft(nufft_type, nufft_sizes, epsilon, n_threads, upsampfac, x, c, f, d):
    """Runs NUFFT with the correct dtype and parameters."""
    opts = {'nthreads': n_threads, 'upsampfac': upsampfac}
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
    x, c, f, d = generate_random_data(nufft_type, nufft_sizes, num_pts, dtype)

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
upsampfacs = [1.25, 2.0]
runs = 5

test_cases = []

for nufft_type in [1, 2]:
    # Test cases including 1D, 2D, and 3D variations.
    test_cases += [
        # Standard cases:
        {"nufft_type": nufft_type, "nufft_sizes": (1000000,), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, 1D"},
        {"nufft_type": nufft_type, "nufft_sizes": (1000, 1000), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, 2D"},
        {"nufft_type": nufft_type, "nufft_sizes": (100, 100, 100), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, 3D"},

        # Sizes > num_pts:
        {"nufft_type": nufft_type, "nufft_sizes": (20000000,), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, Large 1D"},
        {"nufft_type": nufft_type, "nufft_sizes": (5000, 5000), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, Large 2D"},
        {"nufft_type": nufft_type, "nufft_sizes": (500, 500, 500), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, Large 3D"},

        # Sizes < num_pts:
        {"nufft_type": nufft_type, "nufft_sizes": (1000000,), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, Small 1D"},
        {"nufft_type": nufft_type, "nufft_sizes": (50, 50), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, Small 2D"},
        {"nufft_type": nufft_type, "nufft_sizes": (20, 20, 20), "num_pts": 10000000, "n_threads": 1,
         "description": "NUFFT Type 1, Small 3D"},

        {"nufft_type": nufft_type, "nufft_sizes": (10000000,), "num_pts": 100000000, "n_threads": 8,
         "description": "NUFFT Type 1, Small 1D"},
        {"nufft_type": nufft_type, "nufft_sizes": (3162, 3162), "num_pts": 100000000, "n_threads": 8,
         "description": "NUFFT Type 1, Small 2D"},
        {"nufft_type": nufft_type, "nufft_sizes": (216, 216, 216), "num_pts": 100000000, "n_threads": 8,
         "description": "NUFFT Type 1, Small 3D"},

        {"nufft_type": nufft_type, "nufft_sizes": (10000000,), "num_pts": 100000000, "n_threads": 16,
         "description": "NUFFT Type 1, Small 1D"},
        {"nufft_type": nufft_type, "nufft_sizes": (3162, 3162), "num_pts": 100000000, "n_threads": 16,
         "description": "NUFFT Type 1, Small 2D"},
        {"nufft_type": nufft_type, "nufft_sizes": (216, 216, 216), "num_pts": 100000000, "n_threads": 16,
         "description": "NUFFT Type 1, Small 3D"},
    ]

# Run benchmarks and generate plots for each test case and for both float32 and float64.
for case in test_cases:
    for dtype in [np.float32, np.float64]:
        epsilons = np.logspace(-1, -6, num=6) if dtype == np.float32 else np.logspace(-1, -9, num=9)
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
if all_results:
    df = pd.DataFrame(all_results)
    # Reorder columns as desired.
    df = df[['Epsilon', 'Time_1.25', 'Time_2.0', '% Diff', 'NUFFT_type', 'Data_type', 'Size', 'Num_pts', 'Density', 'n_threads']]
    print("\nFinal Benchmark Results:")
    print(df.to_string(index=False))
    df.to_csv('wisdom.csv')
    df.to_latex('wisdom.tex')
    df.to_markdown('wisdom.md')
