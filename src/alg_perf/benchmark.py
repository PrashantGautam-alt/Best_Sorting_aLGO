import time
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar in CLI

from generators import GENERATORS
from features import extract_features

SIZES = [100, 500, 1000, 2500, 5000]
SEEDS = [42, 101, 999, 1234, 5678]  # 5 variants per config
PATTERNS = list(GENERATORS.keys())
RUNS_PER_ALGO = 7  

SORT_KINDS = {
    "quick": "quicksort",
    "merge": "mergesort",
    "heap": "heapsort"
}

# seeds 
DEBUG_SEEDS = {42, 101, 999}

def time_np_sort(arr: np.ndarray, kind: str, n_trials: int = RUNS_PER_ALGO) -> float:
    """Time numpy.sort on a copy of arr for n_trials and return median time (seconds)."""
    times = []
    _ = np.sort(arr, kind=kind)
    for _ in range(n_trials):
        a = arr.copy()
        t0 = time.perf_counter()
        np.sort(a, kind=kind)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))

def run_benchmark():
    results = []

    total_iterations = len(SIZES) * len(PATTERNS) * len(SEEDS)
    print(f"Starting benchmark: {total_iterations} configurations...")

    pbar = tqdm(total=total_iterations, ncols=100)

    for size in SIZES:
        for pattern_name in PATTERNS:
            gen_func = GENERATORS[pattern_name]

            for seed in SEEDS:
                # Generate List
                try:
                    arr = gen_func(size, seed)
                except TypeError:
                   
                    rng = np.random.default_rng(seed)
                    arr = gen_func(rng, size)
                arr = np.asarray(arr)

                # Extract Features
                feats = extract_features(arr)

                # Time Algorithms (using numpy sorts as robust baseline)
                timings = {}
                for alg_name, kind in SORT_KINDS.items():
                    t = time_np_sort(arr, kind=kind)
                    timings[f"time_{alg_name}"] = float(t)

                # Determine Winner (deterministic)
                # map to simple names -> pick argmin
                times_for_choice = {k: timings[f"time_{k}"] for k in SORT_KINDS.keys()}
                fastest_algo = min(times_for_choice, key=times_for_choice.get)

                # Debug: print times for some seeds so you can inspect
                if seed in DEBUG_SEEDS:
                    print(f"DEBUG seed={seed} size={size} pattern={pattern_name}")
                    print(" times:", {k: f"{v:.6f}" for k, v in times_for_choice.items()})

                # Compile Row
                row = {
                    "pattern": pattern_name,
                    "seed": seed,
                    "fastest_label": fastest_algo,
                    **feats,
                    **timings
                }
                results.append(row)
                pbar.update(1)

    pbar.close()

    # Saving to CSV
    df = pd.DataFrame(results)
    output_file = "sorting_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\nBenchmark complete. Data saved to {output_file}")
    # print a small summary
    print("\nfastest_label distribution:")
    print(df["fastest_label"].value_counts())

if __name__ == "__main__":
    run_benchmark()
