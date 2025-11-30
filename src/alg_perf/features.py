import numpy as np

def extract_features(arr):
    """
    Extracts numerical features from a list of integers.
    Returns a dictionary suitable for DataFrame creation.
    """
    arr_np = np.array(arr)
    n = len(arr_np)
    
    if n == 0:
        return {}

    # Basic Stats
    mean_val = np.mean(arr_np)
    std_val = np.std(arr_np)
    min_val = np.min(arr_np)
    max_val = np.max(arr_np)
    
    # Uniqueness
    unique_counts = len(np.unique(arr_np))
    fraction_unique = unique_counts / n
    
    # Sortedness 
    # Fraction of adjacent pairs that are in correct order (arr[i] <= arr[i+1])
    diffs = np.diff(arr_np)
    sorted_pairs = np.sum(diffs >= 0)
    sortedness_score = sorted_pairs / (n - 1) if n > 1 else 1.0
    
    # Number of "runs" (monotonic increasing sequences)
    # A run ends when diffs < 0
    runs_count = np.sum(diffs < 0) + 1 # +1 for the last run
    
    return {
        "length": n,
        "mean": mean_val,
        "std": std_val,
        "min_val": min_val,
        "max_val": max_val,
        "range": max_val - min_val,
        "fraction_unique": fraction_unique,
        "duplicates_ratio": 1.0 - fraction_unique,
        "sortedness_score": sortedness_score,
        "runs_count": runs_count,
        "runs_fraction": runs_count / n
    }