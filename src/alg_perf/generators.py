import random

def get_random_list(size, seed):
    random.seed(seed)
    return [random.randint(0, 100000) for _ in range(size)]

def get_nearly_sorted_list(size, seed):
    """Sorts list then swaps a small percentage of elements."""
    random.seed(seed)
    arr = list(range(size))
    # Swap 5 percent of elements
    swaps = max(1, int(size * 0.05))
    for _ in range(swaps):
        i, j = random.randint(0, size-1), random.randint(0, size-1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def get_reverse_sorted_list(size, seed):
    random.seed(seed)
    return list(range(size, 0, -1))

def get_few_unique_list(size, seed):
    """Many duplicates."""
    random.seed(seed)
    # Only 10 unique values
    return [random.randint(0, 10) for _ in range(size)]

def get_sorted_runs_list(size, seed):
    """Concatenation of several sorted blocks."""
    random.seed(seed)
    arr = []
    current_size = 0
    while current_size < size:
        block_size = random.randint(size // 10, size // 5)
        block_size = min(block_size, size - current_size)
        start_val = random.randint(0, 10000)
        block = sorted([random.randint(start_val, start_val + 5000) for _ in range(block_size)])
        arr.extend(block)
        current_size += block_size
    return arr

# Mapping of names to functions
GENERATORS = {
    "random": get_random_list,
    "nearly_sorted": get_nearly_sorted_list,
    "reverse_sorted": get_reverse_sorted_list,
    "few_unique": get_few_unique_list,
    "sorted_runs": get_sorted_runs_list
}