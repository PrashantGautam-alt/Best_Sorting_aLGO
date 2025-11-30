import heapq

def insertion_sort(arr):
    """
    Simple Insertion Sort.
    Efficient for small or nearly sorted arrays.
    Sorts IN-PLACE.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def quick_sort(arr):
    """
    Standard recursive QuickSort implementation.
    Uses middle element as pivot.
    Returns a NEW sorted list — so we adapt it for in-place by copying back.
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    result = quick_sort(left) + middle + quick_sort(right)


    arr[:] = result
    return arr


def merge_sort(arr):
    """
    Standard recursive MergeSort implementation.
    Returns NEW list; we convert to in-place.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = _merge(left, right)
    arr[:] = result
    return arr


def _merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def heap_sort(arr):
    """
    HeapSort using Python's heapq module.
    1. Heapify (O(N))
    2. Pop all elements (O(N log N))
    Returns NEW list — convert to in-place.
    """
    h = arr[:]
    heapq.heapify(h)
    sorted_list = [heapq.heappop(h) for _ in range(len(h))]
    arr[:] = sorted_list
    return arr


SORTERS = {
    "insertion": insertion_sort,
    "quick": quick_sort,
    "merge": merge_sort,
    "heap": heap_sort,
    "builtin": lambda arr: arr.sort(),  
}
