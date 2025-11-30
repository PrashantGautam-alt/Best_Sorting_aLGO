import time
import os
import random
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_bars(arr, active_indices=None):
    """
    Draws the array as ASCII bars.
    active_indices: list of indices currently being compared/swapped (highlighted).
    """
    clear_screen()
    max_val = max(arr) if arr else 1
    # Normalizing height to fit in standard terminal 
    max_height = 20
    
    # We will print horizontally i.e. one line per number
    # If array is too long, we might just print a subset or scale down.
    # For visualization, we keeping array size small.
    
    print(f"Array Size: {len(arr)}\n")
    
    for i, val in enumerate(arr):
        bar_len = int((val / max_val) * 40) # Maximum 40 chars width
        bar_str = '█' * bar_len
        
        if active_indices and i in active_indices:
            # Highlighting with a marker
            print(f"{i:2} | {bar_str} <-------- ({val})")
        else:
            print(f"{i:2} | {bar_str} ({val})")
            
    time.sleep(0.05) # Controlling animation speed

def bubble_sort_gen(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            yield arr, [j, j+1] # Yield state and active indices
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                yield arr, [j, j+1]

def quick_sort_gen(arr, low, high):
    if low < high:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            yield arr, [j, high] # Visualizing compare with pivot
            if arr[j] <= pivot:
                i = i + 1
                arr[i], arr[j] = arr[j], arr[i]
                yield arr, [i, j] # Visualizing swap
        arr[i+1], arr[high] = arr[high], arr[i+1]
        yield arr, [i+1, high] # Pivot placement
        
        pi = i + 1
        
        yield from quick_sort_gen(arr, low, pi - 1)
        yield from quick_sort_gen(arr, pi + 1, high)

def run_animation():
    print("Select Algorithm:")
    print("1. Bubble Sort")
    print("2. Quick Sort")
    choice = input("Enter choice (1 or 2): ")
    
    # Generating a random small list for visualization
    size = 20
    arr = [random.randint(5, 50) for _ in range(size)]
    
    if choice == '1':
        print("Starting Bubble Sort Animation...")
        time.sleep(1)
        sorter = bubble_sort_gen(arr)
    else:
        print("Starting Quick Sort Animation...")
        time.sleep(1)
        sorter = quick_sort_gen(arr, 0, len(arr)-1)

    try:
        for state, active in sorter:
            draw_bars(state, active)
    except KeyboardInterrupt:
        pass
    
    print("\nSorted!")

if __name__ == "__main__":
    run_animation()