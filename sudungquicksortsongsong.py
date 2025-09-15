import concurrent.futures
import random
import time

def quicksort_parallel(arr, max_depth=2):
    """
    Quicksort song song.
    arr: Mảng cần sắp xếp
    max_depth: Độ sâu tối đa để xử lý song song
    """
    if len(arr) <= 1:
        return arr

    if max_depth == 0:
        return sorted(arr)  # Xử lý tuần tự nếu đạt độ sâu tối đa

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(quicksort_parallel, left, max_depth - 1)
        right_future = executor.submit(quicksort_parallel, right, max_depth - 1)
        
        left_sorted = left_future.result()
        right_sorted = right_future.result()

    return left_sorted + middle + right_sorted

if __name__ == "__main__":
    # Tạo mảng ngẫu nhiên với N phần tử
    N = 106
    data = [random.randint(0, 1000) for _ in range(N)]

    # Sắp xếp song song
    print("Bắt đầu sắp xếp song song...")
    start_time = time.time()
    sorted_data = quicksort_parallel(data, max_depth=2)
    print(f"Sắp xếp song song hoàn thành trong {time.time() - start_time:.2f} giây.")
