# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:37:12 2024

@author: ADMIN
"""

import random
import time

def quicksort(arr):
    """
    Thuật toán Quicksort tuần tự
    arr: mảng cần sắp xếp
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # Chọn phần tử giữa làm pivot
    left = [x for x in arr if x < pivot]  # Mảng con chứa các phần tử nhỏ hơn pivot
    middle = [x for x in arr if x == pivot]  # Mảng con chứa các phần tử bằng pivot
    right = [x for x in arr if x > pivot]  # Mảng con chứa các phần tử lớn hơn pivot
    return quicksort(left) + middle + quicksort(right)

if __name__ == "__main__":
    # Tạo mảng ngẫu nhiên với N phần tử
    N = 106  # Bạn có thể thay đổi N để kiểm tra với số lượng phần tử lớn hơn
    data = [random.randint(0, 1000) for _ in range(N)]

    # Sắp xếp với Quicksort tuần tự
    print("Bắt đầu sắp xếp với Quicksort...")
    start_time = time.time()
    sorted_data = quicksort(data)
    print(f"Sắp xếp hoàn thành trong {time.time() - start_time:.5f} giây.")
    
    # In kết quả (tùy chọn)
    print("Mảng sau khi sắp xếp:")
    print(sorted_data)
