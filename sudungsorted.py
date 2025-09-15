# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:32:30 2024

@author: ADMIN
"""

import random
import time

# Tạo mảng ngẫu nhiên với N phần tử
N = 106  # Số phần tử
data = [random.randint(0, 1000) for _ in range(N)]

# Sắp xếp tăng dần
start_time = time.time()
sorted_data = sorted(data)  # Sắp xếp tăng dần
print(f"Sắp xếp tăng dần hoàn thành trong {time.time() - start_time:.2f} giây.")

# Sắp xếp giảm dần
start_time = time.time()
sorted_desc = sorted(data, reverse=True)
print(f"Sắp xếp giảm dần hoàn thành trong {time.time() - start_time:.2f} giây.")
