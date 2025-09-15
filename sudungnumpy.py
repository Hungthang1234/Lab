# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:33:05 2024

@author: ADMIN
"""

import numpy as np
import time

# Tạo mảng ngẫu nhiên với N phần tử
N = 106
data = np.random.randint(0, 1000, size=N)

# Sắp xếp tăng dần
start_time = time.time()
sorted_data = np.sort(data)  # Sắp xếp tăng dần
print(f"Sắp xếp tăng dần hoàn thành trong {time.time() - start_time:.2f} giây.")

# Sắp xếp giảm dần
sorted_desc = sorted_data[::-1]
print("Sắp xếp giảm dần hoàn tất.")
