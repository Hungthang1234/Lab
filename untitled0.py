# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 07:54:28 2025

@author: SinhVien
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([[73.5, 75.0, 76.5, 79.0, 81.5, 82.5, 84.0, 85.0, 86.5, 87.5, 89.0, 90.0, 91.5]]).T
y = np.array([[1.49, 1.50, 1.51, 1.54, 1.58, 1.59, 1.60, 1.62, 1.63, 1.64, 1.66, 1.67, 1.68]]).T

def _plot(x,y, title ="", xlabel="", ylabel = ""):
    plt.figure(figsize=(14, 8))
    plt.plot(x, y, 'r-o', label="price")
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
# mean price
    ybar = np.mean(y)
    plt.axhline(ybar, linestyle='--', linewidth=4, label="mean")
    plt.axis([x_min*0.95, x_max*1.05, y_min*0.95, y_max*1.05])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.text(x_min, ybar*1.01, "mean", fontsize=16)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.show()
    _plot(x, y,
    title='Giá nhà theo diện tích',
    xlabel='Diện tích (m2)',
    ylabel='Giá nhà (tỷ VND)')
    _plot(x, y)

# tính trung bình
xbar = np.mean(x)
ybar = np.mean(y)
x2bar = np.mean(x**2)
xybar = np.mean(x*y)
# tính w0, w1
w1 = (xbar*ybar-xybar)/(xbar**2-(x2bar))
w0 = ybar-w1*xbar
print('w1: ', w1)
print('w0: ', w0)
    