# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:28:49 2025

@author: SinhVien
"""

from sklearn import linear_model

import matplotlib.pyplot as plt
import numpy as np
# area
x1 = np.array([[73.5,75.,76.5,79.,81.5,82.5,84.,85.,86.5,87.5,89.,90.,91.5]]).T

x2 = np.array([[20, 18, 17, 16, 15, 14, 12, 10, 8, 7, 5, 2, 1]]).T

X = np.concatenate([x1, x2], axis = 1)
y =np.array([[1.49,1.50,1.51,1.54,1.58,1.59,1.60,1.62,1.63,1.64,1.66,1.67,1.68]]).T

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

    _plot(x2, y,
          title='Giá nhà theo khoảng cách tới TT',
          xlabel='Khoảng cách tới TT (km)',
          ylabel='Giá nhà (tỷ VND)')

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=True) 
regr.fit(X, y)
# Compare two results
print( 'Coefficient : ', regr.coef_ )
print( 'Interception : ', regr.intercept_ )