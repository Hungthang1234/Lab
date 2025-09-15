# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:38:17 2025

@author: SinhVien
"""

from sklearn import linear_model

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# area
x1 = np.array([[73.5,75.,76.5,79.,81.5,82.5,84.,85.,86.5,87.5,89.,90.,91.5]]).T

x2 = np.array([[20, 18, 17, 16, 15, 14, 12, 10, 8, 7, 5, 2, 1]]).T

X = np.concatenate([x1, x2], axis = 1)
y =np.array([[1.49,1.50,1.51,1.54,1.58,1.59,1.60,1.62,1.63,1.64,1.66,1.67,1.68]]).T

# Khởi tạo diện tích
x1 = np.arange(90, 111, 1)
x1 = np.expand_dims(x1, axis=1)
# Khởi tạo khoảng cách tới trung tâm
x2 = np.arange(10, 31, 1)
x2 = np.expand_dims(x2, axis=1)
# Ma trận đầu vào
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=True) 
regr.fit(X, y)
# Compare two results
print( 'Coefficient : ', regr.coef_ )
print( 'Interception : ', regr.intercept_ )
ypred = regr.predict(X)
x1grid, x2grid = np.meshgrid(x1, x2)
ys = []
for i in range(len(x1)):
    x1i=x1grid[:, i:(i+1)]
    x2i=x2grid[:, i:(i+1)]
    X = np.concatenate([x1i, x2i], axis=1)
    yi = regr.predict(X)
    ys.append(yi)
ypred = np.concatenate(ys, axis=1)
fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(x1grid, x2grid, ypred,cmap=cm.spring,
                       linewidth=0, antialiased=False)
x_pos = np.arange(80.0,100.0, 5)
x_names = [str(x_tick)+ " km" for x_tick in x_pos]
plt.xticks(x_pos, x_names)
y_pos = np.arange(0.0,30.0, 5)
y_names = [str(y_tick)+ " m2" for y_tick in y_pos]
plt.yticks(y_pos, y_names)
ax.set_zlim(1.5, 2.0)
plt.xlabel('area', fontsize=18)
plt.ylabel('distance', fontsize=18)
plt.show()

