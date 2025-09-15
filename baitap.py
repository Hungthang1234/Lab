# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:43:58 2025

@author: SinhVien
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# Đọc dữ liệu (giả sử đã có dữ liệu trong DataFrame `data`)
data = pd.read_csv('../NguyenLeHungThang-2200005517/international-airline-passengers.csv')

# Xử lý dữ liệu để có cột tháng (giả sử dữ liệu có cột 'Month' và 'Passengers')
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data.set_index('Month', inplace=True)
#kiem tra tap du lieu
print(data.head())
print(data.info())  
print(data.describe())
print(data.dtypes)
print(data.isnull().sum())

# Phân chia tập huấn luyện và tập kiểm tra
train = data.iloc[:-12]
test = data.iloc[-12:]
# Đọc dữ liệu (giả sử đã có dữ liệu trong DataFrame `data`)
data = pd.read_csv('international-airline-passengers.csv')

# Xử lý dữ liệu để có cột tháng (giả sử dữ liệu có cột 'Month' và 'Passengers')
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data.set_index('Month', inplace=True)

# Phân chia tập huấn luyện và tập kiểm tra
train = data.iloc[:-12]
test = data.iloc[-12:]
# Chuyển đổi dữ liệu thời gian thành các số tháng
train['Month_num'] = np.arange(len(train))
test['Month_num'] = np.arange(len(train), len(train) + len(test))

# Xây dựng mô hình hồi quy tuyến tính
X_train = train[['Month_num']]
y_train = train['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60']
X_test = test[['Month_num']]
y_test = test['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60']

model = LinearRegression()
model.fit(X_train, y_train)

# Dự báo
y_pred = model.predict(X_test)

# Tính MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE) on test set: {mse}')
# Thêm các biến x2 và x3 (biến số tháng theo bậc)
train['Month_num_sq'] = train['Month_num']**2
train['Month_num_cb'] = train['Month_num']**3
test['Month_num_sq'] = test['Month_num']**2
test['Month_num_cb'] = test['Month_num']**3

# Xây dựng mô hình hồi quy tuyến tính đa biến
X_train_multivariate = train[['Month_num', 'Month_num_sq', 'Month_num_cb']]
y_train_multivariate = train['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60']
X_test_multivariate = test[['Month_num', 'Month_num_sq', 'Month_num_cb']]
y_test_multivariate = test['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60']

model_multivariate = LinearRegression()
model_multivariate.fit(X_train_multivariate, y_train_multivariate)

# Dự báo
y_pred_multivariate = model_multivariate.predict(X_test_multivariate)

# Tính MSE
mse_multivariate = mean_squared_error(y_test_multivariate, y_pred_multivariate)
print(f'Mean Squared Error (MSE) on test set with multivariate regression: {mse_multivariate}')
# Ridge Regression
ridge = Ridge()
param_grid_ridge = {'alpha': np.logspace(-5, 5, 100)}
ridge_search = GridSearchCV(ridge, param_grid_ridge, cv=5)
ridge_search.fit(X_train_multivariate, y_train_multivariate)

# Lasso Regression
lasso = Lasso()
param_grid_lasso = {'alpha': np.logspace(-5, 5, 100)}
lasso_search = GridSearchCV(lasso, param_grid_lasso, cv=5)
lasso_search.fit(X_train_multivariate, y_train_multivariate)

# In kết quả tìm được
print(f"Best alpha for Ridge: {ridge_search.best_params_['alpha']}")
print(f"Best alpha for Lasso: {lasso_search.best_params_['alpha']}")

# Cây quyết định
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_multivariate, y_train_multivariate)
y_pred_dt = dt_model.predict(X_test_multivariate)
mse_dt = mean_squared_error(y_test_multivariate, y_pred_dt)

# SVR
svm_model = SVR()
svm_model.fit(X_train_multivariate, y_train_multivariate)
y_pred_svm = svm_model.predict(X_test_multivariate)
mse_svm = mean_squared_error(y_test_multivariate, y_pred_svm)

# Rừng ngẫu nhiên
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_multivariate, y_train_multivariate)
y_pred_rf = rf_model.predict(X_test_multivariate)
mse_rf = mean_squared_error(y_test_multivariate, y_pred_rf)

# In MSE
print(f'MSE of Decision Tree: {mse_dt}')
print(f'MSE of SVM: {mse_svm}')
print(f'MSE of Random Forest: {mse_rf}')
