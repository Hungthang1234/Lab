import numpy as np

w1 = np.random.randn()
w2 = np.random.randn()
bias = np.random.randn()

def du_doan(m1, m2):
    return (m1 * w1 + m2 * w2) + bias

m1 = np.array([3, 2, 4, 3, 3.5, 2, 5.5, 1, 4.5])
m2 = np.array([1.5, 1, 1.5, 1, .5, .5, 1, 1, 1])

y_du_doan = du_doan(m1, m2)
print("Trọng số w1:", w1)
print("Trọng số w2:", w2)
print("Bias:", bias)
print("Giá trị dự đoán y:", y_du_doan)