import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1 = np.linspace(-10, 10, 400)
x2 = np.linspace(-100, 100, 400)

y1 = sigmoid(x1)
y2 = sigmoid(x2)

plt.figure(figsize=(8, 6))
plt.plot(x1, y1)
plt.xlabel('x')
plt.ylabel('Sigmoid(X)')
plt.title('Sơ đồ (a)')  
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x2, y2)
plt.xlabel('x')
plt.ylabel('Sigmoid(X)')
plt.title('Sơ đồ (b)')
plt.show()