import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-5, 5, 100)
y = sigmoid(x)
y_p = sigmoid_p(x)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'r-')
ax.plot(x, y_p, 'b-')
plt.show()