import numpy as np
import matplotlib.pyplot as plt

N = 50
x = np.random.rand(N)
y = np.random.rand(N)

color = 'b'

plt.scatter(x, y, s=100, c=color)
plt.show()