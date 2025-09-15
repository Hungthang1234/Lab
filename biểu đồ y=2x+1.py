import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5,5,100)
y = 2*x+1

plt.plot(x, y, '-r', label='y=2x+1')
plt.title('Biểu đồ y=2x+1')
plt.xlabel('x', color='g')
plt.ylabel('y', color='g')
plt.legend(loc='upper left')
plt.grid()
plt.show()