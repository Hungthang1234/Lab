import numpy as np

# ma trận a
a1 = np.array([[0, -1, 2], 
              [4, 11, 2], 
              [3, -1, 2]])

a2 = np.array([[3, -1], 
              [1, 2], 
              [6, 1]])

# thực hiện nhân ma trận
result = np.dot(a1, a2)

print("Kết quả là:")
print(result)