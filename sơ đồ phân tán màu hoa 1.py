import matplotlib.pyplot as plt

data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, .5, 1],
        [2, .5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

for i in range(len(data)):
    point = data[i]
    plt.scatter(point[0], point[1])