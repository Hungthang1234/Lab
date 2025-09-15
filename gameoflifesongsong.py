import numpy as np
import multiprocessing

def count_live_neighbors(board, i, j, N):
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    count = 0
    for di, dj in neighbors:
        ni, nj = i + di, j + dj
        if 0 <= ni < N and 0 <= nj < N:
            count += board[ni][nj]
    return count

def process_cell(i, j, board, N):
    live_neighbors = count_live_neighbors(board, i, j, N)
    if board[i][j] == 1:
        if live_neighbors < 2 or live_neighbors > 3:
            return (i, j, 0)  # Cell dies
    else:
        if live_neighbors == 3:
            return (i, j, 1)  # Cell comes to life
    return (i, j, board[i][j])

def game_of_life_parallel(board, N, k):
    for _ in range(k):
        new_board = np.copy(board)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.starmap(process_cell, [(i, j, board, N) for i in range(N) for j in range(N)])
        for i, j, new_value in results:
            new_board[i][j] = new_value
        board = np.copy(new_board)
    return board

# Example usage
N = 1000  # Size of the board
k = 100   # Number of iterations
board = np.random.randint(2, size=(N, N))  # Initial random board
board = game_of_life_parallel(board, N, k)
