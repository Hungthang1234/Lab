# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:17:01 2024

@author: ADMIN
"""

import numpy as np

def game_of_life(board, N, k):
    def count_live_neighbors(board, i, j, N):
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        count = 0
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N:
                count += board[ni][nj]
        return count

    for _ in range(k):
        new_board = np.copy(board)
        for i in range(N):
            for j in range(N):
                live_neighbors = count_live_neighbors(board, i, j, N)
                if board[i][j] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_board[i][j] = 0  # Cell dies
                else:
                    if live_neighbors == 3:
                        new_board[i][j] = 1  # Cell comes to life
        board = np.copy(new_board)
    return board

# Example usage
N = 1000  # Size of the board
k = 100   # Number of iterations
board = np.random.randint(2, size=(N, N))  # Initial random board
board = game_of_life(board, N, k)
