import numpy as np
from numba import jit, int8, boolean

"""
Dancing Links (Algorithm X) implementation for solving Sudoku puzzles.
"""

@jit(nopython=True)
def get_box(row, col):
    return (row // 3) * 3 + (col // 3)

@jit(nopython=True)
def is_valid_placement(grid, row_counts, col_counts, box_counts, row, col, num):
    return (row_counts[row][num] == 0 and 
            col_counts[col][num] == 0 and 
            box_counts[get_box(row,col)][num] == 0)

@jit(nopython=True)
def count_valid_numbers(grid, row_counts, col_counts, box_counts, row, col):
    if grid[row,col] != 0:
        return 9
    count = 0
    for num in range(9):
        if is_valid_placement(grid, row_counts, col_counts, box_counts, row, col, num):
            count += 1
    return count

@jit(nopython=True)
def find_most_constrained_cell(grid, row_counts, col_counts, box_counts):
    min_possibilities = 10
    best_row = -1
    best_col = -1
    
    for i in range(9):
        for j in range(9):
            if grid[i,j] == 0:
                possibilities = count_valid_numbers(grid, row_counts, col_counts, box_counts, i, j)
                if possibilities < min_possibilities:
                    min_possibilities = possibilities
                    best_row = i
                    best_col = j
    
    return best_row, best_col

@jit(nopython=True)
def place_number(grid, row_counts, col_counts, box_counts, row, col, num):
    grid[row,col] = num + 1
    row_counts[row][num] += 1
    col_counts[col][num] += 1
    box_counts[get_box(row,col)][num] += 1

@jit(nopython=True)
def remove_number(grid, row_counts, col_counts, box_counts, row, col, num):
    grid[row,col] = 0
    row_counts[row][num] -= 1
    col_counts[col][num] -= 1
    box_counts[get_box(row,col)][num] -= 1

@jit(nopython=True)
def solve_recursive(grid, row_counts, col_counts, box_counts):
    row, col = find_most_constrained_cell(grid, row_counts, col_counts, box_counts)
    if row == -1:  # No empty cells left
        return True
    
    for num in range(9):
        if is_valid_placement(grid, row_counts, col_counts, box_counts, row, col, num):
            place_number(grid, row_counts, col_counts, box_counts, row, col, num)
            if solve_recursive(grid, row_counts, col_counts, box_counts):
                return True
            remove_number(grid, row_counts, col_counts, box_counts, row, col, num)
    
    return False

def solve_sudoku(input_grid):
    # Convert input to numpy arrays with correct types
    grid = np.array(input_grid, dtype=np.int8)
    row_counts = np.zeros((9, 9), dtype=np.int8)
    col_counts = np.zeros((9, 9), dtype=np.int8)
    box_counts = np.zeros((9, 9), dtype=np.int8)
    
    # Initialize counts
    for i in range(9):
        for j in range(9):
            if grid[i,j] != 0:
                num = grid[i,j] - 1
                row_counts[i][num] += 1
                col_counts[j][num] += 1
                box_counts[get_box(i,j)][num] += 1
    
    success = solve_recursive(grid, row_counts, col_counts, box_counts)
    return success, grid