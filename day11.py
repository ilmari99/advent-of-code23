import aocd
import os
from matplotlib import pyplot as plt
import numpy as np

DAY = 11
input_file = f"day{DAY}.txt"

data = """...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#....."""

SYMBOL_TO_NUM = {
    ".": 0,
    "#": 1,
}

data = aocd.get_data(day=DAY, year=2023)
if not os.path.exists(input_file):
    with open(input_file, "w") as f:
        f.write(data)
        
data_strings = data.split("\n")
data = [[SYMBOL_TO_NUM[c] for c in row] for row in data_strings]
data = np.array(data)
print(data)
print(f"Data shape: {data.shape}")
        
def expand_galaxy(data):
    """ The galaxy expands, s.t. if there is a column or row with only 0s, 
    the another 0 row/col is added to the right/bottom of the galaxy. """
    cols_to_add = []
    for i in range(data.shape[0]):
        if np.sum(data[i, :]) == 0:
            cols_to_add.append(i)
    rows_to_add = []
    for j in range(data.shape[1]):
        if np.sum(data[:, j]) == 0:
            rows_to_add.append(j)
    if len(cols_to_add) > 0:
        data = np.insert(data, cols_to_add, 0, axis=0)
    if len(rows_to_add) > 0:
        data = np.insert(data, rows_to_add, 0, axis=1)
    return data

def expand_galaxy2(data):
    """ Now, instead of adding a row/col of 0s, we only return the rows and column indices with 0s. """
    cols_to_add = []
    for i in range(data.shape[0]):
        if np.sum(data[i, :]) == 0:
            cols_to_add.append(i)
    rows_to_add = []
    for j in range(data.shape[1]):
        if np.sum(data[:, j]) == 0:
            rows_to_add.append(j)
    return rows_to_add, cols_to_add


# Okay now change every 1 to be unique. So first 1 stays 1, then 2, 3, 4, etc.

def change_ones(data):
    """ Change every 1 to be unique. So first 1 stays 1, then 2, 3, 4, etc. """
    data = data.copy()
    ones = np.where(data == 1)
    count_ones = 0
    for i, j in zip(*ones):
        count_ones += 1
        data[i, j] = count_ones
    return data

data = change_ones(data)
print(data)

# Now we need to find the distance between every pair of galaxies (1,2,3..)
# and then find the minimum distance for each galaxy.
# The output will be a symmetric matrix, where the (i,j) entry is the distance
# between galaxy i and j.

def find_shortest_path(pi, pj, zero_rows = [], zero_cols = []):
    """ The shortest path between two points in 'data'.
    pi and pj are the indices of the two points.
    The shortest path is a straight line between the galaxies.
    """
    pi = (pi[0][0], pi[1][0])
    pj = (pj[0][0], pj[1][0])
    
    
    # Count how many zero columns and rows we need to go through to get to pj
    start_row = min(pi[0], pj[0]) +1
    end_row = max(pi[0], pj[0])
    start_col = min(pi[1], pj[1]) +1
    end_col = max(pi[1], pj[1])
    
    
    col_zeros = 0
    row_zeros = 0
    # Count how many zero cols/rows we need to go through
    # We travel through columns, that are between min(start_col, end_col) and max(start_col, end_col)
    columns_to_travel = np.arange(start_col, end_col)
    # The same for rows
    rows_to_travel = np.arange(start_row, end_row)
    
    for col in columns_to_travel:
        if col in zero_cols:
            col_zeros += 1
    for row in rows_to_travel:
        if row in zero_rows:
            row_zeros += 1
    print(f"Traveling from {pi} to {pj}")
    print(f"Col zeros: {col_zeros}, row zeros: {row_zeros}")
    
    coeff = 1000000
    # Since each zero row/col is replaced coeff empty rows/cols,
    # We need to add the zeros INBETWEEN the two points.
    # So:
    # If pj[0] is smaller than pi[0], pj[0] becomes pj[0] - row_zeros * coeff
    # If pj[0] is larger than pi[0], pj[0] becomes pj[0] + row_zeros * coeff
    # If pj[1] is smaller than pi[1], pj[1] becomes pj[1] - col_zeros * coeff
    # If pj[1] is larger than pi[1], pj[1] becomes pj[1] + col_zeros * coeff
    
    if pj[0] < pi[0]:
        pj = (pj[0] - row_zeros * (coeff-1), pj[1])
    else:
        pj = (pj[0] + row_zeros * (coeff-1), pj[1])
    if pj[1] < pi[1]:
        pj = (pj[0], pj[1] - col_zeros * (coeff-1))
    else:
        pj = (pj[0], pj[1] + col_zeros * (coeff-1))
    print(f"New pj: {pj}")
    
    
    
    # Since the shortest path is a straight line between the two points,
    # The shortest path is when we move approximating the line.
    # Find the slope of the line
    slope_rows = pj[0] - pi[0]
    slope_cols = pj[1] - pi[1]
    #print(f"Slope rows: {slope_rows}, slope cols: {slope_cols}")
    gcd_slopes = np.gcd(slope_rows, slope_cols)
    slope_rows = slope_rows / gcd_slopes
    slope_cols = slope_cols / gcd_slopes
    slope_cols = int(slope_cols)
    slope_rows = int(slope_rows)

    return gcd_slopes * (abs(slope_cols) + abs(slope_rows))# + 100 * count_zeros
    

def find_distances(data):
    """ Find the distance between every pair of galaxies (1,2,3..)
    and then find the minimum distance for each galaxy.
    The output will be a symmetric matrix, where the (i,j) entry is the distance
    between galaxy i and j. """
    zero_cols, zero_rows = expand_galaxy2(data)
    print(f"Zero rows: {zero_rows}, zero cols: {zero_cols}")
    #exit()
    data = data.copy()
    distances = np.zeros((data.max(), data.max()))
    for i in range(1, data.max()+1):
        # The i'th galaxy
        galaxy_i = np.where(data == i)
        # Find every galaxy after i
        for j in range(i+1, data.max()+1):
            galaxy_j = np.where(data == j)
            min_dist = find_shortest_path(galaxy_i, galaxy_j, zero_rows, zero_cols)
            
            distances[i-1, j-1] = min_dist
            distances[j-1, i-1] = min_dist
    return distances

distances = find_distances(data)
print(distances)

# Find the shortest path != 0 on each row
# Sum the upper triangle of the matrix
sum_upper = np.sum(np.triu(distances))
print(f"Sum upper: {sum_upper}")

#aocd.submit(sum_upper, part="b", day=DAY, year=2023)

