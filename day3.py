import aocd
import numpy as np
import time
import os

data = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..
"""
data = aocd.get_data(day=3)
if not os.path.exists("day3.txt"):
    with open("day3.txt", "w") as f:
        f.write(data)

start = time.time()
symbols = set(data)
symbols = set([sym for sym in symbols if sym not in [".","\n"] and not sym.isdigit()])
symbols.remove("*")
print(f"Found symbols = {symbols}")

data = data.split("\n")

# We must find all the numbers that are adjacent to a symbol (other than .)
# A number is adjacent to a symbol if:
# - The symbol is next to the number, i.e. the symbol is in the same row as the number and its index
# is either the index of the first digit of the number - 1 or the index of the last digit of the number + 1
# - OR the symbol is in the previous or next row and its index is the same as the index of any of the digits of the number, OR +1/-1

# consider the data as a matrix
# Lets convert the data to a matrix, where the numbers stay, but symbols are -1 and . are 0
matrix = np.zeros((len(data), len(data[0])), dtype=np.int32)
for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j] == "*":
            matrix[i,j] = -3
        elif data[i][j] == ".":
            matrix[i,j] = -2
        elif data[i][j] in symbols:
            matrix[i,j] = -1
        else:
            matrix[i,j] = int(data[i][j])
            
print(matrix)

# Now we find all symbols in the matrix, and see if they are adjacent to any number
symbol_indices = np.where(matrix == -3)
print(f"Symbol indices = {symbol_indices}")

def get_adjacent_indices(A, ind):
    """ Get all indices adjacent to the given index.
    """
    # Get the indices of the adjacent elements
    indices = [
        (ind[0]-1, ind[1]-1),
        (ind[0]-1, ind[1]),
        (ind[0]-1, ind[1]+1),
        (ind[0], ind[1]-1),
        (ind[0], ind[1]+1),
        (ind[0]+1, ind[1]-1),
        (ind[0]+1, ind[1]),
        (ind[0]+1, ind[1]+1),
    ]
    # Filter out indices that are outside the matrix
    indices = [ind for ind in indices if ind[0] >= 0 and ind[0] < A.shape[0] and ind[1] >= 0 and ind[1] < A.shape[1]]
    return indices

def find_connected_digits(A, ind):
    """ Find all digits that are before or after the given index, and return the indices of all of them.
    """
    if A[ind] < 0:
        return []
    # Then go backwards on the row, to find the first digit
    for j in range(ind[1], -1, -1):
        #print(f"j = {j}")
        if A[ind[0], j] < 0:
            # If the loop breaks, the last digit was at j+1
            j += 1
            break
    # If the loop didn't break, the last digit is at j = 0
    # Now we know that the first digit is at (ind[0], j)
    first_digit_index = (ind[0], j)
    
    # Find the last digit
    for j in range(ind[1], A.shape[1]):
        if A[ind[0], j] < 0:
            # If the loop breaks, the last digit was at j-1
            j -= 1
            break
    # Otherwise, the last digit is at j = A.shape[1]-1
    last_digit_index = (ind[0], j)
    return [(ind[0], j) for j in range(first_digit_index[1], last_digit_index[1]+1)]


found_numbers = []
for symbol_row_idx, symbol_col_idx in zip(symbol_indices[0], symbol_indices[1]):
    #print(f"Symbol index = {(symbol_row_idx, symbol_col_idx)}")
    # Get the indices of the adjacent elements
    adjacent_indices = get_adjacent_indices(matrix, (symbol_row_idx, symbol_col_idx))
    digits_connected_to_symbol = []
    digits_connected_to_symbol_indices = []
    # For each adjacent index we find the indices of digits that are connected to it
    for adj_row_idx, adj_col_idx in adjacent_indices:
        connected_digits = find_connected_digits(matrix, (adj_row_idx, adj_col_idx))
        if not connected_digits:
            continue
        # Now we have the indices of a number that is adjacent to the symbol
        number = int("".join([str(matrix[ind]) for ind in connected_digits]))
        #print(f"Found number = {number}")
        if connected_digits in digits_connected_to_symbol_indices:
            continue
        digits_connected_to_symbol.append(number)
        digits_connected_to_symbol_indices.append(connected_digits)
    if len(digits_connected_to_symbol) == 2:
        found_numbers.append(digits_connected_to_symbol[0] * digits_connected_to_symbol[1])
    #print(f"Matrix = \n{matrix}")

print(matrix[-1,:])
#print(f"Found numbers = {found_numbers}")
print(f"Total = {sum(found_numbers)}")
end = time.time()
print(f"Time taken = {end-start}")
#aocd.submit(sum(found_numbers), day=3, year=2020, part="1")