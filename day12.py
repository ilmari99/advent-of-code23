import itertools
import aocd
import os
from matplotlib import pyplot as plt
import numpy as np
import math

DAY = 12

input_file = f"day{DAY}.txt"

data = """???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""

SYM_TO_NUM = {
    ".": 0,
    "#": 1,
    "?": -1,
}


#data = aocd.get_data(day=DAY, year=2023)
if not os.path.exists(input_file):
    with open(input_file, "w") as f:
        f.write(data)
        
data_strings = data.split("\n")

def is_valid_vector(vector, blocks):
    """ Check if a vector is valid, i.e. the '#' contiguous blocks are 
    distributed exactly as specified in 'blocks'.
    """
    # Check that we have sum(blocks) of ones
    if np.where(vector == 1)[0].shape[0] != sum(blocks):
        return False
    i = 0
    for b in blocks:
        # Find the start and end index of the block of ones starting at i (inclusive)
        # And check if the difference is exactly b
        # At the last block, check that we do not have anymore ones after the end of the block
        
        start_idx = np.where(vector[i:] == 1)[0]
        if len(start_idx) == 0:
            return False
        else:
            start_idx = start_idx[0] + i
        end_idx = np.where(vector[start_idx:] == 0)[0]
        if len(end_idx) == 0:
            end_idx = len(vector)
        else:
            end_idx = end_idx[0] + start_idx
        if end_idx - start_idx != b:
            return False
        i = end_idx
    #print(f"Valid vector: {vector}")
    return True


def count_valid_vectors(vector, blocks):
    """ Count the number of valid vectors that can be formed by filling in the "?" with "." or "#",
    s.t. 'blocks' denotes all the contingent '#' blocks in the vector.
    """   
    vector = np.array([SYM_TO_NUM[c] for c in vector + "?"])
    # Copy the vector and blocks 4 times
    vector = np.tile(vector, 5)[:-1]
    blocks = np.tile(blocks, 5)
    print(vector, blocks)
    
    
    # First, we need to find the indices of the '?'s
    # Naively, every '?' can be either a '.' or a '#'
    # So, lets just try all combinations of '.' and '#'
    # and check if the resulting vector is valid.
    q_indices = np.where(vector == -1)[0]
    
    n_q = len(q_indices)
    combs = itertools.product([0, 1], repeat=n_q)
    n_valid = 0
    for comb in combs:
        vector[q_indices] = comb
        if is_valid_vector(vector, blocks):
            n_valid += 1
    return n_valid

def count_valid_vectors2(vector, blocks):
    """ Count the number of valid vectors that can be formed by filling in the "?" with "." or "#",
    s.t. 'blocks' denotes all the contingent '#' blocks in the vector.
    """   
    vector = np.array([SYM_TO_NUM[c] for c in vector + "?"])
    # Copy the vector and blocks 4 times
    vector = np.tile(vector, 5)[:-1]
    blocks = np.tile(blocks, 5)
    print(vector, blocks)
    
    
    # First, we need to find the indices of the '?'s
    # Naively, every '?' can be either a '.' or a '#'
    # So, lets just try all combinations of '.' and '#'
    # and check if the resulting vector is valid.
    q_indices = np.where(vector == -1)[0]
    
    nmissing_ones = sum(blocks) - np.sum(vector == 1)
    nmissing_zeros = len(q_indices) - nmissing_ones
    
    print(f"nmissing_ones: {nmissing_ones}")
    print(f"nmissing_zeros: {nmissing_zeros}")
    # Generate all index combinations of length nmissing_ones. Then fill in the rest with zeros.
    one_indices = map(list, itertools.combinations(q_indices, nmissing_ones))
    n_valid = 0
    for one_idx in one_indices:
        vector[q_indices] = 0
        vector[one_idx] = 1
        if is_valid_vector(vector, blocks):
            n_valid += 1
    return n_valid


def get_ones_distribution(vector):
    """ Get the distribution of ones in a vector, i.e. the lengths of the contiguous blocks of ones.
    Example:
    [1,1,0,1,-1,-1,0,1,1] -> [2,1,2]
    """
    blocks = []
    i = 0
    while i < len(vector):
        if vector[i] == 1:
            start_idx = i
            while i < len(vector) and vector[i] == 1:
                i += 1
            blocks.append(i - start_idx)
        else:
            i += 1
    return blocks
    
    
    

def count_valid_vectors_fast(vector, blocks):
    """ Count the number of valid vectors that can be formed by filling in the "?" with "." or "#",
    s.t. 'blocks' denotes all the contingent '#' blocks in the vector.
    """   
    vector = np.array([SYM_TO_NUM[c] for c in vector + "?"])
    # Copy the vector and blocks 4 times
    vector = np.tile(vector, 5)[:-1]
    blocks = np.tile(blocks, 5)
    
    print(vector, blocks)
    
    # First, we need to find the indices of the '?'s
    # Naively, every '?' can be either a '.' or a '#'
    # So, lets just try all combinations of '.' and '#'
    # and check if the resulting vector is valid.
    q_indices = np.where(vector == -1)[0]
    
    # We can reduce the number of free variables, by setting every -1 to be 0
    # if setting it to 1 would make the vector invalid.
    
    # We know that we need to have sum(blocks) of ones, so we need to add atleast sum(blocks) - curr_num_ones
    curr_num_ones = np.sum(vector == 1)
    total_needed_num_ones = sum(blocks)
    missing_ones = total_needed_num_ones - curr_num_ones
    
    # We then know that every other -1 must be 0
    num_must_be_zero = len(q_indices) - missing_ones
    
    # We can find the number of valid combinations of 0s and 1s.
    # Every missing values is eiter 0 or 1, so we can find valid combinations with a DFS
    # 1) Make the first missing value 0 and see if the vector is valid so far
    # 2) If it is valid, then put a 0 in the next missing value and see if the vector is valid so far, etc.
    # 3) If when we put a 0, the vector is invalid, then we try 1 instead.
    # 4) If it is still not valid, then we backtrack to the previous missing value and try 1 instead of 0.
    # 5) If we have already tried 1 for the previous missing value, then we backtrack again, etc.
    # 6) When we find a valid vector up to the last missing value, we increment the counter and start backtracking.
    # 7) When we have tried 0 and 1 for the first entry, we are done.
    
    def reset_index(i):
        have_tried_both_values_for_index[i] = False
        current_value_for_index[i] = -1
        vector[q_indices[i]] = -1
        i -= 1
        return i
    
    n_valid_solutions = 0
    # A map, where the key is the index of the missing value, and the value is whther we have tried both 0 and 1 for that index
    have_tried_both_values_for_index = {i : False for i in q_indices}
    # A map, where the key is the index of the missing value, and the value is the current value of the missing value
    current_value_for_index = {i : -1 for i in q_indices}
    
    curr_index = 0
    
    blocks_reached = 0
    
    while True:
        
        curr_index_in_q_indices = q_indices[curr_index]

        # Check if we have tried both values for the first missing value
        if have_tried_both_values_for_index[curr_index]:
            # We have tried both values for the first missing value, so we are done
            if curr_index == 0:
                break
            # We have tried both values for the current missing value, so we need to backtrac
            # Reset
            curr_index = reset_index(curr_index)
            continue
        
        for val in range(current_value_for_index[curr_index_in_q_indices] + 1, 2):
            # Try the next value for the current missing value
            current_value_for_index[curr_index_in_q_indices] = val
            vector[curr_index_in_q_indices] = val
            
            is_valid = is_valid_vector(vector[:curr_index_in_q_indices + 1], blocks[:blocks_reached + 1])
            
            if is_valid:
                break
            
        if is_valid:
            # We have found a valid vector up to the current missing value
            if curr_index == len(q_indices) - 1:
                # We have found a valid vector up to the last missing value
                n_valid_solutions += 1
                # Reset
                curr_index = reset_index(curr_index)
                continue
            else:
                # We have found a valid vector up to the current missing value, so we can move on to the next missing value
                curr_index += 1
                blocks_reached += 1
                last_block_end = curr_index
                continue

            
            
            
            
            
        have_tried_both_values_for_index[curr_index] = True
            
        
        
    
    


    

nvecs = 0
for line in data_strings:
    line_split = line.split(" ")
    vector = line_split[0]
    print(vector, end=" ")
    blocks = [int(x) for x in line_split[1].split(",")]
    
    print(blocks, end=" ")
    
    n = count_valid_vectors2(vector, blocks)
    
    print(f"==> {n}")
    
    nvecs += n
print(nvecs)
    