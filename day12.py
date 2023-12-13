import functools
import itertools
#import aocd
import os
import numpy as np
import multiprocessing as mp
import tqdm

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

with open(input_file, "r") as f:
    data = f.read()
        
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
    vector = np.array([SYM_TO_NUM[c] for c in vector])# + "?"])
    # Copy the vector and blocks 4 times
    #vector = np.tile(vector, 5)[:-1]
    #blocks = np.tile(blocks, 5)
    #print(vector, blocks)
    
    
    # First, we need to find the indices of the '?'s
    # Naively, every '?' can be either a '.' or a '#'
    # So, lets just try all combinations of '.' and '#'
    # and check if the resulting vector is valid.
    q_indices = np.where(vector == -1)[0]
    
    nmissing_ones = sum(blocks) - np.sum(vector == 1)
    nmissing_zeros = len(q_indices) - nmissing_ones
    
    #print(f"nmissing_ones: {nmissing_ones}")
    #print(f"nmissing_zeros: {nmissing_zeros}")
    # Generate all index combinations of length nmissing_ones. Then fill in the rest with zeros.
    one_indices = map(list, itertools.combinations(q_indices, nmissing_ones))
    n_valid = 0
    for one_idx in one_indices:
        vector[q_indices] = 0
        vector[one_idx] = 1
        if is_valid_vector(vector, blocks):
            n_valid += 1
    return n_valid
    

def create_tuples(vector):
    tuples = []
    i = 0
    while i < len(vector):
        if vector[i] == 1:
            start_idx = i
            while i < len(vector) and vector[i] == 1:
                i += 1
            tuples.append([1, i - start_idx])
        elif vector[i] == 0:
            start_idx = i
            while i < len(vector) and vector[i] == 0:
                i += 1
            tuples.append([0, i - start_idx])
        else:
            start_idx = i
            while i < len(vector) and vector[i] == -1:
                i += 1
            tuples.append([-1, i - start_idx])
    return tuples


def count_valid_vectors_fast(vector, blocks, tuples = None):
    """ Count the number of valid vectors that can be formed by filling in the "?" with "." or "#",
    and s.t. 'blocks' denotes all the contingent '#' blocks in the vector.
    In the next section I refer to 'tuples' but I mean lists of two elements.

    We do this by creating a list of tuples, where each tuple contains the symbol and the number of times it occurs.
    We then take the first tuple, and count how many different ways we can fill it such that it has exactly blocks[0] '#'s.

    If the first tuple symbol is '.', we can only fill it by setting all to values to '.' so 1 way.

    If the first tuple symbol is '#', and the count == blocks[0] we can fill it 1 way, and we must reduce the count of the next tuple by 1,
    otherwise there is no solution this path.

    If the first tuple symbol is '?':
        - If count >= blocks[0] AND the symbol of the next element is NOT '#'
        we can fill the tuple in ncr(count, blocks[0]) ways.
        - if count > blocks[0] AND the symbol of the next element IS '#'
        we can fill the tuple in ncr(count-1, blocks[0]) ways.
        - if count < blocks[0] we can only fill it by setting all to values to '.'
    
    This gives us the number of ways to fill in the first tuple.
    We then obtain all the ways to fill the other tuples, and multiply all of them together.
    """

    # Create the list of lists (tuples)
    if not isinstance(vector, np.ndarray):
        vector = np.array([SYM_TO_NUM[c] for c in vector])# + "?"])
        # Copy the vector and blocks 4 times
        #vector = np.tile(vector, 5)[:-1]
        #blocks = np.tile(blocks, 5)
    
    if tuples is None:
        tuples = create_tuples(vector)
    #print(f"Tuple list: {tuples}")
    #else:
    #    print(f"Recursively called with tuple: {tuples}")

    nprod_other_sols = []
    nprod = 1
    for i, tup in enumerate(tuples):
        #print(f"Tuple {i}: {tup}", end=" ")
        #print(f"Blocks: {blocks}")
        symbol, count = tup
        # if the symbol is '#' we can only fill it one-way, and only if the count is equal to the number of blocks
        if symbol == 1:
            if count == blocks[0]:
                nprod *= 1
                blocks = blocks[1:]
                # Reduce the count of the next tuple by 1
                # if the count of the next tuple is 1, then we change the symbol to "." and set the count to 1
                # Also only if we are not at the last tuple
                if i < len(tuples) - 1:
                    if tuples[i+1][1] == 1:
                        # Change the symbol to '.'
                        tuples[i+1][0] = 0
                        tuples[i+1][1] = 1
                    else:
                        tuples[i+1][1] -= 1
            else:
                #print(f"Can be filled in zero ways")
                return sum(nprod_other_sols)
        # if the symbol is '.' we can only fill it one-way
        elif symbol == 0:
            nprod *= 1
        # if the symbol is '?'
        elif symbol == -1:
            # If count > blocks[0] + blocks[1] we must consider,
            # That we can use only this tuple to fill both blocks[0] and blocks[1]
            # Lets split the tuple to three tuples:
            # [-1, blocks[0]], [0,1], [-1, blocks[1]]
            # and then call this function recursively
            if len(blocks) > 1 and count > blocks[0] + blocks[1]:
                print(f"Splitting tuple {tup} into {-1, blocks[0]}, [0,1], {-1, blocks[1]}")
                new_tuples = tuples.copy()
                # We can remove the current tuple, since we replace it with three new ones
                new_tuples[0] = [-1, blocks[0]]
                new_tuples.insert(1, [0,1])
                new_tuples.insert(2, [-1, blocks[1]])
                # Count the number of ways to fill the blocks,
                # when the tuple is split this way
                n_other_sols = count_valid_vectors_fast(vector, blocks, new_tuples)
                # This n_other_sols is the number of different ways to fill the blocks from here on
                # So to get the total number of ways to fill the blocks, we must multiply it with the number of ways to fill up to now
                nprod_other_sols.append(n_other_sols * nprod)
                
            next_symbol = 0 if i == len(tuples) - 1 else tuples[i+1][0]
            # If the count is greater than or equal to the number of blocks, we can fill it
            # So all the different ways to have blocks[0] consecutive ones
            if count >= blocks[0] and next_symbol != 1:
                n = count_ways_to_select_k_consecutive_spots(count, blocks[0])
                nprod *= n
                print(f"tuple {tup} can be filled in {n} ways")
                blocks = blocks[1:]
            # If the count is greater than the number of blocks, we can fill it in ncr(count-1, blocks[0]) ways
            elif count > blocks[0] and next_symbol == 1:
                n = count_ways_to_select_k_consecutive_spots(count-1, blocks[0])
                print(f"tuple {tup} can be filled in {n} ways")
                blocks = blocks[1:]
                nprod *= n
            elif count < blocks[0] and next_symbol == 1:
                # In the case that the next symbol is '#' and the count is less than the number of blocks,
                # we can combine the two blocks into one by setting the last
                # blocks[0] - next_count elements to 1 in this tuple
                # We then need to also remove the next tuple
                # We have 2 solutions
                print(f"Combining tuple {tup} with next tuple {tuples[i+1]}")
                nprod *= 2


            # If the count is less than the number of blocks and the next symbol is not '#'
            # we can only fill it by setting all to values to '.'
            else:
                nprod *= 1

        #print(f"Can be filled in {nprod} ways")
        #print(f"Blocks left: {blocks}")
        if len(blocks) == 0:
            #print(f"No more blocks left to fill")
            break
    if len(blocks) > 0:
        #print(f"Could not fill all blocks")
        return sum(nprod_other_sols)
    return nprod + sum(nprod_other_sols)


#@functools.cache
def check_is_valid_block(current_blocks, blocks):
    curr_blocks_no_zeros = [b for b in current_blocks if b != 0]
    if len(curr_blocks_no_zeros) == len(blocks) and all((b == c for b,c in zip(blocks, curr_blocks_no_zeros))):
        #print(f"Found solution with blocks: current_blocks: {current_blocks}")
        return 1
    return 0

#@functools.cache
def check_is_any_different(current_blocks, blocks):
    return any([b != c for b,c in zip(blocks[:-1], current_blocks[:-1])])


#@functools.lru_cache(maxsize=1024, typed=False)
@functools.cache
def count_valid_vectors_recursive(vector, blocks, current_blocks = tuple()):
    """ Count the number of valid vectors that can be formed by filling in the "?" with "." or "#",
    and s.t. 'blocks' denotes all the contingent '#' blocks in the vector.
    We do this recursively, by trying to substitute the first '?' with '.' and '#', and then recursively
    calling this function with the new vector.

    If a one is placed, we increment current_blocks[-1] by 1.
    If a zero is placed, we append 0 to current_blocks.

    The base cases are:
    - 0 if any value in blocks is different from the corresponding value in current_blocks up to len(current_blocks) - 2
    - 0 if len(current_blocks) > len(blocks)
    - 1 if current_blocks == blocks
    """
    #print(f"Vector: {vector}, current_blocks: {current_blocks}")
    
    
    if len(vector) == 0:
        return check_is_valid_block(current_blocks, blocks)
    
    # If any value in blocks is different from the corresponding
    # value in current_blocks excluding the last value
    if len(current_blocks) > 0 and check_is_any_different(current_blocks, blocks):
        #print(f"Cannot find solution when current_blocks: {current_blocks} and blocks: {blocks}")  
        #print(f"No solution for picked_values: {picked_values}")
        return 0
    
    
    current_blocks = list(current_blocks)
    if not current_blocks:
        current_blocks.append(0)
    
    number_of_sols = 0
    possible_placements = [0, 1] if vector[0] == -1 else [vector[0]]
    #print(f"Possible placements: {possible_placements}")
    placed_zero = False
    # Change the first '?' to '.' or '#'
    for p in possible_placements:
        # If we place a one, we need to increment the last block.
        if p == 1:
            current_blocks[-1] += 1
        
        # If we place a zero, we need to append a new block IF the last block is not zero
        elif p == 0:
            if current_blocks[-1] != 0:
                placed_zero = True
                current_blocks.append(0)

        # Find the number of solutions for the new vector and current_blocks
        number_of_sols += count_valid_vectors_recursive(vector[1:], blocks, tuple(current_blocks))

        # If we placed a one, we need to decrement the last block.
        if p == 1:
            current_blocks[-1] -= 1
        if p == 0 and placed_zero:
            current_blocks.pop()
            #placed_zero = False
    return number_of_sols

@functools.cache
def count_valid_vectors_recursive2(vector, blocks, curr_seq_len = 0):
    """ Count the number of valid vectors that can be formed by filling in the "?" with "." or "#",
    and s.t. 'blocks' denotes all the contingent '#' blocks in the vector.
    We do this recursively, by trying to substitute the first '?' with '.' and '#', reducing the blocks,
    and then recursively calling this function with the new vector and blocks.
    After trying a value, we restore the blocks to their original value.

    If a one is placed, we decrease current_blocks[0] by 1. If current_blocks[0] == 0, we pop it.
    If a zero is placed, we do not change current_blocks.

    The base cases are:
    - 0 sum(blocks) > np.sum(vector == -1) + np.sum(vector == 1)
    - 1 if len(vector) == 0 and len(blocks) == 0
    - 0 if len(vector) == 0 and len(blocks) != 0
    """
    #print(f"Vector: {vector}, blocks: {blocks}, curr_seq_len: {curr_seq_len}")
    #print(f"Current sequence length: {curr_seq_len}, chosen values: {chosen_values}")
    #print()
    
    # vector_is_empty is True if we only have 0s in the vector
    vector_is_empty = len(vector) == 0# or sum(np.abs(vector)) == 0
    vector_has_nzeros = np.sum(vector == 0)
    vector_only_has_zeros = len(vector) == vector_has_nzeros
    vector_has_nzeros_and_minus_ones = np.sum(vector == 0) + np.sum(vector == -1)
    vector_only_has_zeros_or_minus_ones = len(vector) == vector_has_nzeros_and_minus_ones
    #print(f"vector_is_empty: {vector_is_empty}, vector_only_has_zeros: {vector_only_has_zeros}, vector_only_has_zeros_or_minus_ones: {vector_only_has_zeros_or_minus_ones}")
    
    # If the vector has length 0
    if vector_is_empty:
        # We have a solution if we have no more blocks to fill, and the current sequence length is 0
        if len(blocks) == 0 and curr_seq_len == 0:
            #print(f"Found solution because vector is empty and blocks is empty and curr_seq_len is 0")
            return 1
        # We have only have one block left, and the current sequence length is equal to the block
        if len(blocks) == 1 and curr_seq_len == blocks[0]:
            #print(f"Found solution because vector is empty and blocks is 1 and curr_seq_len is blocks[0]")
            return 1
        return 0
    
    if len(blocks) > 0 and curr_seq_len > blocks[0]:
        return 0
    if len(blocks) == 0 and curr_seq_len > 0:
        return 0

    
    #print(f"Looking at vector: {vector}, blocks: {blocks}, curr_seq_len: {curr_seq_len}")#, chosen_values: {chosen_values}")

    number_of_sols = 0
    
    possible_placements = [0, 1] if vector[0] == -1 else [vector[0]]
    
    blocks = list(blocks)
    popped_zero = False
    # Change the first '?' to '.' or '#'
    for p in possible_placements:
        block_copy = blocks.copy()
        curr_seq_len_copy = curr_seq_len
        # If we place a one, we increase the sequence length which we will use when placing a zero
        if p == 1:
            curr_seq_len += 1
        
        # If we place a zero, we let the curr_seq_len do its work;
        # decrease the next block by curr_seq_len
        # If the block becomes < 0, we can not place a zero or a one here so return n_sols
        # If the block becomes 0, we can remove it
        elif p == 0 and len(blocks) > 0:
            if curr_seq_len != 0 and blocks[0] != curr_seq_len:
                #print(f"Cannot find solution when curr_seq_len: {curr_seq_len} and blocks: {blocks}")
                assert blocks == block_copy, f"Blocks changed from {block_copy} to {blocks}"
                assert curr_seq_len == curr_seq_len_copy, f"curr_seq_len changed from {curr_seq_len_copy} to {curr_seq_len}"
                continue
                #return number_of_sols
            blocks[0] -= curr_seq_len
            curr_seq_len_temp = curr_seq_len
            curr_seq_len = 0
            if blocks[0] == 0:
                popped_zero = True
                blocks.pop(0)
                
                
        # Find the number of solutions for the new vector and current_blocks
        number_of_sols += count_valid_vectors_recursive2(vector[1:], tuple(blocks), curr_seq_len)#, chosen_values + [p])
        
        # If we placed a zero, we need to restore the blocks and the sequence length
        if p == 0:
            if popped_zero:
                blocks.insert(0, 0)
                popped_zero = False
            if len(blocks) > 0:
                curr_seq_len = curr_seq_len_temp
                blocks[0] += curr_seq_len
        # If we placed a one, we merely need to restore the sequence length
        elif p == 1:
            curr_seq_len -= 1
        assert blocks == block_copy, f"Blocks changed from {block_copy} to {blocks}"
        assert curr_seq_len == curr_seq_len_copy, f"curr_seq_len changed from {curr_seq_len_copy} to {curr_seq_len}"
    return number_of_sols



def count_ways_to_select_k_consecutive_spots(n, k):
    if k > n or k <= 0:
        return 0
    if k == 1:
        return n
    return n - k + 1

sum_ = 0
for i, line in enumerate(data_strings):
    #if i != 1:
    #    continue
    line_split = line.split(" ")
    vector = line_split[0]
    vector = np.array([SYM_TO_NUM[c] for c in vector])# + "?"])
    blocks = [int(x) for x in line_split[1].split(",")]

    vector = tuple(vector)
    blocks = tuple(blocks)
    
    print(f"Vector: {vector}, blocks: {blocks}")
    n = count_valid_vectors_recursive2(vector, blocks)
    print(f"has {n} valid vectors")
    sum_ += n
print(f"Total number of valid vectors: {sum_}")

#exit()

nseqs = 20
nlines = len(data_strings)
seq_len = nlines // nseqs

# Split the data into nseqs, and then process each sequence in parallel
data_pieces = [data_strings[i*seq_len:(i+1)*seq_len] for i in range(nseqs)]

def process_data(data_piece):
    data_strings, sequence_id = data_piece
    nvecs = 0
    len_data = len(data_strings)
    for i,line in enumerate(data_strings):
        line_split = line.split(" ")
        vector = line_split[0]
        
        if i % 10 == 0:
            print(f"Line {i}/{len_data}")

        vector = np.array([SYM_TO_NUM[c] for c in vector + "?"])

        blocks = [int(x) for x in line_split[1].split(",")]

        vector = np.tile(vector, 5)[:-1]
        blocks = np.tile(blocks, 5)
        
        # From the vector, we can reduce the number of 0s drastically, by combining consecutive 0s into one
        vector = create_tuples(vector)
        vec = []
        for tup in vector:
            if tup[0] == 0:
                vec.append(0)
            else:
                vec += [tup[0]] * tup[1]
                
        vector = vec
        vector = tuple(vector)
        blocks = tuple(blocks)
        
        #print(f"Vector: {vector}", end=" ")
        n = count_valid_vectors_recursive2(vector, blocks)
        #print(f"has {n} valid vectors")
        nvecs += n
    print(f"Found {nvecs} valid vectors in sequence {sequence_id}")
    return nvecs
print(f"{[len(d) for d in data_pieces]}")
# Add the sequence id to the data_pieces
data_pieces = [(d, i) for i,d in enumerate(data_pieces)]



with mp.Pool(processes=10) as pool:
    # Save the result of each process in a list. The results are in order of finishing.
    results = pool.imap_unordered(process_data, data_pieces)
    # Sum the results
    total = sum(results)
    print(f"Total number of valid vectors: {total}")
    
    
    
    
    