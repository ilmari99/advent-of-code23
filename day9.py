import aocd
import os
import numpy as np
import time
DAY = 9
input_file = f"day{DAY}.txt"

data = """0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45"""

data = aocd.get_data(day=DAY, year=2023)
if not os.path.exists(input_file):
    with open(input_file, "w") as f:
        f.write(data)

def extrapolate_sequence(seq):
    """
    Given a sequence of numbers, extrapolate the next number in the sequence by
    
    Taking the difference between each number until we only have a sequence of zeros left (do this K times).
    Then to extrapolate the next number:
    - For S_i in sequences S_{K-1}, S_{K-2}, ... S_1, S_0 (original sequence)
        - Extrapolate the sequence S_i by adding a number S_i[-1] + S_{i+1}[-1], so we add the last value from the next sequence
    """
    sequences = [seq]
    seq_temp = seq
    while not all((x == 0 for x in seq_temp)):
        seq_temp = list(np.diff(seq_temp))
        sequences.append(seq_temp)
        
    # Extrapolate the next number
    for i in range(len(sequences) - 2, -1, -1):
        sequences[i].append(sequences[i][-1] + sequences[i+1][-1])
        
    
    return sequences[0][-1]

def extrapolate_forward(seq):
    """
    Given a sequence of numbers, extrapolate the previous number in the sequence
    """
    sequences = [seq]
    seq_temp = seq
    while not all((x == 0 for x in seq_temp)):
        seq_temp = list(np.diff(seq_temp))
        sequences.append(seq_temp)
    
    print(f"Diff sequences:")
    for seq in sequences:
        print(seq)
        
    # Extrapolate the previous number.
    # The previous number must be the first number in the sequence minus the first number in the next sequence
    for i in range(len(sequences) - 2, -1, -1):
        sequences[i].insert(0, sequences[i][0] - sequences[i+1][0])
    
    print(f"Extrapolated sequences:")
    for seq in sequences:
        print(seq)
          
    print()
    
    return sequences[0][0]

extrap_nums = []
for line in data.split("\n"):
    seq = [int(x) for x in line.split(" ")]
    extrap_num = extrapolate_forward(seq)
    extrap_nums.append(extrap_num)

print(sum(extrap_nums))

aocd.submit(sum(extrap_nums), day=DAY, year=2023, part="b")