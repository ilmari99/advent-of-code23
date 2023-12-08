import aocd
import os
import numpy as np
import time

data = """LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)"""

data = aocd.get_data(day=8, year=2023)
if not os.path.exists("day8.txt"):
    with open("day8.txt", "w") as f:
        f.write(data)

start_time = time.time()
    
data = data.split("\n")

lr_sequence = data[0].strip()
#print(lr_sequence)
print(f"Length of LR sequence: {len(lr_sequence)}")

splitted_lines = [line.split(" = ") for line in data[1:] if len(line) > 1]
#print(splitted_lines)

maps = {}

# Create a map of node to (left, right) nodes
for line in splitted_lines:
    tup = line[1].replace("(", "").replace(")", "").split(", ")
    maps[line[0]] = (tup[0], tup[1])

print(f"Found {len(maps)} nodes")
    
# Start from all nodes ending with Z, and find the length it takes to reach zi from zi
curr_nodes = [node for node in maps.keys() if node.endswith("Z")]
steps_to_reach_z = {}
steps_to_reach_z_again = {}
finished_steps = 0
nsteps = 0
while True:
    step = lr_sequence[nsteps % len(lr_sequence)]
    
    for i in range(len(curr_nodes)):
        # If any node has reached a node ending with Z, then we need to remove it from the list
        if curr_nodes[i].endswith("Z"):
            if curr_nodes[i] not in steps_to_reach_z:
                steps_to_reach_z[curr_nodes[i]] = nsteps
                #print(f"Node {curr_nodes[i]} reached Z in {nsteps} steps")
            elif curr_nodes[i] not in steps_to_reach_z_again:
                steps_to_reach_z_again[curr_nodes[i]] = nsteps
                print(f"Node {curr_nodes[i]} cycle length is {nsteps - steps_to_reach_z[curr_nodes[i]]}")
            else:
                continue

        if step == "L":
            curr_nodes[i] = maps[curr_nodes[i]][0]
        else:
            curr_nodes[i] = maps[curr_nodes[i]][1]
    
    nsteps += 1
        
    # When all nodes have reached z atleast twice, we know the cycle lengths
    if len(steps_to_reach_z_again) == len(curr_nodes):
        # The cycle length is the number of steps between consecutive visits to Z
        cycle_lengths = [steps_to_reach_z_again[node] - steps_to_reach_z[node] for node in steps_to_reach_z]
        
        # The cycles all finish at the same time, when the number of steps is a multiple of each cycle length
        cycles_lcm = np.lcm.reduce(cycle_lengths)
        break
    

print(f"Steps required: {cycles_lcm}")
print(f"Time taken: {time.time() - start_time} seconds")
#aocd.submit(steps, day=8, year=2023, part='b')