import aocd
import os
import numpy as np

data = """Time:      7  15   30
Distance:  9  40  200"""

data = aocd.get_data(day=6, year=2023)
if not os.path.exists("day6.txt"):
    with open("day6.txt", "w") as f:
        f.write(data)
        
# Read the times and distances to two lists
times = []
distances = []

lines = data.split("\n")
print(f"Lines: {lines}")

times_split = lines[0].split(" ")
print(times_split)

dists_split = lines[1].split(" ")
print(dists_split)
    
times = [num for num in times_split[1:] if num]
distances = [num for num in dists_split[1:] if num]

print(times)
print(distances)

# # convert times to single integer by joining
times = [int("".join(times))]
distances = [int("".join(distances))]

def calculate_button_presses_brute_force(allowed_time, record_distance):
    """ 
    allowed_time: The time allowed for the race.
    record_distance: The distance of the record.
    
    Every second we keep the button pressed, the boat is charged, st.
    when the button is released the boat will go at the speed it was charged
    forever.
    
    We want to find all different lengths that the button can be pressed,
    such that the boat will reach the record distance.
    
    """
    dists_traveled = []
    for i in range(allowed_time):
        time_btn_pressed = i
        speed = i
        dist_traveled = (allowed_time - time_btn_pressed) * speed
        dists_traveled.append(dist_traveled)
    print(dists_traveled)
    return dists_traveled

def calculate_button_presses(allowed_time, record_distance):
    """ We can calculate the distance traveled for each t by
    x = T*t - t^2, where T is the total allowed time.
    We want x >= r + 1, where r is the record distance.
    To solve the interval, we need to solve
    (T-t)t >= r + 1
    -t^2 + Tt - r - 1 >= 0
    """
    a = -1
    b = allowed_time
    c = -record_distance - 1
    sol1 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    sol2 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    print(f"Sol1: {sol1}, sol2: {sol2}")
    # The number of solutions, is the number of integers between sol1 and sol2
    # The lower solution need to rounded up, and the upper solution rounded down
    if sol1 > sol2:
        sol1 = np.floor(sol1)
        sol2 = np.ceil(sol2)
    else:
        sol1 = np.ceil(sol1)
        sol2 = np.floor(sol2)
    print(f"Sol1: {sol1}, sol2: {sol2}")
    # Number of integers between sol1 and sol2, inclusive
    return abs(sol2 - sol1) + 1
    
    

ndists_gt_record = []
for time, dist in zip(times, distances):
    print(f"Time: {time}, dist: {dist}")
    dists_traveled = calculate_button_presses(time, dist)
    print(f"NUmber of solutions: {dists_traveled}")
    ndists_gt_record.append(dists_traveled)
    #dists_gt_record = [d for d in dists_traveled if d > dist]
    #print(f"Dists greater than record: {dists_gt_record}")
    #print(f"Number of dists greater than record: {len(dists_gt_record)}")
    #ndists_gt_record.append(len(dists_gt_record))
    #print()

out = ndists_gt_record[0]
    
aocd.submit(int(out), day=6, year=2023, part="b")

    