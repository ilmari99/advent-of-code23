import os
import time
import aocd


data = """seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4"""

data = aocd.get_data(day=5, year=2023)
if not os.path.exists("day5.txt"):
    with open("day5.txt", "w") as f:
        f.write(data)
# Split at empty lines
data = data.split("\n\n")
for i in range(len(data)):
    print(f"{data[i]}")
    
def parse_part(part):
    """ Parse the part describing the map in ranges
    """
    # name is the first line in part
    name = part.split(":\n")[0]
    part = part.split(":\n")[1]
    ranges = []
    # The rest of the lines are the ranges
    # Each range has three numbers: dest start, source start, length
    # So the map is range(dest_start, dest_start+length) = range(source_start, source_start+length)
    for range_str in part.split("\n"):
        dest_start, source_start, length = [int(num) for num in range_str.split(" ")]
        ranges.append((dest_start, source_start, length))
    return name, ranges

def get_seeds(first_row):
    """ Return the seeds
    """
    seed_elems = [int(num) for num in first_row.split("seeds: ")[1].split(" ")]
    return seed_elems

def get_seed_ranges(first_row):
    seeds = get_seeds(first_row)
    # Every even index is a start, and every odd index is a length
    seed_ranges = []
    for i in range(0, len(seeds), 2):
        seed_ranges.append((seeds[i], seeds[i+1]))
    return seed_ranges

def is_in_seed_range(val, seed_range):
    """ Return whether val is in seed_range
    """
    return seed_range[0] <= val < seed_range[0] + seed_range[1]
    
class DestinationSourceMap:
    """ Class containing a seed-to-dest-to-seed map
    """
    def __init__(self, part, index = 0):
        self.part = part
        self.name, self.ranges = parse_part(part)
        self.index = index

    def get_seed_to_dest(self, seed):
        """ Return the destination of seed
        This is done by
        1. finding the range that seed is in. (if not found, return seed)
        2. Return the corresponding destination value:
        """
        for dest_start, source_start, length in self.ranges:
            if source_start <= seed < source_start + length:
                return dest_start + (seed - source_start)
        return seed
    
    def get_dest_to_seed(self, dest, searched_ranges = set()):
        """ Return the seed that corresponds to dest
        This is done by
        1. finding the range that dest is in. (if not found, return dest)
        2. Return the corresponding seed value:
        """
        i = 0
        
        # Keep track of the last searched range index
        last_searched_range_index = 0
        for dest_start, source_start, length in self.ranges:
            #print(f"Searching range {i}/{len(self.ranges)-1} in map {self.index}")
            if (self.index, i) in searched_ranges:
                i += 1
                continue
            
            if dest_start <= dest < dest_start + length:
                return source_start + (dest - dest_start), i, self.ranges[i]
            last_searched_range_index = i
            i += 1
            
        #print(f"Last searched range index: {last_searched_range_index} in map {self.index}")
        return dest, last_searched_range_index, self.ranges[last_searched_range_index]
    
def find_starting_value(value, d_s_maps, searched_ranges = set()):
    """
    - If the starting value is found (i.e. we go trough a route we have not gone through before)
    then we return the starting value at level 0. If not, we return False
    """
    shortest_range = float("inf")
    for d_s_map in reversed(d_s_maps):
        ds_i = d_s_map.index
        # Value is the seed in map ds_i, i.e. the destination in map ds_i-1
        value, rangei, range_obj = d_s_map.get_dest_to_seed(value, searched_ranges=searched_ranges)
        range_length = range_obj[2]
        if range_length < shortest_range:
            shortest_range = range_length
        if not value:
            break
    return value, ds_i, rangei, range_obj, shortest_range

d_s_maps = []
for ind, part in enumerate(data[1:]):
    d_s_maps.append(DestinationSourceMap(part, index=ind))

#seeds = get_seeds(data[0])
seed_ranges = get_seed_ranges(data[0])
# Start from the smallest value in the location map, and find the seed that corresponds to it
# Then go to the previous map, and find the seed that corresponds to the seed in the location map
# Repeat for all maps
# Check if the final seed is in the seed ranges
# If not, go to the location map and pick the next smallest value
# Repeat until the seed is in the seed ranges
# Then return the smallest value



#print(f"Smallest possible locations: {smallest_locations}")
# Searched ranges contain tuples of (dest_source_map_index, range_index) that have been searched
searched_ranges = set()
location = 0
max_location = 100000000
while location < max_location:
    value = location
    #print(f"Location: {location}")
    # Location is the value in the final map
    # For each map before that, we need to find the seed that corresponds to the value
    # We then use that seed to find the value in the next map, and so on until we reach the first map

    # returns the starting value (or False if it can not be the smallest start value)
    # dsi is the index of the level that the value was found in
    value, dsi, rangei, range_obj, shortest_range = find_starting_value(value, d_s_maps, searched_ranges=searched_ranges)
    
    #print(f"Shortest range: {shortest_range}")
    
    if value and any((is_in_seed_range(value, seed_range) for seed_range in seed_ranges)):
        #print(f"Found seed {value} at location {location}")
        smallest_found_seed = location
        break
    
    # If the starting location is not in the seed ranges, we can discard all the values in the range
    # (dest_start_map_index, range_index)
    #searched_ranges.add((dsi, rangei))

    location += 1# shortest_range - 1
    
    #print(f"Location {location} corresponds to seed {value} in map {dsi} range {rangei}")
    #if location % 0 == 0:
    #print(f"Searched ranges: {searched_ranges}")
    

print(smallest_found_seed)
print(f"Correct answer: {12634632}")
#aocd.submit(smallest_found_seed, day=5, year=2023, part='b')
    