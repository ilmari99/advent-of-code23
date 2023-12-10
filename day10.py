""" 
| is a vertical pipe connecting north and south.
- is a horizontal pipe connecting east and west.
L is a 90-degree bend connecting north and east.
J is a 90-degree bend connecting north and west.
7 is a 90-degree bend connecting south and west.
F is a 90-degree bend connecting south and east.
. is ground; there is no pipe in this tile.
S is the starting position of the animal; there is a pipe on this tile, but your sketch doesn't show what shape the pipe has.

We want to find the part in the loop, that start from S, that is furthest away from S (number of steps).


"""

NUMBER_TO_DIRECTION = {
    0: "N",
    1: "E",
    2: "S",
    3: "W"
}

SYMBOL_TO_CONNECTIONS = {
    "|" : [0,2],
    "-" : [1,3],
    "L" : [0,1],
    "J" : [0,3],
    "7" : [2,3],
    "F" : [1,2],
    "S" : [0,2],#[0,1,2,3],
    ".": []
}

SYMBOL_TO_NUMBER = {
    "|" : 0,
    "-" : 1,
    "L" : 2,
    "J" : 3,
    "7" : 4,
    "F" : 5,
    "S" : 6,
    ".": 7
}

DIRECTION_TO_COORDINATE_CHANGE = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}

NUMBER_TO_SYMBOL = { v:k for k,v in SYMBOL_TO_NUMBER.items()}

from dataclasses import dataclass
import itertools
import time
import aocd
import os
from matplotlib import pyplot as plt
import numpy as np

DAY = 10
input_file = f"day{DAY}.txt"

data = """.....
.S-7.
.|.|.
.L-J.
....."""

data = aocd.get_data(day=DAY, year=2023)
if not os.path.exists(input_file):
    with open(input_file, "w") as f:
        f.write(data)

# Convert the data to a matrix of numbers
DATA = np.array([[SYMBOL_TO_NUMBER[symbol] for symbol in line] for line in data.splitlines()])
print(DATA)
print(DATA.shape)

# Coordinates to pipe objects
FOUND_PIPES = {
    
}
        
class Pipe:
    """ A pipe object. Each pipe connects two other pipes,
    except '.' which is ground and has no connections, and 'S' which is the starting position and connects each direction.
    This pipe object is used as a node in a graph.
    
    The class has attributes:
    - symbol: The symbol representing the pipe
    - connections: A list of the directions this pipe connects to
    - neighbours: A list of the pipes this pipe connects to in the same order as the connections list
    - coordinates: The coordinates of the pipe in the grid
    """
    
    def __init__(self, symbol, coordinates):
        """ This is called when a new pipe is created.
        """
        # Coordinates in the full grid
        self.coordinates = coordinates
        
        # The symbol and symbol_num of the pipe
        self.symbol = symbol
        self.symbol_num = SYMBOL_TO_NUMBER[self.symbol]
        
        # How the pipe connects to other pipes
        self.connections = SYMBOL_TO_CONNECTIONS[symbol]
        
        self.steps = -1
        
        self.neighbours_ = []
        
        FOUND_PIPES[coordinates] = self
        print(f"Created pipe {self}")
        
    @classmethod
    def get_pipe(cls, coordinates):
        """ Get the pipe at the given coordinates.
        If the pipe has not been found yet, create it.
        """
        if coordinates in FOUND_PIPES:
            return FOUND_PIPES[coordinates]
        else:
            return cls(NUMBER_TO_SYMBOL[DATA[coordinates]], coordinates)
    
    def __repr__(self) -> str:
        return f"Pipe(\'{self.symbol}\', {self.coordinates})"
    
    @property
    def neighbours(self):
        """ Return the neighbours of this pipe. If they have not been found yet, find them.
        """
        if self.neighbours_ == []:
            self.neighbours_ = self.find_neighbours()
        return self.neighbours_
    
    def direction_to_real_direction(self, direction):
        """ Given a direction (as a number), return the direction (the index change) in the grid.
        """
        # Means we are going up
        if direction == 0:
            return (-1, 0)
        # Means we are going right
        elif direction == 1:
            return (0, 1)
        # Means we are going down
        elif direction == 2:
            return (1, 0)
        # Means we are going left
        elif direction == 3:
            return (0, -1)
        
        
    def find_neighbours(self):
        """ Find the neighbours of this pipe from the grid.
        """
        # For each connection, find the neighbour in that direction
        neighbours = []
        new_connections = []
        for connection in self.connections:
            coord_change = DIRECTION_TO_COORDINATE_CHANGE[connection]
            neighbour_coord = (self.coordinates[0] + coord_change[0], self.coordinates[1] + coord_change[1])
            neighbour = DATA[neighbour_coord]
            if neighbour == SYMBOL_TO_NUMBER["."]:
                # Lets not add ground as a neighbour because it is not traversable
                continue
            # Also, we might not have a neighbour at the location because it is outside the grid
            if any((x < 0 for x in neighbour_coord)) or any((c1 >= DATA.shape[i] for i,c1 in enumerate(neighbour_coord))):
                continue
            
            neighbour = Pipe.get_pipe(neighbour_coord)
            
            # Only add, if the neighbour has a connection back to this pipe            
            if connection == 0:
                if 2 not in neighbour.connections:
                    continue
            elif connection == 1:
                if 3 not in neighbour.connections:
                    continue
            elif connection == 2:
                if 0 not in neighbour.connections:
                    continue
            elif connection == 3:
                if 1 not in neighbour.connections:
                    continue

            neighbours.append(neighbour)
            new_connections.append(connection)
        self.connections = new_connections
        return neighbours
    
    def __eq__(self, other):
        """ Two pipes are equal if they have the same coordinates.
        """
        return self.coordinates == other.coordinates

def create_graph(head, verbose=1):
    head.steps = 0
    
    head_neighs = head.neighbours
    for neigh in head_neighs:
        neigh.steps = 1

    for starting_direction in head.neighbours:
        print(f"Starting direction: {starting_direction}")
        
        curr_pipe = starting_direction
        prev_pipe = head
        
        while True:
            # Find the next pipe, which is the neighbour of the current pipe that is not the previous pipe
            next_pipe = [neighbour for neighbour in curr_pipe.neighbours if neighbour != prev_pipe][0]
            if verbose:
                print(f"Current pipe: {curr_pipe}")
                print(f"Next pipe: {next_pipe}")
                print(f"Current pipe steps: {curr_pipe.steps}")
                print(f"Next pipe steps: {next_pipe.steps}\n")
                #exit()
            
            # Update steps to the next pipe if it has not been found yet
            if next_pipe.steps == -1:
                next_pipe.steps = curr_pipe.steps + 1
                
            # If the number of steps to reach next pipe is greater than curr_pipe.steps + 1, update it
            # because we have found a shorter path to next_pipe
            elif next_pipe.steps > curr_pipe.steps + 1:
                next_pipe.steps = curr_pipe.steps + 1
            
            elif next_pipe.steps <= curr_pipe.steps:
                break
            
            
            # Update the pipes
            prev_pipe = curr_pipe
            curr_pipe = next_pipe
    return head

# Find the coordinate of the starting pipe
start_coords = np.argwhere(DATA == SYMBOL_TO_NUMBER["S"])[0]
start_coords = (start_coords[0], start_coords[1])
print(f"Coordinates of S: {start_coords}")

# Create the graph of pipes
head = create_graph(Pipe.get_pipe(start_coords))

# Find the pipe that is furthest away from the starting pipe
max_num_steps = max((pipe.steps for pipe in FOUND_PIPES.values()))
print(f"Max number of steps: {max_num_steps}")

# Now we want to find the largest area of ground '.' that is enclosed by pipes.
# To do this, we need to find the coordinates of every pipe that encloses the area.
area_edge_pipes = FOUND_PIPES

# Make a matrix, but with only the pipes
pipe_matrix = -np.ones(DATA.shape)
for pipe in FOUND_PIPES.values():
    pipe_matrix[pipe.coordinates] = pipe.symbol_num

#Highligth the starting pipe
pipe_matrix[start_coords] = 8

plt.imshow(pipe_matrix)

# An important distinction is that an area is not enclosed by just being surrounded by pipes.
# It must be enclosed by pipes that are connected to each other:
# - It does not count if two pipes run parallel to each other, but are not connected.

# We need to find the edges of the pipeline and calculate the total area
# We then have the inner edges of the pipeline.
# Then, we can find all the squares in that are enclosed by the coordinates,
# Calculate tha areas of the squares, and sum them up.


# calculate the coordinates of the inner edges of the pipeline
# Each pipe block of course has 4 corners, i.e. 4 possible 'inner edges'
# What corner we consider the inner edge depends on the direction of the pipe

# FOUND_PIPES is otherwise in order, BUT its third element should be the last element
found_pipes_list = list(FOUND_PIPES.items())
should_be_last_element = found_pipes_list.pop(1)
print(f"Should be last element: {should_be_last_element}")
found_pipes_list.append(should_be_last_element)
FOUND_PIPES = dict(found_pipes_list)



# FOUND_PIPES contains the grids indices, where a pipe is located.
# We want an array, which contains straight lines from (x1, y1) to (x2, y2) for each section of the pipeline.

pipelines = []
pipes_in_list = list(FOUND_PIPES.items())
pipeline_end_idx = pipes_in_list[0][1].coordinates
pipeline_begin_idx = pipes_in_list[-1][1].coordinates
pipelines.append((pipeline_begin_idx, pipeline_end_idx))
direction = (pipeline_end_idx[0] - pipeline_begin_idx[0], pipeline_end_idx[1] - pipeline_begin_idx[1])
prev_direction = direction

pipeline_begin_idx = pipeline_end_idx
num_direction_changes = 0

def get_adjustment(direction, prev_direction):
    return (0,0)

# We want to create a list of tuples, where one tuple describes two coordinates: The start and end of a straight line
# So everytime the direction changes
for i, (coordinates, pipe) in enumerate(pipes_in_list):
    if i == 0:
        continue
    prev_pipe = pipes_in_list[i-1][1]
    direction = (coordinates[0] - prev_pipe.coordinates[0], coordinates[1] - prev_pipe.coordinates[1])
    
    if any((abs(x) > 1 for x in direction)) or sum(abs(x) for x in direction) > 1:
        continue
    
    # If the direction changes, then we know that the previous pipe was the end of a pipeline.
    if direction != prev_direction:
        
        adjustment = get_adjustment(direction, prev_direction)
        num_direction_changes += 1

        pipeline_begin_inner_coordinate = (pipeline_begin_idx[0] + adjustment[0], pipeline_begin_idx[1] + adjustment[1])
        pipeline_end_inner_coordinate = (prev_pipe.coordinates[0] + adjustment[0], prev_pipe.coordinates[1] + adjustment[1])
        
        pipelines.append((pipeline_begin_inner_coordinate, pipeline_end_inner_coordinate))
        prev_direction = direction
        pipeline_begin_idx = prev_pipe.coordinates

pipelines.append((pipeline_begin_idx, pipeline_end_idx))
# Plot the pipelines
for pipeline in pipelines:
    plt.plot([pipeline[0][1], pipeline[1][1]], [pipeline[0][0], pipeline[1][0]], color="red", linewidth=2)
    
# We now have a list of start-end coordinates.
# The coordinates enclose an area.
# We want to know the area of the enclosed area,
# But only counting full squares, i.e. squares that are not cut off by the pipeline

def shoelace_formula(coordinates):
    """ Calculate the area of a polygon defined by the coordinates using the shoelace formula.
    """
    # The shoelace formula is the sum of the products of the x-coordinates of the points,
    # minus the sum of the products of the y-coordinates of the points
    # The coordinates are in the order (x1, y1), (x2, y2), ...
    # The last point is the same as the first point
    x_coordinates = coordinates[:,1]
    y_coordinates = coordinates[:,0]
    area = 0.5 * abs(np.dot(x_coordinates, np.roll(y_coordinates, 1)) - np.dot(y_coordinates, np.roll(x_coordinates, 1)))
    return area

# Faltten the pipelines, so we only have a list of coordinates
pipelines = np.array([coordinate for pipeline in pipelines for coordinate in pipeline])
r = shoelace_formula(pipelines)
print(f"Area enclosed by the full pipeline: {r}")

# We reduce the area by the area of the pipes
# The area of the pipes is the number of pipes
pipe_area = (len(FOUND_PIPES) / 2) - 1
r -= pipe_area
print(f"Area enclosed by the pipeline: {r}")





plt.show()