from aocd.models import Puzzle
import aocd
import os
problem = Puzzle(year=2023, day=2)

if not os.path.exists("day2.txt"):
    with open("day2.txt", "w") as f:
        f.write(problem.input_data)

"""
As you walk, the Elf shows you a small bag and some cubes which
are either red, green, or blue. Each time you play this game, he
will hide a secret number of cubes of each color in the bag, and
your goal is to figure out information about the number of cubes.

To get information, once a bag has been loaded with cubes, the Elf
will reach into the bag, grab a handful of random cubes, show them
to you, and then put them back in the bag. He'll do this a few times per game.

You play several games and record the information from each game
(your puzzle input). Each game is listed with its ID number (like
the 11 in Game 11: ...) followed by a semicolon-separated list of
subsets of cubes that were revealed from the bag (like 3 red, 5 green, 4 blue).

Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green,
"""
import re

COLORS = ["red", "green", "blue"]



class Game:
    """ Class game, that contains information about a single game.
    """
    def __init__(self, line):
        self.id = int(line.split(":")[0].split(" ")[1])
        # Reveals is a list of tuples, where each tuple has three elements and means a reveal of a color (indexing by COLORS)
        self.reveals = []
        self.minimal_bag = None
        # Loop through each section where the line is split by ";"
        for r in line.split(":")[1].split(";"):
            r = r.strip()
            # r is for example: "3 blue, 4 red"
            # Split by "," to get the count and color of each cude in the reveal
            temp_reveal = [0 for _ in range(len(COLORS))]
            for c in r.split(", "):
                # c is for example: "3 blue"
                # Split by " " to get the count and color
                count, color = c.split(" ")
                # Add the reveal to the list of reveals
                index = COLORS.index(color)
                temp_reveal[index] = int(count)
            # Add the reveal to the list of reveals
            self.reveals.append(tuple(temp_reveal))

    def __repr__(self):
        return f"Game {self.id}: {self.reveals}"
    
    def is_possible(self, bag):
        """ Check if the reveals in this game are possible for the given bag.
        """
        # Loop through each reveal
        for reveal in self.reveals:
            # Loop through each color
            for i in range(len(COLORS)):
                # Check if the reveal is possible
                if reveal[i] > bag[i]:
                    return False
        # If all reveals are possible, return True
        return True
    
    def smallest_possible_bag(self):
        """ Return the smallest possible bag that can contain the reveals of this game.
        """
        bag = [0 for _ in range(len(COLORS))]
        # Loop through each reveal
        for reveal in self.reveals:
            # Loop through each color
            for i in range(len(COLORS)):
                # Check if the reveal is possible
                if reveal[i] > bag[i]:
                    bag[i] = reveal[i]
        self.minimal_bag = bag
        return bag
    
    def power(self):
        """ Return the power of this game.
        """
        if self.minimal_bag is None:
            self.smallest_possible_bag()
        return self.minimal_bag[0] * self.minimal_bag[1] * self.minimal_bag[2]

def power(bag):
    """ Calculate the power of a bag.
    """
    return bag[0] * bag[1] * bag[2]
    
if __name__ == "__main__":
    # Get the input data
    data = problem.input_data.split("\n")
    bag = [12, 13, 14]
    games = []
    possible_games = []
    smallest_bags = []
    for line in data:
        game = Game(line)
        sm_bag = game.smallest_possible_bag()
        smallest_bags.append(sm_bag)
        games.append(game)
    print(f"Found {len(possible_games)} possible games")
    # Calculate the sum of the powers of the smallest bags
    sum_of_powers = 0
    for bag in smallest_bags:
        sum_of_powers += power(bag)
    print(f"Sum of powers = {sum_of_powers}")
    aocd.submit(sum_of_powers, day=2, year=2023, part="b")
        

