import os
import aocd

data = """Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11"""
data = aocd.get_data(day=4, year=2023)
if not os.path.exists("day4.txt"):
    with open("day4.txt", "w") as f:
        f.write(data)
data = data.split("\n")


print(data)

class Card:
    id_ = 0
    my_nums = []
    win_nums = []
    def __init__(self, line, ncopies = 1):
        """
        win | my
        Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
        Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
        """
        self.id_ = line.split(":")[0].split(" ")[-1].strip()
        self.id_ = int(self.id_)
        cards_part = line.split(":")[1].strip()
        self.win_nums = [int(num) for num in cards_part.split("|")[0].split(" ") if num]
        self.my_nums = [int(num) for num in cards_part.split("|")[1].split(" ") if num]
        self.ncopies = ncopies
        
    def num_hits(self):
        hits = 0
        for num in self.my_nums:
            if num in self.win_nums:
                hits += 1
        return hits
    
    
    def amount_win(self):
        """ One match one point.
        If more than one, the next match always doubles the points.
        """
        hits = self.num_hits()
        if hits > 1:
            return (2**(hits-1))
        return hits
    
    def __str__(self):
        return f"Card {self.id_}: {self.win_nums} | {self.my_nums}"

cards = []
ncopies = {}
total_num_cards = 0
for i,line in enumerate(data):
    card_i_copies = ncopies.get(i, 1)
    card = Card(line, card_i_copies)
    # Count the nubmer of hits
    # If there are for example 2 hits, then card i+1, i+2 will get copied card_i_copies times
    nhits = card.num_hits()
    print(f"Card {i+1} has {nhits} hits and {card_i_copies} copies")
    for j in range(1, nhits+1):
        ncopies[i+j] = ncopies.get(i+j, 1) + card_i_copies
        print(f"Card {i+j+1} will get {card_i_copies} copies")
    cards.append(card)
    total_num_cards += card_i_copies

print(ncopies)
print(total_num_cards)