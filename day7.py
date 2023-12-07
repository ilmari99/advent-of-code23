from typing import Any
import aocd
import os
from collections import Counter
import bisect
from itertools import combinations_with_replacement
import functools
import random
import time

data = """32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483"""

start_time = time.time()

#data = aocd.get_data(day=7, year=2023)
#if not os.path.exists("day7.txt"):
#    with open("day7.txt", "w") as f:
#        f.write(data)
        
with open("day7_bigboy.txt", "r") as f:
    data = f.read()
        
CARD_STRENGTHS = { **{str(i): i for i in range(2, 10)} , **{"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14} }
INVERSE_CARD_STRENGTHS = {v: k for k, v in CARD_STRENGTHS.items()}


def generate_case(nhands):
    """ Generates a random case with nhands hands.
    """
    hands = []
    for _ in range(nhands):
        hand = []
        for _ in range(5):
            card = random.choice(list(CARD_STRENGTHS.keys()))
            hand.append(card)
        bid = random.randint(1, 10000)
        hands.append((hand, bid))
    data = "\n".join(["".join(hand) + " " + str(bid) for hand, bid in hands])
    return data

def hand_evaluation_primary(cards):
    """ Returns the primary evaluation score of a hand.
    
    If no pairs, returns the highest card rank of a card.
    One pair: k + 14,
    2 <= p < 2*14
    
    Two pair: k1 + k2 + 2*14,
    2*14 < p < 4*14
    
    Three of a kind: k + 4*14
    4*14 < p < 5*14
    
    Full house: k1 + k2 + 5*14
    5*14 < p < 7*14
    
    Four of a kind: k + 6*14
    6*14 < p < 7*14
    
    Five of a kind: k + 7*14
    7*14 < p < 8*14
    """
    card_counts = Counter(cards)
    max_count = max(card_counts.values())
    # If no pairs
    if max_count == 1:
        score = 2#max(hand.cards)
    
    # If greatest count is 2 and no other counts are >= 2
    # One pair
    elif max_count == 2 and len([count for count in card_counts.values() if count >= 2]) == 1:
        return 14#max(hand.cards) + 14
    
    # If greatest count is 2 and there is another count == 2
    elif max_count == 2 and len([count for count in card_counts.values() if count == 2]) == 2:
        score = 2*14#sum([card for card, count in card_counts.items() if count == 2]) + 2*14
    
    # If exactly one card has count 3, and no other counts are >= 2
    # Three of a kind
    elif max_count == 3 and len([count for count in card_counts.values() if count >= 2]) == 1:
        score = 4*14#max(hand.cards) + 4*14
    
    # If one count is 3 and another count is 2
    # Full house
    elif max_count == 3 and len([count for count in card_counts.values() if count == 2]) == 1:
        score = 5*14#sum([card for card, count in card_counts.items() if count == 2]) + 5*14
    
    # If one count is 4
    elif max_count == 4:
        score = 6*14#max(hand.cards) + 6*14
    
    # If one count is 5
    elif max_count == 5:
        score = 7*14#max(hand.cards) + 7*14
    
    else:
        raise ValueError("Invalid hand.")
    
    return score

def hand_evaluation_secondary(cards):
    """ The secondary evaluation considers the order of cards in the hand.
    """
    scores = []
    for i, card in enumerate(cards):
        scores.append(str(card + 14))
    # Concatenate the scores, and convert to int
    score = int("".join(scores))
    return score

        
class Hand:
    """ Class representing a hand of cards.
    """
    def __init__(self, cards_str, bid=0):
        # Cards are a string "A35T9" so lets separate them to a list
        #print(f"Received cards in string form: {cards_str}")
        self.cards = cards_str
        #print(f"Parsed cards: {self.cards}")
        self.cards = [CARD_STRENGTHS[card] for card in self.cards]
        self.cards_tuple = self.cards
        # Jacks (11) can be substituted for any card
        # Check what should we substitute our J's to
        # In order to maximize our primary score
        self.jack_indices = tuple([i for i, card in enumerate(self.cards) if card == 11])
        self.cards_tuple = tuple([c if c != 11 else 1 for c in self.cards])
        highest_score = 0
        best_substitution = None
        for substitution in combinations_with_replacement(range(2, 15), len(self.jack_indices)):
            #print(f"Substituting jacks with {substitution}")
            # Substitute the jacks
            for i, j in enumerate(self.jack_indices):
                self.cards[j] = substitution[i]
            # Calculate the primary score
            score = hand_evaluation_primary(self.cards)
            if score > highest_score:
                highest_score = score
                best_substitution = substitution
            if score == 7*14:
                break
        for i,j in enumerate(self.jack_indices):
            self.cards[j] = best_substitution[i]
        self.best_cards = tuple(self.cards)
        self.best_substitution = best_substitution        
        self.primary_score = highest_score
        self.secondary_score = None#self.calc_secondary_score(is_same_type = True)
        self.bid = bid
    
    def __hash__(self):
        return hash(self.cards_tuple)
    
    def __repr__(self):
        original_hand = [card if i not in self.jack_indices else 11 for i, card in enumerate(self.cards)]
        return f"Hand: {[INVERSE_CARD_STRENGTHS[card] for card in original_hand]}, bid: {self.bid}"
    
    #@functools.lru_cache(maxsize=128)
    def calc_secondary_score(self,is_same_type = False):
        
        if is_same_type:
            secondary_score = hand_evaluation_secondary(self.cards_tuple)
        else:
            secondary_score = hand_evaluation_secondary(self.best_cards)
        return secondary_score
            
    
    def __gt__(self, other):
        if self.primary_score > other.primary_score:
            return True
        elif self.primary_score < other.primary_score:
            return False
        # If the primary score is the same
        else:
            if self.secondary_score is None:
                self.secondary_score = self.calc_secondary_score(True)
            if other.secondary_score is None:
                other.secondary_score = other.calc_secondary_score(True)
                
            self_sec_score = self.secondary_score#calc_secondary_score(True)
            other_sec_score = other.secondary_score#calc_secondary_score(True)
            if self_sec_score > other_sec_score:
                return True
            elif self_sec_score < other_sec_score:
                return False
            
    
    def __lt__(self, other):
        if self.primary_score < other.primary_score:
            return True
        elif self.primary_score > other.primary_score:
            return False
        else:
            if self.secondary_score is None:
                self.secondary_score = self.calc_secondary_score(True)
            if other.secondary_score is None:
                other.secondary_score = other.calc_secondary_score(True)
                
            self_sec_score = self.secondary_score
            other_sec_score = other.secondary_score
            
            if self_sec_score < other_sec_score:
                return True
            elif self_sec_score > other_sec_score:
                return False
    

hands = []
for line in data.split("\n"):
    hand_data = line.split(" ")
    card_str = hand_data[0]
    bid = int(hand_data[1])
    #print(f"Creating hand with cards {card_str} and bid {bid}")
    hand = Hand(card_str, bid)
    # Add to hands, and maintain sorted order
    bisect.insort(hands, hand)

#print(hands)
# The rank of the hand is 1, if the hand loses, and len(hands) if the hand is the best hand
# The win of a hand is rank*bid
wins = []
for i, hand in enumerate(hands):
    win = (i+1)*hand.bid
    #print(f"Cards: {hand.cards_tuple}, Best cards: {hand.best_cards}")
    #print(f"Hand primary score: {hand.primary_score}, secondary score: {hand.calc_secondary_score(True)}")
    wins.append(win)
    #print()
ans = sum(wins)
print(ans)

print(f"Time taken (gold): {time.time() - start_time:.2f} seconds")

#aocd.submit(ans, day=7, year=2023, part="b")


    
        