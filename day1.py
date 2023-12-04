import aocd
import os

data = aocd.get_data(day=1, year=2023)#.split("\n")
if not os.path.exists("day1.txt"):
    with open("day1.txt", "w") as f:
        f.write(data)
        
        
data = data.split("\n")

digit_identifiers = [str(i) for i in range(10)] + [
    "one","two","three","four","five","six","seven","eight","nine","zero"
]

digit_string_to_digit = {
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"zero":0
}

total = 0
for line in data:
    print(line)
    # Read each line to a list of words with varying lengths, i.e. "abcde" -> ["a","b","c","d","e","ab","bc","cd","de","abc","bcd","cde","abcd","bcde","abcde"]
    words = []
    word_len = max([len(digit_identifiers[i]) for i in range(len(digit_identifiers))])
    for i in range(len(line)):
        for j in range(word_len):
            if i+j+1 <= len(line):
                words.append(line[i:i+j+1])
    print(words)
    
    # Find the first word that is a digit identifier
    first_digit = 0
    for word in words:
        if word in digit_identifiers:
            first_digit = word
            break
    print(first_digit)
    if len(first_digit) > 1:
        first_digit = digit_string_to_digit[first_digit]
    #print(first_digit)
    
    second_digit = 0
    words.reverse()
    
    for word in words:
        if word in digit_identifiers:
            second_digit = word
            break
    print(second_digit)
    if len(second_digit) > 1:
        second_digit = digit_string_to_digit[second_digit]
    p = int("".join([str(first_digit), str(second_digit)]))
    print(p)
    print()
    total += p
    
print(total)

aocd.submit(total, day=1, year=2023, part="2")
    