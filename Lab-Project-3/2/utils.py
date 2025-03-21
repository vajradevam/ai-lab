import random

def count_inversions(puzzle):
    # Flatten the puzzle and ignore the empty tile (0)
    flat_puzzle = [tile for row in puzzle for tile in row if tile != 0]
    
    inversions = 0
    for i in range(len(flat_puzzle)):
        for j in range(i + 1, len(flat_puzzle)):
            if flat_puzzle[i] > flat_puzzle[j]:
                inversions += 1
    return inversions

def generate_8_puzzle():
    while True:
        # Create a list of numbers from 0 to 8 (0 represents the empty tile)
        numbers = list(range(9))
        random.shuffle(numbers)
        
        # Convert the list into a 3x3 2D list
        puzzle = [numbers[i:i+3] for i in range(0, 9, 3)]
        
        # Check if the number of inversions is even
        if count_inversions(puzzle) % 2 == 0:
            return puzzle

def display_puzzle(puzzle):
    for row in puzzle:
        print(" ".join(str(tile) if tile != 0 else " " for tile in row))
    print()

