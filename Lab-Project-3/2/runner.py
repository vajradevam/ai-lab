from utils import *
from solver import *

# Generate a random 8-puzzle
puzzle = generate_8_puzzle()
# Display the puzzle
display_puzzle(puzzle)

# Solve using Manhattan distance heuristic
solution_path, nodes_expanded, path_cost = a_star(puzzle, heuristic="manhattan")

print(solution_path, nodes_expanded, path_cost)

from visualizer import PuzzleVisualizer

# Create visualizer object
visualizer = PuzzleVisualizer(solution_path)

# Visualize with a time step of 1 second
visualizer.visualize(time_step=1.0)

# Keep the window open
visualizer.keep_window_open()