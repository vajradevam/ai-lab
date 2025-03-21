from solver import *
from visualizer import *

# Example usage
n = 8  # Size of the chessboard
path, solution_found, final_fitness = hill_climbing(n)

print("Path taken:")
for board in path:
    print_chessboard(board)

print(f"Solution found: {solution_found}")
print(f"Final fitness value: {final_fitness}")

visualizer = NQueensVisualizer(path, window_size=800, time_step=0.2)
visualizer.run()