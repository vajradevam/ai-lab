from utils import generate_maze, print_maze
from solver import best_first_search, manhattan_distance
from vizualizer import MazeVisualizer

maze = generate_maze(rows=100, cols=100, sparsity=0.2)
maze_states, path_length, visited = best_first_search(maze)
visualizer = MazeVisualizer(maze_states)
visualizer.visualize_step_by_step(step_delay=0.01)