from typing import Callable

from utils import generate_maze, print_maze

from visualizer import MazeVisualizer

import solvers

def Solve(algorithm: Callable):
    maze = generate_maze(10, 10, 0.8)
    maze_states = algorithm(maze)

    vis = MazeVisualizer(maze_states)
    # vis.visualize_step_by_step(step_delay=0.5)
    # vis.visualize_simple()

    # for maze in maze_states:
    #     print_maze(maze)
    #     print("# ---")

Solve(algorithm=solvers.find_path_dfs)