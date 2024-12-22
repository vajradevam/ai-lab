import time
import tracemalloc
from typing import Callable

from utils import generate_maze, print_maze
from visualizer import MazeVisualizer
import solvers

import tkinter as tk
from tkinter import ttk

def Solve(algorithm: Callable, visualize: bool, rows: int, cols: int, timestep: float):
    maze = generate_maze(rows, cols, 0.8)
    visited, maze_states = algorithm(maze)

    vis = MazeVisualizer(maze_states)
    if visualize:
        vis.visualize_step_by_step(step_delay=timestep)
    else:
        vis.visualize_simple()


def Performance(algorithm: Callable, rows: int, cols: int):
    maze = generate_maze(rows, cols, 0.8)

    tracemalloc.start()
    start_time = time.time()

    visited, maze_states = algorithm(maze)

    final = maze_states[-1]
    shortest = sum(row.count('p') for row in final)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    performance_metrics = {
        "Algorithm:": algorithm.__name__,
        "Nodes visited:": visited,
        "Number of nodes in Path:": shortest,
        "Execution Time:": f"{end_time - start_time:.2f} seconds",
        "Current Memory Usage:": f"{current / 10**6:.2f} MB",
        "Peak Memory Usage:": f"{peak / 10**6:.2f} MB",
        "#": f" --- "
    }
    return performance_metrics


def run_performance_metrics(rows, cols, result_display):

    # if algorithm_name == "BFS":
    #     algorithm = solvers.find_path_bfs
    # if algorithm_name == "DFS":
    #     algorithm = solvers.find_path_bfs
    # if algorithm_name == "BFS Bidirectional":
    #     algorithm = solvers.find_path_bfs
    # if algorithm_name == "DFS Bidirectional":
    #     algorithm = solvers.find_path_bfs

    algorithms = [
        solvers.find_path_bfs,
        solvers.find_path_dfs,
        solvers.find_path_bfs_bidirectional,
        solvers.find_path_dfs_bidirectional
    ]

    result_display.delete(1.0, tk.END)  # Clear previous results

    for algorithm in algorithms:

        performance_metrics = Performance(algorithm, rows, cols)

        for metric, value in performance_metrics.items():
            result_display.insert(tk.END, f"{metric}: {value}\n")


def run_gui():
    # Create the main window
    root = tk.Tk()
    root.title("Maze Solver")

    # Add dropdown for solve/analyze selection
    ttk.Label(root, text="Select Action:").grid(row=0, column=0)
    action_var = tk.StringVar()
    action_menu = ttk.Combobox(root, textvariable=action_var, values=["Solve", "Analyze"])
    action_menu.grid(row=0, column=1)
    action_menu.set("Solve")

    # Add dropdown for algorithm selection
    ttk.Label(root, text="Select Algorithm:").grid(row=1, column=0)
    algorithm_var = tk.StringVar()
    algorithm_menu = ttk.Combobox(root, textvariable=algorithm_var, values=[
        "BFS", "DFS", "BFS Bidirectional", "DFS Bidirectional"
        ])
    algorithm_menu.grid(row=1, column=1)
    algorithm_menu.set("BFS")

    # Add dropdown for final vs step-by-step selection
    ttk.Label(root, text="Visualization Mode:").grid(row=2, column=0)
    mode_var = tk.StringVar()
    mode_menu = ttk.Combobox(root, textvariable=mode_var, values=["Final", "Step-by-Step"])
    mode_menu.grid(row=2, column=1)
    mode_menu.set("Step-by-Step")

    # Add text entry for timestep
    ttk.Label(root, text="Enter Time Step (seconds):").grid(row=3, column=0)
    timestep_var = tk.StringVar()
    timestep_entry = ttk.Entry(root, textvariable=timestep_var)
    timestep_entry.grid(row=3, column=1)

    ttk.Label(root, text="Rows: ").grid(row=4, column=0)
    rows_var = tk.StringVar()  # Initialize rows_var correctly
    rows_entry = ttk.Entry(root, textvariable=rows_var)
    rows_entry.grid(row=4, column=1)

    ttk.Label(root, text="Columns: ").grid(row=5, column=0)
    cols_var = tk.StringVar()  # Initialize cols_var correctly
    cols_entry = ttk.Entry(root, textvariable=cols_var)
    cols_entry.grid(row=5, column=1)

    # Add a button to run performance metrics
    def on_run():
        action = action_var.get()

        rows = int(rows_var.get())
        cols = int(cols_var.get())

        if action == "Solve":
            algorithm_name = algorithm_var.get()

            if algorithm_name == "BFS":
                algorithm = solvers.find_path_bfs
            if algorithm_name == "DFS":
                algorithm = solvers.find_path_dfs
            if algorithm_name == "BFS Bidirectional":
                algorithm = solvers.find_path_bfs_bidirectional
            if algorithm_name == "DFS Bidirectional":
                algorithm = solvers.find_path_dfs_bidirectional

            timestep = float(timestep_var.get())  # Get timestep value
            mode = mode_var.get()  # Get visualization mode

            visualize = mode == "Step-by-Step"
            Solve(algorithm, visualize, rows, cols, timestep)

        if action == "Analyze":
            performance_text = result_display
            run_performance_metrics(rows, cols, performance_text)

    run_button = ttk.Button(root, text="Run", command=on_run)
    run_button.grid(row=6, column=0, columnspan=2)

    # Add a text box for performance metrics display
    result_display = tk.Text(root, height=15, width=50)
    result_display.grid(row=7, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
