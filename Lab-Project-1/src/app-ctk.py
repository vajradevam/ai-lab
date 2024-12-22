import time
import tracemalloc
from typing import Callable

from utils import generate_maze, print_maze
from visualizer import MazeVisualizer
import solvers

import customtkinter as ctk

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

    algorithms = [
        solvers.find_path_bfs,
        solvers.find_path_dfs,
        solvers.find_path_bfs_bidirectional,
        solvers.find_path_dfs_bidirectional
    ]

    result_display.delete(1.0, ctk.END)  # Clear previous results

    for algorithm in algorithms:

        performance_metrics = Performance(algorithm, rows, cols)

        for metric, value in performance_metrics.items():
            result_display.insert(ctk.END, f"{metric}: {value}\n")


def run_gui():
    # Create the main window
    root = ctk.CTk()
    root.title("Maze Solver Performance")
    root.geometry("600x600")  # Set a fixed window size for better layout

    # Add a frame to organize the layout
    main_frame = ctk.CTkFrame(root)
    main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)

    # Configure rows and columns in the main frame to allow expansion
    main_frame.grid_columnconfigure(0, weight=1, minsize=150)
    main_frame.grid_columnconfigure(1, weight=1, minsize=150)
    main_frame.grid_rowconfigure(0, weight=0)
    main_frame.grid_rowconfigure(1, weight=0)
    main_frame.grid_rowconfigure(2, weight=0)
    main_frame.grid_rowconfigure(3, weight=0)
    main_frame.grid_rowconfigure(4, weight=0)
    main_frame.grid_rowconfigure(5, weight=0)
    main_frame.grid_rowconfigure(6, weight=0)
    main_frame.grid_rowconfigure(7, weight=1)  # Allow the output box to expand

    # Font size for labels, entries, buttons, and comboboxes
    font_size = 14

    # Add dropdown for solve/analyze selection
    ctk.CTkLabel(main_frame, text="Select Action:", font=("Arial", font_size)).grid(row=0, column=0, sticky="w", pady=10, padx=10)
    action_var = ctk.StringVar()
    action_menu = ctk.CTkComboBox(main_frame, variable=action_var, values=["Solve", "Analyze"], font=("Arial", font_size))
    action_menu.grid(row=0, column=1, pady=10, padx=10)
    action_menu.set("Solve")

    # Add dropdown for algorithm selection
    ctk.CTkLabel(main_frame, text="Select Algorithm:", font=("Arial", font_size)).grid(row=1, column=0, sticky="w", pady=10, padx=10)
    algorithm_var = ctk.StringVar()
    algorithm_menu = ctk.CTkComboBox(main_frame, variable=algorithm_var, values=[
        "BFS", "DFS", "BFS Bidirectional", "DFS Bidirectional"
        ], font=("Arial", font_size))
    algorithm_menu.grid(row=1, column=1, pady=10, padx=10)
    algorithm_menu.set("BFS")

    # Add dropdown for final vs step-by-step selection
    ctk.CTkLabel(main_frame, text="Visualization Mode:", font=("Arial", font_size)).grid(row=2, column=0, sticky="w", pady=10, padx=10)
    mode_var = ctk.StringVar()
    mode_menu = ctk.CTkComboBox(main_frame, variable=mode_var, values=["Final", "Step-by-Step"], font=("Arial", font_size))
    mode_menu.grid(row=2, column=1, pady=10, padx=10)
    mode_menu.set("Step-by-Step")

    # Add text entry for timestep
    ctk.CTkLabel(main_frame, text="Enter Time Step (seconds):", font=("Arial", font_size)).grid(row=3, column=0, sticky="w", pady=10, padx=10)
    timestep_var = ctk.StringVar()
    timestep_entry = ctk.CTkEntry(main_frame, textvariable=timestep_var, font=("Arial", font_size))
    timestep_entry.grid(row=3, column=1, pady=10, padx=10)

    ctk.CTkLabel(main_frame, text="Rows: ", font=("Arial", font_size)).grid(row=4, column=0, sticky="w", pady=10, padx=10)
    rows_var = ctk.StringVar()  # Initialize rows_var correctly
    rows_entry = ctk.CTkEntry(main_frame, textvariable=rows_var, font=("Arial", font_size))
    rows_entry.grid(row=4, column=1, pady=10, padx=10)

    ctk.CTkLabel(main_frame, text="Columns: ", font=("Arial", font_size)).grid(row=5, column=0, sticky="w", pady=10, padx=10)
    cols_var = ctk.StringVar()  # Initialize cols_var correctly
    cols_entry = ctk.CTkEntry(main_frame, textvariable=cols_var, font=("Arial", font_size))
    cols_entry.grid(row=5, column=1, pady=10, padx=10)

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

    run_button = ctk.CTkButton(main_frame, text="Run", command=on_run, width=200, height=40, font=("Arial", font_size))
    run_button.grid(row=6, column=0, columnspan=2, pady=20, padx=10)

    # Add a text box for performance metrics display
    result_display = ctk.CTkTextbox(main_frame, height=15, font=("Arial", font_size))
    result_display.grid(row=7, column=0, columnspan=2, pady=10, padx=10, sticky="nsew") 

    root.mainloop()

if __name__ == "__main__":
    run_gui()
