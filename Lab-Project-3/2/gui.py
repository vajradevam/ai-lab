import customtkinter as ctk
from utils import *
from solver import *
from visualizer import PuzzleVisualizer
import time
import tracemalloc
import threading

class PuzzleApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("8-Puzzle Solver")
        self.geometry("600x500")
        self.resizable(False, False)  # Disable window resizing

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Dropdown for algorithm selection
        self.algorithm_label = ctk.CTkLabel(self, text="Select Algorithm:")
        self.algorithm_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.algorithm_var = ctk.StringVar(value="a_star")
        self.algorithm_dropdown = ctk.CTkOptionMenu(self, values=["a_star", "greedy_bfs"], variable=self.algorithm_var)
        self.algorithm_dropdown.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="ew")

        # Dropdown for heuristic selection
        self.heuristic_label = ctk.CTkLabel(self, text="Select Heuristic:")
        self.heuristic_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.heuristic_var = ctk.StringVar(value="manhattan")
        self.heuristic_dropdown = ctk.CTkOptionMenu(self, values=["manhattan", "misplaced"], variable=self.heuristic_var)
        self.heuristic_dropdown.grid(row=1, column=1, padx=20, pady=10, sticky="ew")

        # Entry for time step
        self.timestep_label = ctk.CTkLabel(self, text="Time Step (seconds):")
        self.timestep_label.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.timestep_entry = ctk.CTkEntry(self)
        self.timestep_entry.insert(0, "1.0")
        self.timestep_entry.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

        # Buttons for solve and analyze
        self.solve_button = ctk.CTkButton(self, text="Solve", command=self.solve_puzzle)
        self.solve_button.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        self.analyze_button = ctk.CTkButton(self, text="Analyze", command=self.analyze_puzzle)
        self.analyze_button.grid(row=3, column=1, padx=20, pady=20, sticky="ew")

        # Text box for performance metrics
        self.metrics_text = ctk.CTkTextbox(self, width=400, height=200)
        self.metrics_text.grid(row=4, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew")

    def solve_puzzle(self):
        # Run the solver in a separate thread to keep the GUI responsive
        threading.Thread(target=self._solve_puzzle, daemon=True).start()

    def _solve_puzzle(self):
        puzzle = generate_8_puzzle()

        algorithm = self.algorithm_var.get()
        heuristic = self.heuristic_var.get()
        time_step = float(self.timestep_entry.get())

        if algorithm == "a_star":
            solution_path, nodes_expanded, path_cost = a_star(puzzle, heuristic=heuristic)
        elif algorithm == "greedy_bfs":
            solution_path, nodes_expanded, path_cost = greedy_bfs(puzzle, heuristic=heuristic)

        visualizer = PuzzleVisualizer(solution_path)
        visualizer.visualize(time_step=time_step)
        visualizer.keep_window_open()

    def analyze_puzzle(self):
        puzzle = generate_8_puzzle()

        algorithms = ["a_star", "greedy_bfs"]
        heuristics = ["manhattan", "misplaced"]

        results = []

        for algo in algorithms:
            for heuristic in heuristics:
                tracemalloc.start()
                start_time = time.time()

                if algo == "a_star":
                    solution_path, nodes_expanded, path_cost = a_star(puzzle, heuristic=heuristic)
                elif algo == "greedy_bfs":
                    solution_path, nodes_expanded, path_cost = greedy_bfs(puzzle, heuristic=heuristic)

                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                execution_time = end_time - start_time
                mem_usage = peak / 10**6  # Convert to MB

                results.append({
                    "Algorithm": algo,
                    "Heuristic": heuristic,
                    "Nodes Expanded": nodes_expanded,
                    "Path Cost": path_cost,
                    "Execution Time (s)": execution_time,
                    "Memory Usage (MB)": mem_usage
                })

        # Display results in the text box
        self.metrics_text.delete("1.0", ctk.END)
        self.metrics_text.insert(ctk.END, "Algorithm\tHeuristic\tNodes Expanded\tPath Cost\tExecution Time (s)\tMemory Usage (MB)\n")
        for result in results:
            self.metrics_text.insert(ctk.END, f"{result['Algorithm']}\t{result['Heuristic']}\t{result['Nodes Expanded']}\t{result['Path Cost']}\t{result['Execution Time (s)']:.4f}\t{result['Memory Usage (MB)']:.2f}\n")

if __name__ == "__main__":
    app = PuzzleApp()
    app.mainloop()