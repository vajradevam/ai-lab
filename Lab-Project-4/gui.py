import customtkinter as ctk
from solver import hill_climbing
from visualizer import NQueensVisualizer
import time
import tracemalloc
from multiprocessing import Process

class NQueensSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("N-Queens Solver")
        self.root.geometry("600x500")

        # Input for number of queens
        self.label_n = ctk.CTkLabel(root, text="Number of Queens (n):")
        self.label_n.pack(pady=(10, 0))
        self.entry_n = ctk.CTkEntry(root)
        self.entry_n.pack(pady=(0, 10))

        # Input for time step
        self.label_time_step = ctk.CTkLabel(root, text="Time Step (seconds):")
        self.label_time_step.pack(pady=(10, 0))
        self.entry_time_step = ctk.CTkEntry(root)
        self.entry_time_step.pack(pady=(0, 20))

        # Button to start solving
        self.solve_button = ctk.CTkButton(root, text="Solve", command=self.solve)
        self.solve_button.pack(pady=(10, 20))

        # Text box for statistics
        self.stats_text = ctk.CTkTextbox(root, height=100)
        self.stats_text.pack(pady=(10, 0), padx=10, fill="both", expand=True)

    def solve(self):
        n = int(self.entry_n.get())
        time_step = float(self.entry_time_step.get())

        self.stats_text.delete("1.0", "end")

        # Start tracking memory usage
        tracemalloc.start()

        # Start tracking execution time
        start_time = time.time()

        # Run the solver
        path, solution_found, final_fitness = hill_climbing(n)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Display statistics
        stats = (f"Execution Time: {execution_time:.2f} seconds\n"
                 f"Memory Used: {peak / 10**6:.2f} MB\n"
                 f"Final Fitness: {final_fitness}\n"
                 f"Solution Found: {solution_found}\n")
        self.stats_text.insert("1.0", stats)

        # Launch the visualizer in a separate process
        visualizer_process = Process(target=self.run_visualizer, args=(path, time_step))
        visualizer_process.start()

    def run_visualizer(self, path, time_step):
        """Helper function to run the visualizer in a separate process."""
        visualizer = NQueensVisualizer(path, window_size=800, time_step=time_step)
        visualizer.run()

if __name__ == "__main__":
    root = ctk.CTk()
    app = NQueensSolverApp(root)
    root.mainloop()