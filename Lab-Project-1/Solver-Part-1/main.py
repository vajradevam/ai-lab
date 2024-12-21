import time
import tracemalloc
import curses
import inspect
from typing import Callable

from utils import generate_maze, print_maze
from visualizer import MazeVisualizer
import solvers

def Solve(algorithm: Callable, visualize: bool, rows: int, cols: int, timestep: float):
    maze = generate_maze(rows, cols, 0.8)
    visited, maze_states = algorithm(maze)

    vis = MazeVisualizer(maze_states)
    if visualize:
        vis.visualize_step_by_step(step_delay=timestep)
    else:
        vis.visualize_simple()

def Performance(algorithm: Callable, stdscr: curses.window, rows: int, cols: int):
    # Generate the maze for performance testing
    maze = generate_maze(rows, cols, 0.8)
    
    tracemalloc.start()
    start_time = time.time()

    # Run the algorithm on the maze
    visited, maze_states = algorithm(maze)

    final = maze_states[-1]
    shortest = (lambda matrix: sum(row.count('p') for row in matrix))(final)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Display performance details
    stdscr.addstr(f"Nodes visited: {visited}\n")
    stdscr.addstr(f"Number of nodes in Path: {shortest}\n")
    stdscr.addstr("# ---\n")
    signature = inspect.signature(algorithm)
    stdscr.addstr(f"Algorithm name: {algorithm.__name__}\n")
    stdscr.addstr(f"Function signature: {signature}\n")
    stdscr.addstr(f"Execution Time: {end_time - start_time:.3f} seconds\n")
    stdscr.addstr(f"Current Memory Usage: {current / 10**6:.3f} MB\n")
    stdscr.addstr(f"Peak Memory Usage: {peak / 10**6:.3f} MB\n")
    stdscr.addstr("\n")

    stdscr.getch()  # Wait for user input to continue

def get_maze_size(stdscr: curses.window):
    # Enable input echoing so the user can see what they type
    curses.echo()

    # Prompt user for the length and breadth of the maze
    stdscr.clear()
    stdscr.addstr("Enter Maze Dimensions:\n")
    stdscr.addstr("Length (rows): ")
    stdscr.refresh()

    length = stdscr.getstr().decode("utf-8")
    while not length.isdigit():
        stdscr.addstr("Please enter a valid number for length (rows): ")
        stdscr.refresh()
        length = stdscr.getstr().decode("utf-8")
    rows = int(length)

    stdscr.clear()
    stdscr.addstr("Breadth (columns): ")
    stdscr.refresh()

    breadth = stdscr.getstr().decode("utf-8")
    while not breadth.isdigit():
        stdscr.addstr("Please enter a valid number for breadth (columns): ")
        stdscr.refresh()
        breadth = stdscr.getstr().decode("utf-8")
    cols = int(breadth)

    # Disable input echoing after input is complete
    curses.noecho()

    return rows, cols

def get_timestep(stdscr: curses.window):
    # Enable input echoing so the user can see what they type
    curses.echo()

    # Ask for the time step for visualization
    stdscr.clear()
    stdscr.addstr("Enter Time Step Delay for Visualization (seconds): ")
    stdscr.refresh()

    timestep = stdscr.getstr().decode("utf-8")
    while not timestep.replace('.', '', 1).isdigit() or float(timestep) <= 0:
        stdscr.addstr("Please enter a valid positive number for time step: ")
        stdscr.refresh()
        timestep = stdscr.getstr().decode("utf-8")

    # Disable input echoing after input is complete
    curses.noecho()

    return float(timestep)

def main(stdscr: curses.window):
    # Clear the screen
    stdscr.clear()

    # Display initial message
    stdscr.addstr("Maze Solver TUI\n")
    stdscr.addstr("1. Solve\n")
    stdscr.addstr("2. Analyze Performance\n")
    stdscr.refresh()

    # Wait for user to select the option
    choice = stdscr.getch()

    if choice == ord('1'):
        # Solve
        stdscr.clear()
        stdscr.addstr("Solve Maze\n")
        stdscr.refresh()

        # Step 1: Get Maze Dimensions
        rows, cols = get_maze_size(stdscr)

        # Step 2: Choose Algorithm
        stdscr.clear()
        stdscr.addstr("Select an Algorithm:\n")
        stdscr.addstr("1. DFS (Depth First Search)\n")
        stdscr.addstr("2. BFS (Breadth First Search)\n")
        stdscr.refresh()

        choice_algorithm = stdscr.getch()

        if choice_algorithm == ord('1'):
            algorithm = solvers.find_path_dfs
            stdscr.addstr("DFS selected.\n")
        elif choice_algorithm == ord('2'):
            algorithm = solvers.find_path_bfs
            stdscr.addstr("BFS selected.\n")
        else:
            stdscr.addstr("Invalid selection. Exiting...\n")
            stdscr.refresh()
            stdscr.getch()
            return

        stdscr.refresh()

        # Step 3: Choose Operation (Run or Analyze Performance)
        stdscr.clear()
        stdscr.addstr("Select an Operation:\n")
        stdscr.addstr("1. Run Algorithm (Final State)\n")
        stdscr.addstr("2. Run Algorithm (Visualize Each Timestep)\n")
        stdscr.refresh()

        choice_operation = stdscr.getch()

        if choice_operation == ord('1'):
            visualize = False
            stdscr.addstr("Run selected (Final State).\n")
        elif choice_operation == ord('2'):
            visualize = True
            stdscr.addstr("Visualize selected (Each Timestep).\n")
            # Step 4: Get Time Step for visualization (without pressing Enter)
            stdscr.addstr("Enter time step (in seconds): ")
            curses.echo()  # Enable input echo
            timestep_str = stdscr.getstr().decode('utf-8')
            curses.noecho()  # Disable input echo
            try:
                timestep = float(timestep_str)
            except ValueError:
                timestep = 0.5  # Default value if input is invalid
            stdscr.addstr(f"Time step set to {timestep} seconds.\n")
        else:
            stdscr.addstr("Invalid selection. Exiting...\n")
            stdscr.refresh()
            stdscr.getch()
            return

        stdscr.refresh()

        # Step 5: Execute the algorithm based on the selection
        stdscr.clear()
        stdscr.addstr("Executing...\n")
        stdscr.refresh()

        Solve(algorithm, visualize, rows, cols, timestep if visualize else 0)

        stdscr.getch()

    elif choice == ord('2'):
        # Analyze Performance
        stdscr.clear()
        stdscr.addstr("Analyze Performance\n")
        stdscr.refresh()

        # Step 1: Get Maze Dimensions
        rows, cols = get_maze_size(stdscr)

        # Step 2: Analyze Performance of Both Algorithms without Enter key
        stdscr.addstr("------DFS------\n")
        Performance(solvers.find_path_dfs, stdscr, rows, cols)

        stdscr.addstr("------BFS------\n")
        Performance(solvers.find_path_bfs, stdscr, rows, cols)

    else:
        stdscr.addstr("Invalid selection. Exiting...\n")
        stdscr.refresh()
        stdscr.getch()
        return

# Run the curses application
if __name__ == "__main__":
    curses.wrapper(main)