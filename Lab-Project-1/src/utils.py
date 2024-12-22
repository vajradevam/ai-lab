import random

def generate_maze(length, breadth, path_usage_percentage: float = 0.8):
    """
    Generate a maze with a given length and breadth.
    The maze includes a random start point (S), a random end point (E), walls (#), and open paths (.).

    Args:
        length (int): Number of rows in the maze.
        breadth (int): Number of columns in the maze.
        path_usage_percentage (float): Percentage of the Maze which is open.

    Returns:
        list: A 2D list representing the generated maze.
    """
    maze = [["#" for _ in range(breadth)] for _ in range(length)]

    start_x, start_y = random.randint(0, length - 1), random.randint(0, breadth - 1)
    end_x, end_y = random.randint(0, length - 1), random.randint(0, breadth - 1)

    while (start_x, start_y) == (end_x, end_y):
        end_x, end_y = random.randint(0, length - 1), random.randint(0, breadth - 1)

    maze[start_x][start_y] = "S"
    maze[end_x][end_y] = "E"

    open_cells = int((length * breadth) * path_usage_percentage)
    visited = set()
    visited.add((start_x, start_y))

    while len(visited) < open_cells:
        x, y = random.randint(0, length - 1), random.randint(0, breadth - 1)
        if (x, y) not in visited and maze[x][y] != "S" and maze[x][y] != "E":
            maze[x][y] = "."
            visited.add((x, y))

    return maze

def print_maze(maze):
    """
    Print the maze in a human-readable format.

    Args:
        maze (list): A 2D list representing the maze.
    """
    for row in maze:
        print(" ".join(row))