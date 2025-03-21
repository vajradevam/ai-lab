from utils import generate_maze, print_maze
import heapq

def manhattan_distance(a, b):
    """
    Calculates the Manhattan distance between two points.
    
    :param a: Tuple (x1, y1).
    :param b: Tuple (x2, y2).
    :return: Manhattan distance between the two points.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def best_first_search(maze):
    """
    Solves the maze using Best-First Search with Manhattan distance as the heuristic.
    
    :param maze: A 2D list representing the maze.
    :return: A tuple containing:
             - List of maze states.
             - Length of the final path.
             - Number of nodes visited.
    """
    # Find the start and goal positions
    start = None
    goal = None
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'G':
                goal = (i, j)
    if not start or not goal:
        raise ValueError("Start or goal not found in the maze.")

    # Priority queue for Best-First Search
    # Each element is a tuple: (heuristic, x, y, path)
    priority_queue = []
    heapq.heappush(priority_queue, (manhattan_distance(start, goal), start[0], start[1], [start]))

    # Visited nodes to avoid revisiting
    visited = set()
    visited.add(start)

    # List to store maze states at each step
    maze_states = []

    # Number of nodes visited
    nodes_visited = 0

    while priority_queue:
        # Get the node with the smallest heuristic value
        _, x, y, path = heapq.heappop(priority_queue)

        # Increment the number of nodes visited
        nodes_visited += 1

        # Mark the current node as visited
        if maze[x][y] != 'S' and maze[x][y] != 'G':
            maze[x][y] = 'c'

        # Append the current maze state to the list
        maze_states.append([row[:] for row in maze])

        # Check if the goal is reached
        if (x, y) == goal:
            # Mark the final path
            for (px, py) in path:
                if maze[px][py] != 'S' and maze[px][py] != 'G':
                    maze[px][py] = 'p'
            maze_states.append([row[:] for row in maze])
            return maze_states, len(path), nodes_visited

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]):
                if (nx, ny) not in visited and maze[nx][ny] != '#':
                    visited.add((nx, ny))
                    heapq.heappush(priority_queue, (manhattan_distance((nx, ny), goal), nx, ny, path + [(nx, ny)]))

    # If no path is found
    return maze_states, 0, nodes_visited