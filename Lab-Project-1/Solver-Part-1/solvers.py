from collections import deque

def find_path_bfs(maze):
    """
    Find the path from 'S' to 'E' using Breadth-First Search and return all states.

    Args:
        maze (list): The maze represented as a 2D list.

    Returns:
        tuple: (final_maze, maze_states) where final_maze is the solved maze and 
               maze_states is a list of all intermediate states. Returns (None, [])
               if no path exists.
    """
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    maze_states = []  # Store all intermediate states

    # Locate 'S' and 'E'
    start = end = None
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 'S':
                start = (i, j)
            if maze[i][j] == 'E':
                end = (i, j)

    if not start or not end:
        return []

    # BFS implementation
    queue = deque([(start, [])])
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) in visited:
            continue

        visited.add((x, y))
        path = path + [(x, y)]

        # Create a copy of the maze for this state
        maze_copy = [row[:] for row in maze]
        for px, py in path:
            if maze_copy[px][py] == '.':
                maze_copy[px][py] = '@'
        maze_states.append(maze_copy)  # Store the current state

        if (x, y) == end:
            # Mark the final path in the maze
            for px, py in path:
                if maze[px][py] == '.':
                    maze[px][py] = 'p'

            # Store final state
            maze_states.append(maze)
            return maze_states

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'E'}:
                queue.append(((nx, ny), path))

    return []

def find_path_dfs(maze):
    """
    Find the path from 'S' to 'E' using Depth-First Search and return all states.

    Args:
        maze (list): The maze represented as a 2D list.

    Returns:
        tuple: (final_maze, maze_states) where final_maze is the solved maze and 
               maze_states is a list of all intermediate states. Returns (None, [])
               if no path exists.
    """
    rows, cols = len(maze), len(maze[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    maze_states = []  # Store all intermediate states

    # Locate 'S' and 'E'
    start = end = None
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 'S':
                start = (i, j)
            if maze[i][j] == 'E':
                end = (i, j)

    if not start or not end:
        return []

    # DFS implementation
    stack = [(start, [])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()

        if (x, y) in visited:
            continue

        visited.add((x, y))
        path = path + [(x, y)]

        # Create a copy of the maze for this state
        maze_copy = [row[:] for row in maze]
        for px, py in path:
            if maze_copy[px][py] == '.':
                maze_copy[px][py] = '@'
        maze_states.append(maze_copy)  # Store the current state

        if (x, y) == end:
            # Mark the final path in the maze
            for px, py in path:
                if maze[px][py] == '.':
                    maze[px][py] = 'p'
            # Store the final maze
            maze_states.append(maze)
            return maze_states

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'E'}:
                stack.append(((nx, ny), path))

    return []