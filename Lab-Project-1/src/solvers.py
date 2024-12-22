from collections import deque

def reconstruct_path(parent_dict, meeting_point):
    """
    Reconstruct the path from the meeting point using the parent dictionary.
    
    Args:
        parent_dict (dict): The parent dictionary that stores the parent of each node.
        meeting_point (tuple): The node where the two searches meet.
    
    Returns:
        list: The reconstructed path as a list of coordinates.
    """
    path = []
    current = meeting_point
    while current is not None:
        path.append(current)
        current = parent_dict.get(current)
    return path[::-1]  # Reverse to get the correct order

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
            return len(visited), maze_states

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
            return len(visited), maze_states

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'E'}:
                stack.append(((nx, ny), path))

    return []

def find_path_bfs_bidirectional(maze):
    """
    Find the path from 'S' to 'E' using Bidirectional Breadth-First Search and return all states.

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
        return None, []

    # Bidirectional BFS initialization
    queue_s = deque([(start, [])])
    queue_e = deque([(end, [])])
    visited_s = {start}
    visited_e = {end}

    # Track parents for both searches
    parent_s = {start: None}
    parent_e = {end: None}

    # Keep track of the intermediate states
    visited_nodes_s = set()
    visited_nodes_e = set()

    while queue_s and queue_e:
        # Expand from the start side
        (x_s, y_s), path_s = queue_s.popleft()

        # Check for intersection
        if (x_s, y_s) in visited_e:
            # Reconstruct path
            path = path_s + reconstruct_path(parent_e, (x_s, y_s))
            break

        # Expand the start side search
        for dx, dy in directions:
            nx, ny = x_s + dx, y_s + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'E'} and (nx, ny) not in visited_s:
                queue_s.append(((nx, ny), path_s + [(nx, ny)]))
                visited_s.add((nx, ny))
                parent_s[(nx, ny)] = (x_s, y_s)

        # Expand from the end side
        (x_e, y_e), path_e = queue_e.popleft()

        # Check for intersection
        if (x_e, y_e) in visited_s:
            # Reconstruct path
            path = reconstruct_path(parent_s, (x_e, y_e)) + path_e
            break

        # Expand the end side search
        for dx, dy in directions:
            nx, ny = x_e + dx, y_e + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'S'} and (nx, ny) not in visited_e:
                queue_e.append(((nx, ny), path_e + [(nx, ny)]))
                visited_e.add((nx, ny))
                parent_e[(nx, ny)] = (x_e, y_e)

        # Create state for visualization after each expansion
        maze_copy = [row[:] for row in maze]
        for px, py in visited_s:
            if maze_copy[px][py] == '.':
                maze_copy[px][py] = '@'
        for px, py in visited_e:
            if maze_copy[px][py] == '.':
                maze_copy[px][py] = '@'
        maze_states.append(maze_copy)

    else:
        # No path found
        return None, []

    # Mark the final path in the maze for visualization
    for x, y in path:
        if maze[x][y] == '.':
            maze[x][y] = 'p'

    # Store the final maze state
    maze_copy = [row[:] for row in maze]
    maze_states.append(maze_copy)

    return len(visited_s) + len(visited_e), maze_states

def find_path_dfs_bidirectional(maze):
    """
    Find the path from 'S' to 'E' using Bidirectional Depth-First Search and return all states.

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
        return None, []

    # Bidirectional DFS initialization
    stack_s = [(start, [])]  # Stack for search from start
    stack_e = [(end, [])]    # Stack for search from end
    visited_s = {start}      # Visited set for search from start
    visited_e = {end}        # Visited set for search from end
    parent_s = {start: None}  # Parent dictionary for path reconstruction from start
    parent_e = {end: None}    # Parent dictionary for path reconstruction from end

    # While both stacks are not empty
    while stack_s and stack_e:
        # Expand from the start side
        (x_s, y_s), path_s = stack_s.pop()
        
        # Check if intersection with end side
        if (x_s, y_s) in visited_e:
            # Reconstruct path from start to intersection
            path = path_s + reconstruct_path(parent_e, (x_s, y_s))
            break

        # Explore neighbors in DFS manner (LIFO)
        for dx, dy in directions:
            nx, ny = x_s + dx, y_s + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'E'} and (nx, ny) not in visited_s:
                stack_s.append(((nx, ny), path_s + [(nx, ny)]))
                visited_s.add((nx, ny))
                parent_s[(nx, ny)] = (x_s, y_s)

        # Expand from the end side
        (x_e, y_e), path_e = stack_e.pop()

        # Check if intersection with start side
        if (x_e, y_e) in visited_s:
            # Reconstruct path from end to intersection
            path = reconstruct_path(parent_s, (x_e, y_e)) + path_e
            break

        # Explore neighbors in DFS manner (LIFO)
        for dx, dy in directions:
            nx, ny = x_e + dx, y_e + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] in {'.', 'S'} and (nx, ny) not in visited_e:
                stack_e.append(((nx, ny), path_e + [(nx, ny)]))
                visited_e.add((nx, ny))
                parent_e[(nx, ny)] = (x_e, y_e)

        # Create state for visualization after each expansion
        maze_copy = [row[:] for row in maze]
        for px, py in visited_s:
            if maze_copy[px][py] == '.':
                maze_copy[px][py] = '@'
        for px, py in visited_e:
            if maze_copy[px][py] == '.':
                maze_copy[px][py] = '@'
        maze_states.append(maze_copy)

    else:
        # No path found
        return None, []

    # Mark the final path in the maze for visualization
    for x, y in path:
        if maze[x][y] == '.':
            maze[x][y] = 'p'

    # Store the final maze state
    maze_copy = [row[:] for row in maze]
    maze_states.append(maze_copy)

    return len(visited_s) + len(visited_e), maze_states