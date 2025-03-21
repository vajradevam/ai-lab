import random

def generate_maze(rows=10, cols=10, sparsity=0.2):
    """
    Generates a maze with the given dimensions and sparsity.
    
    :param rows: Number of rows in the maze (default: 10).
    :param cols: Number of columns in the maze (default: 10).
    :param sparsity: Probability of a cell being a wall (default: 0.2).
    :return: A 2D list representing the maze.
    """
    # Initialize the maze with paths (.)
    maze = [['.' for _ in range(cols)] for _ in range(rows)]
    
    # Randomly place walls (#) based on sparsity
    for i in range(rows):
        for j in range(cols):
            if random.random() < sparsity:
                maze[i][j] = '#'
    
    # Function to find a random position that is not a wall
    def get_random_position():
        while True:
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            if maze[row][col] != '#':
                return (row, col)
    
    # Place the start (S) and goal (G) positions
    start = get_random_position()
    goal = get_random_position()
    
    # Ensure start and goal are different
    while start == goal:
        goal = get_random_position()
    
    # Mark the start and goal positions
    maze[start[0]][start[1]] = 'S'
    maze[goal[0]][goal[1]] = 'G'
    
    return maze

def print_maze(maze):
    """
    Prints the maze in a readable format.
    
    :param maze: A 2D list representing the maze.
    """
    for row in maze:
        print(' '.join(row))