import random

def generate_chessboard(n):
    """
    Generates an n x n chessboard with n queens placed randomly.
    
    :param n: Size of the chessboard (n x n) and number of queens
    :return: A 2D list representing the chessboard
    """
    # Initialize the chessboard with empty positions (0)
    chessboard = [[0 for _ in range(n)] for _ in range(n)]
    
    # Randomly place n queens on the board (one per column)
    for col in range(n):
        row = random.randint(0, n - 1)
        chessboard[row][col] = 1  # Place a queen (1)
    
    return chessboard

def print_chessboard(chessboard):
    """
    Prints the chessboard in a readable format.
    
    :param chessboard: A 2D list representing the chessboard
    """
    for row in chessboard:
        print(' '.join(map(str, row)))
    print()

def fitness(chessboard):
    """
    Calculates the fitness of the chessboard (number of conflicts/attacks between queens).
    
    :param chessboard: A 2D list representing the chessboard
    :return: Number of conflicts (lower is better)
    """
    n = len(chessboard)
    conflicts = 0
    
    # Find all queen positions
    queens = [(i, j) for i in range(n) for j in range(n) if chessboard[i][j] == 1]
    
    # Check for conflicts between queens
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            x1, y1 = queens[i]
            x2, y2 = queens[j]
            # Check same row or same diagonal
            if x1 == x2 or abs(x1 - x2) == abs(y1 - y2):
                conflicts += 1
    
    return conflicts

def generate_neighbors(chessboard):
    """
    Generates all possible neighboring states by moving one queen at a time within its column.
    
    :param chessboard: A 2D list representing the chessboard
    :return: A list of neighboring chessboards
    """
    n = len(chessboard)
    neighbors = []
    
    # Find all queen positions
    queens = [(i, j) for i in range(n) for j in range(n) if chessboard[i][j] == 1]
    
    # Generate neighbors by moving each queen to a new position in its column
    for idx, (i, j) in enumerate(queens):
        for new_i in range(n):
            if new_i != i:
                new_board = [row.copy() for row in chessboard]
                new_board[i][j] = 0  # Remove queen from old position
                new_board[new_i][j] = 1  # Place queen in new position
                neighbors.append(new_board)
    
    return neighbors

def hill_climbing(n):
    """
    Solves the N-Queens problem using hill-climbing.
    
    :param n: Size of the chessboard (n x n) and number of queens
    :return: Path taken, whether a solution was found, and final fitness value
    """
    # Generate initial state
    current_board = generate_chessboard(n)
    current_fitness = fitness(current_board)
    path = [current_board]
    
    while True:
        # Generate all neighbors
        neighbors = generate_neighbors(current_board)
        
        # Evaluate fitness of all neighbors
        best_neighbor = None
        best_fitness = current_fitness
        
        for neighbor in neighbors:
            neighbor_fitness = fitness(neighbor)
            if neighbor_fitness < best_fitness:
                best_neighbor = neighbor
                best_fitness = neighbor_fitness
        
        # If no better neighbor, stop
        if best_fitness >= current_fitness:
            break
        
        # Move to the best neighbor
        current_board = best_neighbor
        current_fitness = best_fitness
        path.append(current_board)
    
    # Check if a solution was found (no conflicts)
    solution_found = current_fitness == 0
    return path, solution_found, current_fitness

