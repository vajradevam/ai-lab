import heapq

# Define the goal state of the 8-puzzle
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Helper function to find the position of the empty tile (0)
def find_empty_tile(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                return (i, j)
    return None

# Helper function to calculate the number of misplaced tiles
def misplaced_tiles_heuristic(puzzle):
    count = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != GOAL_STATE[i][j] and puzzle[i][j] != 0:
                count += 1
    return count

# Helper function to calculate the Manhattan distance
def manhattan_distance_heuristic(puzzle):
    distance = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                goal_row, goal_col = divmod(puzzle[i][j] - 1, 3)
                distance += abs(i - goal_row) + abs(j - goal_col)
    return distance

# Function to generate possible moves from the current state
def generate_moves(puzzle):
    moves = []
    empty_row, empty_col = find_empty_tile(puzzle)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for dr, dc in directions:
        new_row, new_col = empty_row + dr, empty_col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_puzzle = [row[:] for row in puzzle]  # Create a copy of the puzzle
            new_puzzle[empty_row][empty_col], new_puzzle[new_row][new_col] = new_puzzle[new_row][new_col], new_puzzle[empty_row][empty_col]
            moves.append(new_puzzle)
    return moves

# A* algorithm implementation
def a_star(puzzle, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, 0, puzzle, []))  # (f, g, puzzle, path)
    closed_set = set()
    nodes_expanded = 0

    while open_list:
        f, g, current_puzzle, path = heapq.heappop(open_list)
        nodes_expanded += 1

        if current_puzzle == GOAL_STATE:
            return path + [current_puzzle], nodes_expanded, g  # Return solution path, nodes expanded, and path cost

        closed_set.add(tuple(map(tuple, current_puzzle)))

        for move in generate_moves(current_puzzle):
            if tuple(map(tuple, move)) not in closed_set:
                h = manhattan_distance_heuristic(move) if heuristic == "manhattan" else misplaced_tiles_heuristic(move)
                heapq.heappush(open_list, (g + 1 + h, g + 1, move, path + [current_puzzle]))

    return None, nodes_expanded, 0  # If no solution is found

# Greedy Best-First Search implementation
def greedy_bfs(puzzle, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, puzzle, []))  # (h, puzzle, path)
    closed_set = set()
    nodes_expanded = 0

    while open_list:
        h, current_puzzle, path = heapq.heappop(open_list)
        nodes_expanded += 1

        if current_puzzle == GOAL_STATE:
            return path + [current_puzzle], nodes_expanded, len(path)  # Return solution path, nodes expanded, and path cost

        closed_set.add(tuple(map(tuple, current_puzzle)))

        for move in generate_moves(current_puzzle):
            if tuple(map(tuple, move)) not in closed_set:
                h_value = manhattan_distance_heuristic(move) if heuristic == "manhattan" else misplaced_tiles_heuristic(move)
                heapq.heappush(open_list, (h_value, move, path + [current_puzzle]))

    return None, nodes_expanded, 0  # If no solution is found



