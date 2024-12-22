import pygame
import time

class MazeVisualizer:
    def __init__(self, maze_states):
        """
        Initialize the MazeVisualizer object.

        Parameters:
        - maze_states: List of 2D arrays representing different states of the maze.
        - cell_size: Size of each cell in pixels (default is 40).

        Colors are predefined for the grid, path, walls, start, end, visited path, and actual path.
        """
        pygame.init()
        self.maze_states = maze_states
        self.grid_color = (200, 200, 200)  # Grid line color
        self.path_color = (216, 220, 227)  # Path color
        self.wall_color = (50, 50, 50)     # Wall color
        self.start_color = (0, 0, 255)     # Start point color
        self.end_color = (255, 0, 0)       # End point color
        self.visited_path_color = (153, 149, 255) # Visiting Nodes
        self.actual_path_color = (0, 207, 39) # Actual path color

        # Determine the dimensions of the maze
        self.rows = len(maze_states[0])
        self.cols = len(maze_states[0][0])
        self.cell_size = 800 / self.rows
        self.screen_width = self.cols * self.cell_size
        self.screen_height = self.rows * self.cell_size
        # Initialize the Pygame screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Visualization")

    def draw_maze(self, maze):
        """
        Draw the maze grid on the screen based on the current state.

        Parameters:
        - maze: 2D array representing the current state of the maze.

        Each cell's color is determined based on its value:
        - "#" for walls
        - "." for paths
        - "S" for the start point
        - "E" for the end point
        - "@" for visited paths
        - "p" for the actual path.
        """
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * self.cell_size  # Top-left x-coordinate of the cell
                y = row * self.cell_size  # Top-left y-coordinate of the cell

                # Determine the color based on the cell value
                if maze[row][col] == "#":
                    color = self.wall_color
                elif maze[row][col] == ".":
                    color = self.path_color
                elif maze[row][col] == "S":
                    color = self.start_color
                elif maze[row][col] == "E":
                    color = self.end_color
                elif maze[row][col] == "@":
                    color = self.visited_path_color
                elif maze[row][col] == "p":
                    color = self.actual_path_color

                # Draw the cell
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
                # Draw the grid border for the cell
                pygame.draw.rect(self.screen, self.grid_color, (x, y, self.cell_size, self.cell_size), 1)

    def visualize_simple(self):
        """
        Visualize the maze in its final state.

        This function displays the last state of the maze and runs continuously
        until the user closes the Pygame window.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear the screen and redraw the maze
            self.screen.fill(self.grid_color)
            self.draw_maze(self.maze_states[-1])  # Display the final state of the maze
            pygame.display.flip()

        # Quit Pygame
        pygame.quit()

    def visualize_step_by_step(self, step_delay=0.5):
        """
        Visualize the maze states step by step.

        Parameters:
        - step_delay: Time delay (in seconds) between consecutive states (default is 0.5).

        Each state of the maze is displayed sequentially with a delay,
        allowing for visualization of the maze-solving process.
        """
        step_index = 0  
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  
                    pygame.quit()
                    return

            if step_index < len(self.maze_states): 
                self.screen.fill(self.grid_color)  
                self.draw_maze(self.maze_states[step_index])  
                pygame.display.flip()  
                time.sleep(step_delay) 
                step_index += 1

# visualizer = MazeVisualizer(maze_states)
# visualizer.visualize_simple()
# visualizer.visualize_step_by_step(step_delay=1)
