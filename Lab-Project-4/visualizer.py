import pygame
import sys
import time

class NQueensVisualizer:
    def __init__(self, states, window_size=600, time_step=1, queen_image_path="queen.png"):
        """
        Initialize the visualizer.

        :param states: List of nxn states, each representing a board configuration.
        :param window_size: Fixed size of the Pygame window (default: 600x600).
        :param time_step: Time to display each state in seconds (default: 1).
        :param queen_image_path: Path to the queen image file (default: "queen.png").
        """
        self.states = states
        self.time_step = time_step
        self.n = len(states[0])  # Assuming all states are nxn

        # Initialize Pygame
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("N-Queens Visualizer")

        # Calculate cell size based on window size and n
        self.cell_size = self.window_size // self.n

        # Load the queen image
        self.queen_image = pygame.image.load("./assets/queen.png")
        # Scale the queen image to fit the cell size
        self.queen_image = pygame.transform.scale(
            self.queen_image, (self.cell_size, self.cell_size)
        )

        # Colors (green and white like Chess.com)
        self.GREEN = (118, 150, 86)  # Dark green for chessboard
        self.WHITE = (238, 238, 210)  # Light white for chessboard

    def draw_board(self, state):
        """Draw the chessboard and queens based on the current state."""
        self.screen.fill(self.WHITE)
        for row in range(self.n):
            for col in range(self.n):
                # Alternate colors for the chessboard
                color = self.WHITE if (row + col) % 2 == 0 else self.GREEN
                pygame.draw.rect(
                    self.screen,
                    color,
                    (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size),
                )

                # Draw the queen image if the cell is occupied
                if state[row][col] == 1:
                    self.screen.blit(
                        self.queen_image,
                        (col * self.cell_size, row * self.cell_size),
                    )

    def run(self):
        """Run the visualizer, displaying each state in the list."""
        for state in self.states:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.draw_board(state)
            pygame.display.flip()
            time.sleep(self.time_step)

        # Keep the window open after the last state
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()