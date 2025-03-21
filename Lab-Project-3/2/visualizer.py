import pygame
import sys
import time

# Constants
WINDOW_SIZE = 600
TILE_SIZE = WINDOW_SIZE // 3
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
TILE_COLOR = (214, 217, 54) 
GRID_COLOR = (50, 50, 50)  # Dark gray for grid lines
SHADOW_COLOR = (0, 0, 0, 100)  # Semi-transparent black for shadow


class PuzzleVisualizer:
    def __init__(self, solution_path):
        """
        Initialize the visualizer with the solution path.
        :param solution_path: List of 3x3 states representing the solution path.
        """
        self.solution_path = solution_path
        self.screen = None
        self.clock = None
        self.font = None
        self._initialize_pygame()

    def _initialize_pygame(self):
        """Initialize Pygame and set up the display."""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("8-Puzzle Visualizer")
        self.clock = pygame.time.Clock()

        # Load a custom font (replace 'arial.ttf' with your font file)
        self.font = pygame.font.Font("./fonts/Knight-warrior.otf", 100)  # Default font if custom font is not found

    def draw_puzzle(self, state):
        """
        Draw the puzzle state on the screen.
        :param state: Current state of the puzzle (3x3 grid as a list of lists).
        """
        if len(state) != 3 or any(len(row) != 3 for row in state):
            raise ValueError(f"Invalid state: {state}. A state must be a 3x3 grid.")

        self.screen.fill(WHITE)

        # Draw grid lines
        for i in range(1, 3):
            pygame.draw.line(self.screen, GRID_COLOR, (i * TILE_SIZE, 0), (i * TILE_SIZE, WINDOW_SIZE), 4)
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * TILE_SIZE), (WINDOW_SIZE, i * TILE_SIZE), 4)

        # Draw tiles
        for i in range(3):
            for j in range(3):
                tile_value = state[i][j]
                if tile_value != 0:
                    # Draw tile background with shadow
                    tile_rect = pygame.Rect(j * TILE_SIZE + 5, i * TILE_SIZE + 5, TILE_SIZE - 10, TILE_SIZE - 10)
                    pygame.draw.rect(self.screen, SHADOW_COLOR, tile_rect.move(2, 2))  # Shadow
                    pygame.draw.rect(self.screen, TILE_COLOR, tile_rect)  # Tile

                    # Draw tile value
                    text = self.font.render(str(tile_value), True, BLACK)
                    text_rect = text.get_rect(center=(j * TILE_SIZE + TILE_SIZE // 2, i * TILE_SIZE + TILE_SIZE // 2))
                    self.screen.blit(text, text_rect)

        pygame.display.flip()

    def visualize(self, time_step):
        """
        Visualize the solution path with a given time step.
        :param time_step: Delay between states (in seconds).
        """
        for state in self.solution_path:
            try:
                self.draw_puzzle(state)
                time.sleep(time_step)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._quit_pygame()
            except ValueError as e:
                print(f"Skipping invalid state: {e}")
                continue

    def _quit_pygame(self):
        """Quit Pygame and exit the program."""
        pygame.quit()
        sys.exit()

    def keep_window_open(self):
        """
        Keep the Pygame window open until the user closes it.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.clock.tick(FPS)
        self._quit_pygame()