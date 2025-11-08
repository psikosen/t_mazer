"""
GUI module for visualizing maze generation and solving.
"""
import pygame
import numpy as np
import time
import warnings


class MazeGUI:
    """Interactive GUI for visualizing maze generation and solving."""
    
    # Color definitions
    COLORS = {
        'wall': (40, 40, 40),       # Dark gray
        'path': (255, 255, 255),    # White
        'start': (50, 200, 50),     # Green
        'end': (200, 50, 50),       # Red
        'current': (50, 150, 250),  # Blue
        'visited': (180, 180, 250), # Light blue
        'solution': (250, 250, 100), # Yellow
        'background': (20, 20, 20)  # Very dark gray
    }
    
    def __init__(self, cell_size=20, margin=1):
        """
        Initialize the maze GUI.
        
        Args:
            cell_size (int): Size of each cell in pixels
            margin (int): Margin between cells in pixels
        """
        self.cell_size = cell_size
        self.margin = margin
        self.screen = None
        self.clock = None
        self.font = None
        self.width = 0
        self.height = 0
        self.maze = None
        self.is_initialized = False
        
    def initialize(self, maze):
        """
        Initialize the pygame window based on maze dimensions.
        
        Args:
            maze (numpy.ndarray): The maze to display
        """
        self.maze = maze
        height, width = maze.shape
        self.width = width
        self.height = height
        
        # Calculate window size
        window_width = width * (self.cell_size + self.margin) + self.margin
        window_height = height * (self.cell_size + self.margin) + self.margin
        
        # Initialize pygame (safe to call multiple times)
        try:
            # pygame.get_init() is available in pygame 2.0+
            if not pygame.get_init():
                pygame.init()
        except AttributeError:
            # Fallback for older pygame versions
            pygame.init()
        
        # If display already exists, close it first
        if self.screen is not None:
            pygame.display.quit()
        
        # Create new display
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("T_Mazer - Ternary Fair Play Maze Solver")
        self.clock = pygame.time.Clock()
        
        # Initialize font with error handling (optional - not currently used)
        # Font is kept for potential future use (e.g., text labels)
        # Note: Font module may not be available on all pygame installations
        self.font = None
        # Suppress warnings during font initialization attempt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.filterwarnings("ignore", module="pygame.font")
            try:
                # Use getattr to check for font module without triggering errors
                font_module = getattr(pygame, 'font', None)
                if font_module is not None:
                    # Try to get SysFont function
                    sysfont = getattr(font_module, 'SysFont', None)
                    if sysfont is not None:
                        try:
                            self.font = sysfont('Arial', 12)
                        except (NotImplementedError, ImportError, AttributeError, TypeError):
                            pass
            except (NotImplementedError, ImportError, AttributeError, TypeError):
                # Font module not available - this is fine
                pass
        
        self.is_initialized = True
        
    def draw_maze(self, visited=None, solution_path=None, current_pos=None):
        """
        Draw the maze with optional solver visualization.
        
        Args:
            visited (set): Set of visited positions
            solution_path (list): Current solution path
            current_pos (tuple): Current solver position
        """
        if not self.is_initialized:
            raise RuntimeError("GUI not initialized. Call initialize() first.")
            
        # Fill background
        self.screen.fill(self.COLORS['background'])
        
        # Convert to sets/lists if None
        visited = visited if visited else set()
        solution_path = solution_path if solution_path else []
        
        # Draw each cell
        for y in range(self.height):
            for x in range(self.width):
                # Calculate position
                rect_x = x * (self.cell_size + self.margin) + self.margin
                rect_y = y * (self.cell_size + self.margin) + self.margin
                rect = pygame.Rect(rect_x, rect_y, self.cell_size, self.cell_size)
                
                # Determine cell color
                if (y, x) == (1, 1):  # Start position
                    color = self.COLORS['start']
                elif (y, x) == (self.height - 2, self.width - 2):  # End position
                    color = self.COLORS['end']
                elif current_pos and (y, x) == current_pos:  # Current position
                    color = self.COLORS['current']
                elif (y, x) in solution_path:  # Solution path
                    color = self.COLORS['solution']
                elif (y, x) in visited:  # Visited position
                    color = self.COLORS['visited']
                elif self.maze[y, x] == 1:  # Wall
                    color = self.COLORS['wall']
                else:  # Path
                    color = self.COLORS['path']
                
                # Draw the cell
                pygame.draw.rect(self.screen, color, rect)
        
        # Update display
        pygame.display.flip()
        
    def visualize_solving(self, maze, solver, delay=0.1):
        """
        Visualize the maze-solving process in real-time.
        
        Args:
            maze (numpy.ndarray): The maze to solve
            solver (MazeSolver): The maze solver to use
            delay (float): Delay between steps in seconds
            
        Returns:
            list: Final solution path
            list: Reasoning tokens used
        """
        # Initialize GUI if needed (check shape instead of object identity)
        if (not self.is_initialized or 
            self.maze is None or 
            self.maze.shape != maze.shape):
            self.initialize(maze)
        
        # Get the full solution first
        solution_path, reasoning_tokens = solver.solve(maze)
        
        # Reset for visualization
        visited = set()
        
        # Initial display - show empty maze
        self.draw_maze(visited, [], None)
        time.sleep(delay)
        
        # Visualize the solution path step by step
        for i, pos in enumerate(solution_path):
            current_pos = pos
            visited.add(pos)
            
            # Display current state
            self.draw_maze(visited, solution_path[:i+1], current_pos)
            
            # Process any pending events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None, None
            
            # Pause to make visualization visible
            time.sleep(delay)
        
        # Final pause to show completed solution
        time.sleep(delay * 2)
        
        return solution_path, reasoning_tokens
        
    def close(self):
        """Close the GUI and clean up resources."""
        if self.is_initialized:
            try:
                if self.screen is not None:
                    pygame.display.quit()
                pygame.quit()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.screen = None
                self.clock = None
                self.font = None
                self.is_initialized = False
