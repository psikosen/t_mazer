#!/usr/bin/env python3
"""
T_Mazer - Main executable with all versions included.
"""
import sys
import os
import time
import random
import argparse

# Try to import pygame and numpy, but handle gracefully if they're not available
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ===== SIMPLE VERSION (NO DEPENDENCIES) =====

class SimpleMaze:
    def __init__(self, width=20, height=20, complexity=0.7, density=0.7):
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
        self.maze = [[0 for _ in range(width)] for _ in range(height)]
    
    def generate(self):
        # Create borders
        for x in range(self.width):
            self.maze[0][x] = 1
            self.maze[self.height-1][x] = 1
        for y in range(self.height):
            self.maze[y][0] = 1
            self.maze[y][self.width-1] = 1
        
        # Add random walls
        wall_count = int((self.width * self.height) * self.complexity * self.density)
        for _ in range(wall_count):
            x = random.randint(1, self.width-2)
            y = random.randint(1, self.height-2)
            # Don't block start or end
            if (y, x) != (1, 1) and (y, x) != (self.height-2, self.width-2):
                self.maze[y][x] = 1
        
        # Ensure a path exists
        # Simple horizontal and vertical paths
        mid_y = self.height // 2
        for x in range(1, self.width-1):
            self.maze[mid_y][x] = 0
        
        # Path from start to horizontal path
        for y in range(1, mid_y+1):
            self.maze[y][1] = 0
        
        # Path from horizontal to end
        for y in range(mid_y, self.height-1):
            self.maze[y][self.width-2] = 0
        
        return self.maze

class SimpleSolver:
    def __init__(self):
        self.solution = []
    
    def solve(self, maze, width, height):
        # Simple BFS algorithm
        start = (1, 1)
        end = (height-2, width-2)
        
        queue = [start]
        visited = {start}
        parents = {start: None}
        
        while queue:
            y, x = queue.pop(0)
            
            if (y, x) == end:
                # Reconstruct path
                path = []
                current = (y, x)
                while current != start:
                    path.append(current)
                    current = parents[current]
                
                path.reverse()
                self.solution = [start] + path
                return self.solution
            
            # Try all four directions
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                
                # Check if valid move
                if (0 <= ny < height and 
                    0 <= nx < width and 
                    maze[ny][nx] == 0 and
                    (ny, nx) not in visited):
                    queue.append((ny, nx))
                    visited.add((ny, nx))
                    parents[(ny, nx)] = (y, x)
        
        return []  # No solution found

def display_simple_maze(maze, path=None, current=None):
    height = len(maze)
    width = len(maze[0])
    
    # Convert path to set for quick lookup
    path_set = set(path) if path else set()
    
    # Characters for display
    wall_char = "██"
    path_char = "  "
    start_char = "S "
    end_char = "E "
    solution_char = "● "
    current_char = "◉ "
    
    # Top border
    print("┌" + "──" * width + "┐")
    
    for y in range(height):
        print("│", end="")
        for x in range(width):
            if current and (y, x) == current:
                print(current_char, end="")
            elif (y, x) == (1, 1):
                print(start_char, end="")
            elif (y, x) == (height-2, width-2):
                print(end_char, end="")
            elif (y, x) in path_set:
                print(solution_char, end="")
            elif maze[y][x] == 1:
                print(wall_char, end="")
            else:
                print(path_char, end="")
        print("│")
    
    # Bottom border
    print("└" + "──" * width + "┘")

def run_simple_version(args):
    print(f"T_Mazer - Simple Version (No Dependencies)")
    print(f"Generating maze with dimensions {args.width}x{args.height}...")
    
    # Generate maze
    maze_gen = SimpleMaze(args.width, args.height, args.complexity, args.density)
    maze = maze_gen.generate()
    
    # Show initial maze
    print("\nInitial Maze:")
    display_simple_maze(maze)
    
    # Solve maze
    print("\nSolving maze...")
    solver = SimpleSolver()
    solution = solver.solve(maze, args.width, args.height)
    
    if not solution:
        print("No solution found!")
        return
    
    # Display solution step by step
    print(f"\nSolution found! Path length: {len(solution)}")
    print("Visualizing solution (press Ctrl+C to stop)...")
    
    try:
        for i, pos in enumerate(solution):
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"Step {i+1}/{len(solution)}")
            display_simple_maze(maze, solution[:i+1], pos)
            time.sleep(args.delay)
        
        print("\nMaze solved!")
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")

# ===== NUMPY VERSION =====

if HAS_NUMPY:
    class NumpyMazeGenerator:
        def __init__(self, width=20, height=20, complexity=0.7, density=0.7):
            self.width = width
            self.height = height
            self.complexity = complexity
            self.density = density
        
        def generate(self):
            # Initialize maze with all walls
            shape = (self.height, self.width)
            maze = np.ones(shape, dtype=np.int8)
            
            # Define a recursive function to carve passages
            def carve_passages(x, y):
                # Mark current cell as path
                maze[y, x] = 0
                
                # Define possible directions (right, down, left, up)
                directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
                random.shuffle(directions)
                
                # Try each direction
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    
                    # Check if the neighbor is valid and unvisited
                    if (0 <= nx < self.width and 0 <= ny < self.height and maze[ny, nx] == 1):
                        # Carve passage by setting the wall between current and neighbor to path
                        maze[y + dy//2, x + dx//2] = 0
                        
                        # Recursively carve from the neighbor
                        carve_passages(nx, ny)
            
            # Start carving from a random even coordinate
            start_x = random.randint(0, (self.width - 1) // 2) * 2 + 1
            start_y = random.randint(0, (self.height - 1) // 2) * 2 + 1
            
            # Ensure the starting point is valid
            if start_x >= self.width:
                start_x = self.width - 2
            if start_y >= self.height:
                start_y = self.height - 2
                
            # Start the recursive carving
            carve_passages(start_x, start_y)
            
            # Create entry and exit
            maze[1, 1] = 0  # Entry
            maze[self.height - 2, self.width - 2] = 0  # Exit
            
            # Make sure borders are walls
            maze[0, :] = maze[self.height - 1, :] = 1
            maze[:, 0] = maze[:, self.width - 1] = 1
            
            # Ensure there's a path
            if not self.ensure_solvable(maze):
                # Create a direct path if needed
                y1, x1 = 1, 1
                y2, x2 = self.height - 2, self.width - 2
                
                # Horizontal path
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    maze[y1, x] = 0
                
                # Vertical path
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    maze[y, x2] = 0
            
            return maze
        
        def ensure_solvable(self, maze):
            start = (1, 1)
            end = (maze.shape[0] - 2, maze.shape[1] - 2)
            
            # BFS to find a path
            queue = [start]
            visited = {start}
            
            while queue:
                current = queue.pop(0)
                
                if current == end:
                    return True
                
                # Try all four directions
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = current[0] + dy, current[1] + dx
                    
                    # Check if the move is valid
                    if (0 <= ny < maze.shape[0] and 
                        0 <= nx < maze.shape[1] and 
                        maze[ny, nx] == 0 and
                        (ny, nx) not in visited):
                        queue.append((ny, nx))
                        visited.add((ny, nx))
            
            # No path found
            return False
    
    class NumpyMazeSolver:
        """Maze solver using NumPy arrays."""
        
        def __init__(self):
            self.solution_path = []
            self.visited = set()
        
        def solve(self, maze):
            self.solution_path = []
            self.visited = set()
            
            start = (1, 1)
            end = (maze.shape[0] - 2, maze.shape[1] - 2)
            
            # BFS to find shortest path
            queue = [start]
            self.visited.add(start)
            parents = {start: None}
            
            while queue:
                current = queue.pop(0)
                
                if current == end:
                    # Path found, reconstruct it
                    path = []
                    while current != start:
                        path.append(current)
                        current = parents[current]
                    
                    # Reverse path and add start
                    path.reverse()
                    self.solution_path = [start] + path
                    return self.solution_path, ["[Step]"] * len(self.solution_path)
                
                # Try all four directions
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = current[0] + dy, current[1] + dx
                    
                    # Check if move is valid
                    if (0 <= ny < maze.shape[0] and 
                        0 <= nx < maze.shape[1] and 
                        maze[ny, nx] == 0 and
                        (ny, nx) not in self.visited):
                        queue.append((ny, nx))
                        self.visited.add((ny, nx))
                        parents[(ny, nx)] = current
            
            return [], []  # No path found

    def display_numpy_maze(maze, path=None, current_pos=None):
        """Display a NumPy maze in text format."""
        height, width = maze.shape
        
        # Convert path to a set for quick lookup
        path_set = set(path) if path else set()
        
        # Characters for display
        wall_char = "██"
        path_char = "  "
        start_char = "S "
        end_char = "E "
        solution_char = "● "
        current_char = "◉ "
        
        # Top border
        print("┌" + "──" * width + "┐")
        
        for y in range(height):
            print("│", end="")
            for x in range(width):
                if current_pos and (y, x) == current_pos:
                    print(current_char, end="")
                elif (y, x) == (1, 1):
                    print(start_char, end="")
                elif (y, x) == (height - 2, width - 2):
                    print(end_char, end="")
                elif (y, x) in path_set:
                    print(solution_char, end="")
                elif maze[y, x] == 1:
                    print(wall_char, end="")
                else:
                    print(path_char, end="")
            print("│")
        
        # Bottom border
        print("└" + "──" * width + "┘")

    def run_numpy_version(args):
        print(f"T_Mazer - NumPy Version (Text-based)")
        print(f"Generating maze with dimensions {args.width}x{args.height}...")
        
        # Generate maze
        generator = NumpyMazeGenerator(
            width=args.width,
            height=args.height,
            complexity=args.complexity,
            density=args.density
        )
        maze = generator.generate()
        
        # Show initial maze
        print("\nInitial Maze:")
        display_numpy_maze(maze)
        
        # Solve maze
        print("\nSolving maze...")
        solver = NumpyMazeSolver()
        solution_path, _ = solver.solve(maze)
        
        if not solution_path:
            print("No solution found!")
            return
        
        # Display solution step by step
        print(f"\nSolution found! Path length: {len(solution_path)}")
        print("Visualizing solution (press Ctrl+C to stop)...")
        
        try:
            for i, pos in enumerate(solution_path):
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"Step {i+1}/{len(solution_path)}")
                display_numpy_maze(maze, solution_path[:i+1], pos)
                time.sleep(args.delay)
            
            print("\nMaze solved!")
        except KeyboardInterrupt:
            print("\nVisualization stopped by user.")

# ===== PYGAME VERSION =====

if HAS_PYGAME and HAS_NUMPY:
    class PyGameMazeGUI:
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
            """Initialize the pygame window based on maze dimensions."""
            self.maze = maze
            height, width = maze.shape
            self.width = width
            self.height = height
            
            # Calculate window size
            window_width = width * (self.cell_size + self.margin) + self.margin
            window_height = height * (self.cell_size + self.margin) + self.margin
            
            # Initialize pygame
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("T_Mazer - Ternary Fair Play Maze Solver")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 12)
            self.is_initialized = True
            
        def draw_maze(self, visited=None, solution_path=None, current_pos=None):
            """Draw the maze with optional solver visualization."""
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
            """Visualize the maze-solving process in real-time."""
            # Initialize GUI if needed
            if not self.is_initialized or self.maze is not maze:
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
                        pygame.quit()
                        return None, None
                
                # Pause to make visualization visible
                time.sleep(delay)
            
            # Final pause to show completed solution
            time.sleep(delay * 2)
            
            return solution_path, reasoning_tokens
            
        def close(self):
            """Close the GUI and clean up resources."""
            if self.is_initialized:
                pygame.quit()
                self.is_initialized = False

    def run_pygame_version(args):
        """Run the full pygame version of T_Mazer."""
        # First import our specially designed classes from the modules
        from src.maze_generator import MazeGenerator
        from src.maze_solver import MazeSolver
        
        print("T_Mazer - Full Graphical Version")
        print(f"Generating maze with dimensions {args.width}x{args.height}...")
        
        # Initialize maze generator
        generator = NumpyMazeGenerator(
            width=args.width,
            height=args.height,
            complexity=args.complexity,
            density=args.density
        )
        
        # Generate a maze
        maze = generator.generate()
        print(f"Maze generated with dimensions {args.width}x{args.height}")
        
        # Initialize solver
        solver = NumpyMazeSolver()
        
        # Initialize GUI
        gui = PyGameMazeGUI(cell_size=args.cell_size)
        gui.initialize(maze)
        
        # Visualize solving process
        print("Visualizing maze solving...")
        solution_path, reasoning_tokens = gui.visualize_solving(
            maze, solver, delay=args.delay
        )
        
        if solution_path is None:
            print("Visualization was terminated by the user")
            return
        
        print(f"Maze solved! Solution path length: {len(solution_path)}")
        
        # Keep the window open until the user closes it
        print("Close the window to exit")
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                time.sleep(0.1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            gui.close()

# ===== MAIN APPLICATION =====

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="T_Mazer - Ternary Fair Play Maze Solver")
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=20,
        help="Width of the maze (default: 20)"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=20,
        help="Height of the maze (default: 20)"
    )
    
    parser.add_argument(
        "--complexity", 
        type=float, 
        default=0.7,
        help="Maze complexity factor 0.0-1.0 (default: 0.7)"
    )
    
    parser.add_argument(
        "--density", 
        type=float, 
        default=0.7,
        help="Maze density factor 0.0-1.0 (default: 0.7)"
    )
    
    parser.add_argument(
        "--cell-size", 
        type=int, 
        default=30,
        help="Cell size in pixels (default: 30)"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.1,
        help="Delay between visualization steps in seconds (default: 0.1)"
    )
    
    parser.add_argument(
        "--force-text", 
        action="store_true",
        help="Force text mode even if pygame is available"
    )
    
    parser.add_argument(
        "--force-simple", 
        action="store_true",
        help="Force simple mode (no dependencies)"
    )
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    print("\nT_Mazer - Ternary Fair Play Maze Solver\n")
    
    # Parse command line arguments
    args = parse_args()
    
    # Determine which version to run
    if args.force_simple:
        print("Using Simple version (forced by --force-simple)")
        run_simple_version(args)
    elif args.force_text:
        if HAS_NUMPY:
            print("Using Text version (forced by --force-text)")
            run_numpy_version(args)
        else:
            print("Cannot run Text version without NumPy. Falling back to Simple version.")
            run_simple_version(args)
    elif HAS_PYGAME and HAS_NUMPY:
        print("Using Graphical version (pygame available)")
        run_pygame_version(args)
    elif HAS_NUMPY:
        print("Using Text version (pygame not available)")
        run_numpy_version(args)
    else:
        print("Using Simple version (no dependencies available)")
        run_simple_version(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
