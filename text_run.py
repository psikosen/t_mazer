#!/usr/bin/env python3
"""
Text-only version of the T_Mazer application that doesn't require pygame.
"""
import sys
import os
import time
import random
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import numpy, which is still required
try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Please install it with 'pip install numpy'")
    sys.exit(1)

# Define a simple text-based maze generator
class SimpleTextMazeGenerator:
    def __init__(self, width=20, height=20, complexity=0.7, density=0.7):
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
    
    def generate(self):
        # Create a simple maze with borders
        maze = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Add walls around the perimeter
        maze[0, :] = maze[-1, :] = 1
        maze[:, 0] = maze[:, -1] = 1
        
        # Add random walls within the maze
        for _ in range(int((self.width * self.height) * self.complexity * self.density)):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            
            # Don't block the start or end points
            if (y, x) != (1, 1) and (y, x) != (self.height - 2, self.width - 2):
                maze[y, x] = 1
        
        # Ensure there's a path from start to end
        start = (1, 1)
        end = (self.height - 2, self.width - 2)
        
        # Create a simple path (this isn't a true maze algorithm, just a demonstration)
        # Horizontal path
        for x in range(1, self.width - 1):
            maze[self.height // 2, x] = 0
        
        # Vertical paths from start and end to the horizontal path
        for y in range(1, self.height // 2 + 1):
            maze[y, 1] = 0  # Path from start
            maze[y, self.width - 2] = 0  # Path from end
        
        for y in range(self.height // 2, self.height - 1):
            maze[y, self.width - 2] = 0  # Path to end
        
        return maze

# Simple text-based maze solver
class SimpleTextMazeSolver:
    def __init__(self):
        self.solution_path = []
        self.visited = set()
    
    def solve(self, maze):
        self.solution_path = []
        self.visited = set()
        
        start = (1, 1)
        end = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        # Simple BFS to find a path
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
                return self.solution_path, ["[Solving]"] * len(self.solution_path)
            
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

def display_maze(maze, path=None, current_pos=None):
    """Display the maze in text format."""
    height, width = maze.shape
    
    # Convert path to a set for quick lookup
    path_set = set(path) if path else set()
    
    # Create display characters
    chars = {
        'wall': '█',
        'path': ' ',
        'start': 'S',
        'end': 'E',
        'current': '◉',
        'visited': '·',
        'solution': '●'
    }
    
    print('┌' + '─' * width + '┐')
    
    for y in range(height):
        print('│', end='')
        for x in range(width):
            if (y, x) == (1, 1):
                print(chars['start'], end='')
            elif (y, x) == (height - 2, width - 2):
                print(chars['end'], end='')
            elif current_pos and (y, x) == current_pos:
                print(chars['current'], end='')
            elif (y, x) in path_set:
                print(chars['solution'], end='')
            elif maze[y, x] == 1:
                print(chars['wall'], end='')
            else:
                print(chars['path'], end='')
        print('│')
    
    print('└' + '─' * width + '┘')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="T_Mazer - Text Version")
    
    parser.add_argument("--width", type=int, default=20, help="Width of the maze (default: 20)")
    parser.add_argument("--height", type=int, default=20, help="Height of the maze (default: 20)")
    parser.add_argument("--complexity", type=float, default=0.7, help="Maze complexity factor 0.0-1.0 (default: 0.7)")
    parser.add_argument("--density", type=float, default=0.7, help="Maze density factor 0.0-1.0 (default: 0.7)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps in seconds (default: 0.1)")
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    args = parse_args()
    
    print("T_Mazer - Text Version (No pygame required)")
    print(f"Generating maze with dimensions {args.width}x{args.height}...")
    
    # Generate maze
    generator = SimpleTextMazeGenerator(
        width=args.width, 
        height=args.height,
        complexity=args.complexity,
        density=args.density
    )
    maze = generator.generate()
    
    # Display initial maze
    print("\nInitial Maze:")
    display_maze(maze)
    
    # Solve maze
    print("\nSolving maze...")
    solver = SimpleTextMazeSolver()
    solution_path, _ = solver.solve(maze)
    
    if not solution_path:
        print("No solution found!")
        return
    
    # Display solution step by step
    print(f"\nSolution path length: {len(solution_path)}")
    print("Visualizing solution (press Ctrl+C to stop)...")
    
    try:
        for i, pos in enumerate(solution_path):
            # Clear screen (platform-dependent)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"Step {i+1}/{len(solution_path)}")
            display_maze(maze, solution_path[:i+1], pos)
            time.sleep(args.delay)
        
        print("\nMaze solved!")
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
