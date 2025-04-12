#!/usr/bin/env python3
"""
Ultra-simple version of the T_Mazer application that doesn't require any external dependencies.
"""
import sys
import os
import time
import random
import argparse

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

def display_maze(maze, path=None, current=None):
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

def parse_args():
    parser = argparse.ArgumentParser(description="T_Mazer - Simple Version (No Dependencies)")
    
    parser.add_argument("--width", type=int, default=15, help="Width of the maze (default: 15)")
    parser.add_argument("--height", type=int, default=10, help="Height of the maze (default: 10)")
    parser.add_argument("--complexity", type=float, default=0.7, help="Maze complexity (default: 0.7)")
    parser.add_argument("--density", type=float, default=0.7, help="Maze density (default: 0.7)")
    parser.add_argument("--delay", type=float, default=0.1, help="Visualization delay (default: 0.1)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"T_Mazer - Simple Version (No Dependencies Required)")
    print(f"Generating maze with dimensions {args.width}x{args.height}...")
    
    # Generate maze
    maze_gen = SimpleMaze(args.width, args.height, args.complexity, args.density)
    maze = maze_gen.generate()
    
    # Show initial maze
    print("\nInitial Maze:")
    display_maze(maze)
    
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
            display_maze(maze, solution[:i+1], pos)
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
