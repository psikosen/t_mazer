"""
Procedural maze generator module.
"""
import random
import numpy as np


class MazeGenerator:
    """Procedural maze generator with customizable complexity."""
    
    def __init__(self, width=20, height=20, complexity=0.7, density=0.7):
        """
        Initialize a new maze generator.
        
        Args:
            width (int): Width of the maze grid
            height (int): Height of the maze grid
            complexity (float): Complexity factor (0.0-1.0)
            density (float): Density factor (0.0-1.0)
        """
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
        
    def generate(self):
        """
        Generate a random maze using a recursive depth-first algorithm.
        
        Returns:
            numpy.ndarray: 2D array representing the maze (0=path, 1=wall)
        """
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
        
        # Ensure the maze is solvable
        if not self.ensure_solvable(maze):
            # If not solvable, create a direct path from start to end
            # This should rarely happen with our algorithm, but just in case
            y1, x1 = 1, 1
            y2, x2 = self.height - 2, self.width - 2
            
            # Create horizontal path
            for x in range(min(x1, x2), max(x1, x2) + 1):
                maze[y1, x] = 0
            
            # Create vertical path
            for y in range(min(y1, y2), max(y1, y2) + 1):
                maze[y, x2] = 0
        
        # Create entry and exit points
        maze[1, 1] = 0  # Start point
        maze[self.height - 2, self.width - 2] = 0  # End point
        
        return maze
    
    def ensure_solvable(self, maze):
        """
        Ensure the maze has at least one valid path from start to end.
        Uses a breadth-first search to find a path.
        
        Args:
            maze (numpy.ndarray): Maze array
            
        Returns:
            bool: True if the maze is solvable, False otherwise
        """
        start = (1, 1)
        end = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        # BFS to find a path
        queue = [start]
        visited = {start}
        parents = {start: None}
        
        while queue:
            current = queue.pop(0)
            
            if current == end:
                # Path found, trace it back
                path = []
                while current != start:
                    path.append(current)
                    current = parents[current]
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
                    parents[(ny, nx)] = current
        
        # No path found
        return False
