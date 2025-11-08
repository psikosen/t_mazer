"""
Advanced maze generator with adaptive complexity based on solver performance.
Creates progressively challenging mazes based on self-reflection insights.
"""
import numpy as np
import random
from collections import deque


class AdvancedMazeGenerator:
    """
    Advanced maze generator that adapts difficulty based on solver performance.
    """
    
    def __init__(self, base_width=20, base_height=20, initial_complexity=0.5):
        """
        Initialize advanced maze generator.
        
        Args:
            base_width (int): Base width for mazes
            base_height (int): Base height for mazes
            initial_complexity (float): Starting complexity (0.0-1.0)
        """
        self.base_width = base_width
        self.base_height = base_height
        self.complexity = initial_complexity
        self.performance_history = []
        self.current_difficulty = 'medium'
        
    def generate(self, target_difficulty=None, solver_stats=None):
        """
        Generate a maze with adaptive difficulty.
        Always generates a NEW maze (never reuses).
        
        Args:
            target_difficulty (str): 'easy', 'medium', 'hard', or None for adaptive
            solver_stats (dict): Current solver performance statistics
            
        Returns:
            np.ndarray: Generated maze (always a new, unique maze)
        """
        # Adjust complexity based on performance
        if solver_stats:
            self._adjust_complexity(solver_stats)
        
        if target_difficulty:
            self._set_difficulty(target_difficulty)
        
        # Add random variation to complexity each generation
        # Use time for seed variation to ensure different mazes each time
        import time as time_module
        random.seed(int(time_module.time() * 1000) % 1000000)  # Fresh seed each generation
        variation = random.uniform(-0.1, 0.1)
        self.complexity = max(0.2, min(1.0, self.complexity + variation))
        
        # Determine maze dimensions (with some variation)
        width, height = self._calculate_dimensions()
        
        # Add random size variation each time (ensures different sizes)
        # More variation to ensure mazes change each run
        size_var = random.uniform(0.85, 1.15)  # Increased variation range
        width = max(11, int(width * size_var))
        height = max(11, int(height * size_var))
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1
        
        # Generate NEW maze with calculated parameters (always fresh)
        maze = self._generate_adaptive_maze(width, height)
        
        return maze
    
    def _calculate_dimensions(self):
        """Calculate maze dimensions based on current complexity."""
        # Increase size with complexity
        size_multiplier = 1.0 + (self.complexity * 0.5)
        
        width = int(self.base_width * size_multiplier)
        height = int(self.base_height * size_multiplier)
        
        # Ensure minimum size and odd dimensions for path generation
        width = max(11, width)
        height = max(11, height)
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1
        
        return width, height
    
    def _set_difficulty(self, difficulty):
        """Set difficulty level explicitly."""
        difficulty_map = {
            'easy': 0.3,
            'medium': 0.5,
            'hard': 0.7,
            'expert': 0.9
        }
        self.complexity = difficulty_map.get(difficulty, 0.5)
        self.current_difficulty = difficulty
    
    def _adjust_complexity(self, solver_stats):
        """
        Adjust complexity based on solver performance.
        
        Args:
            solver_stats (dict): Statistics from self-reflection
        """
        # Get recent performance metrics
        efficiency = solver_stats.get('average_efficiency', 0.5)
        success_rate = solver_stats.get('success_rate', 0.5)
        
        # If solver is performing well, increase difficulty
        if efficiency > 0.8 and success_rate > 0.9:
            self.complexity = min(1.0, self.complexity + 0.05)
            self.current_difficulty = self._complexity_to_difficulty()
        # If solver is struggling, decrease difficulty
        elif efficiency < 0.4 or success_rate < 0.6:
            self.complexity = max(0.2, self.complexity - 0.05)
            self.current_difficulty = self._complexity_to_difficulty()
        # Otherwise maintain current difficulty
    
    def _complexity_to_difficulty(self):
        """Convert complexity value to difficulty string."""
        if self.complexity < 0.4:
            return 'easy'
        elif self.complexity < 0.6:
            return 'medium'
        elif self.complexity < 0.8:
            return 'hard'
        else:
            return 'expert'
    
    def _generate_adaptive_maze(self, width, height):
        """
        Generate maze with adaptive features based on complexity.
        
        Args:
            width (int): Maze width
            height (int): Maze height
            
        Returns:
            np.ndarray: Generated maze
        """
        # Start with all walls
        maze = np.ones((height, width), dtype=np.int8)
        
        # Generate base structure using recursive backtracking
        self._recursive_backtrack(maze, width, height)
        
        # Add adaptive features based on complexity
        self._add_adaptive_features(maze)
        
        # Ensure solvability
        self._ensure_solvable(maze)
        
        return maze
    
    def _recursive_backtrack(self, maze, width, height):
        """Generate base maze structure using recursive backtracking."""
        # Always use fresh random seed for variation
        random.seed()  # Reset to system random
        
        # Start at a random odd coordinate
        start_x = random.randrange(1, width - 1, 2)
        start_y = random.randrange(1, height - 1, 2)
        
        stack = [(start_x, start_y)]
        visited = {(start_x, start_y)}
        
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        while stack:
            x, y = stack[-1]
            
            # Find unvisited neighbors
            neighbors = []
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (1 <= nx < width - 1 and 
                    1 <= ny < height - 1 and 
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny, dx, dy))
            
            if neighbors:
                # Choose random neighbor
                nx, ny, dx, dy = random.choice(neighbors)
                
                # Carve path
                maze[y, x] = 0
                maze[y + dy//2, x + dx//2] = 0
                maze[ny, nx] = 0
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()
        
        # Set borders
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1
        
        # Ensure start and end are open
        maze[1, 1] = 0
        maze[height - 2, width - 2] = 0
    
    def _add_adaptive_features(self, maze):
        """Add features that make the maze more challenging based on complexity."""
        height, width = maze.shape
        
        # Add dead ends based on complexity
        dead_end_count = int(self.complexity * (width * height) / 50)
        self._add_dead_ends(maze, dead_end_count)
        
        # Add loops and alternative paths (higher complexity = fewer loops)
        if self.complexity < 0.7:
            loop_count = int((1 - self.complexity) * 10)
            self._add_loops(maze, loop_count)
        
        # Add bottlenecks (narrow passages)
        if self.complexity > 0.5:
            bottleneck_count = int((self.complexity - 0.5) * 5)
            self._add_bottlenecks(maze, bottleneck_count)
    
    def _add_dead_ends(self, maze, count):
        """Add dead-end branches to increase difficulty."""
        height, width = maze.shape
        added = 0
        
        for _ in range(count * 2):  # Try twice as many attempts
            if added >= count:
                break
            
            # Find a path cell
            y = random.randrange(2, height - 2)
            x = random.randrange(2, width - 2)
            
            if maze[y, x] == 0:
                # Try to create a dead end
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)
                
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if (1 <= ny < height - 1 and 
                        1 <= nx < width - 1 and 
                        maze[ny, nx] == 1):
                        # Create short dead end
                        maze[ny, nx] = 0
                        # Make it a dead end by walling off further progress
                        if (1 <= ny + dy < height - 1 and 
                            1 <= nx + dx < width - 1):
                            maze[ny + dy, nx + dx] = 1
                        added += 1
                        break
    
    def _add_loops(self, maze, count):
        """Add loops (alternative paths) to make maze easier to solve but harder to optimize."""
        height, width = maze.shape
        added = 0
        
        for _ in range(count * 3):
            if added >= count:
                break
            
            y = random.randrange(2, height - 2)
            x = random.randrange(2, width - 2)
            
            if maze[y, x] == 1:
                # Check if we can create a loop by connecting nearby paths
                neighbors_path = []
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 
                        0 <= nx < width and 
                        maze[ny, nx] == 0):
                        neighbors_path.append((ny, nx))
                
                if len(neighbors_path) >= 2:
                    # Can create a loop
                    maze[y, x] = 0
                    added += 1
    
    def _add_bottlenecks(self, maze, count):
        """Add bottlenecks (narrow passages) that require precise navigation."""
        height, width = maze.shape
        added = 0
        
        for _ in range(count * 5):
            if added >= count:
                break
            
            # Find a path that can be narrowed
            y = random.randrange(2, height - 2)
            x = random.randrange(2, width - 2)
            
            if maze[y, x] == 0:
                # Check if we can add adjacent walls to create bottleneck
                can_bottleneck = True
                for dy, dx in [(0, 1), (0, -1)]:
                    if not (1 <= y + dy < height - 1):
                        can_bottleneck = False
                        break
                    if maze[y + dy, x] == 0:
                        can_bottleneck = False
                        break
                
                if can_bottleneck:
                    # Create bottleneck by ensuring only one direction is open
                    maze[y + 1, x] = 1
                    maze[y - 1, x] = 1
                    added += 1
    
    def _ensure_solvable(self, maze):
        """Ensure the maze is solvable using BFS."""
        start = (1, 1)
        end = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                return True  # Solvable
            
            y, x = current
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < maze.shape[0] and 
                    0 <= nx < maze.shape[1] and 
                    maze[ny, nx] == 0 and
                    (ny, nx) not in visited):
                    queue.append((ny, nx))
                    visited.add((ny, nx))
                    parent[(ny, nx)] = current
        
        # Not solvable - create direct path
        y1, x1 = start
        y2, x2 = end
        
        # Horizontal path
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 <= y1 < maze.shape[0]:
                maze[y1, x] = 0
        
        # Vertical path
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 <= x2 < maze.shape[1]:
                maze[y, x2] = 0
        
        return True
    
    def get_difficulty_info(self):
        """Get current difficulty information."""
        return {
            'complexity': self.complexity,
            'difficulty': self.current_difficulty,
            'base_size': (self.base_width, self.base_height)
        }

