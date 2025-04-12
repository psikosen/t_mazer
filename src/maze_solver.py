"""
Maze solver module with TFP-inspired algorithms.
"""
import random
import numpy as np


class TFPToken:
    """Fair Play token definitions for structured reasoning."""
    
    # Main section tokens
    M_START = "[M_Start]"
    M_END = "[M_End]"
    
    # Logic section tokens
    L_START = "[L_Start]"
    L_END = "[L_End]"
    
    # Action tokens
    A_SEQUENCE = "[A:Sequence]"
    A_BRANCH = "[A:Branch]"
    
    # Behavior tokens
    B_EXPLORE = "[B:Explore]"
    B_DEAD_END = "[B:DeadEnd]"
    B_BACKTRACK = "[B:Backtrack]"
    
    # Thinking token
    THINK = "<think>"


class MazeSolver:
    """TFP-inspired maze solver."""
    
    def __init__(self, learning_rate=0.1, exploration_rate=0.3):
        """
        Initialize a new maze solver.
        
        Args:
            learning_rate (float): Rate at which the solver learns
            exploration_rate (float): Initial exploration vs. exploitation balance
        """
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.solution_path = []
        self.reasoning_tokens = []
        self.visited = set()
        self.dead_ends = set()
        
    def solve(self, maze):
        """
        Solve the given maze.
        
        Args:
            maze (numpy.ndarray): 2D array representing the maze
            
        Returns:
            list: Path from start to end
            list: Reasoning tokens used during solving
        """
        # Reset state
        self.solution_path = []
        self.reasoning_tokens = []
        self.visited = set()
        self.dead_ends = set()
        
        # Start at entry point
        start_pos = (1, 1)
        end_pos = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        # Add the start position to the path
        self.solution_path.append(start_pos)
        self.visited.add(start_pos)
        
        # Add start token
        self.reasoning_tokens.append(TFPToken.M_START)
        
        # Initial random exploration (as requested)
        current_pos = start_pos
        for _ in range(3):  # Do 3 random initial moves
            if random.random() < self.exploration_rate:
                next_pos = self._random_valid_move(maze, current_pos)
                if next_pos:
                    self.reasoning_tokens.append(TFPToken.B_EXPLORE)
                    self.solution_path.append(next_pos)
                    self.visited.add(next_pos)
                    current_pos = next_pos
        
        # Continue with actual solving
        success = self._solve_dfs(maze, current_pos, end_pos)
        
        # Add end token
        self.reasoning_tokens.append(TFPToken.M_END)
        
        # If no solution was found, at least return the start position
        if not self.solution_path:
            self.solution_path.append(start_pos)
            
        return self.solution_path, self.reasoning_tokens
    
    def _solve_dfs(self, maze, current_pos, end_pos):
        """
        Solve the maze using depth-first search with TFP structure.
        
        Args:
            maze (numpy.ndarray): Maze representation
            current_pos (tuple): Current position (y, x)
            end_pos (tuple): Target position (y, x)
            
        Returns:
            bool: True if a path was found
        """
        # Check if we reached the target
        if current_pos == end_pos:
            return True
            
        # Get possible moves
        possible_moves = self._get_valid_moves(maze, current_pos)
        
        # If there are multiple paths, add branching token
        if len(possible_moves) > 1:
            self.reasoning_tokens.append(TFPToken.A_BRANCH)
            self.reasoning_tokens.append(TFPToken.L_START)
            self.reasoning_tokens.append(
                f"{TFPToken.THINK} Evaluating {len(possible_moves)} possible paths"
            )
            self.reasoning_tokens.append(TFPToken.L_END)
        elif possible_moves:
            self.reasoning_tokens.append(TFPToken.A_SEQUENCE)
        
        # Try each possible move
        for next_pos in possible_moves:
            # Mark as visited
            self.visited.add(next_pos)
            self.solution_path.append(next_pos)
            
            # TFP structured exploration
            self.reasoning_tokens.append(TFPToken.B_EXPLORE)
            
            # Recursively explore from this position
            if self._solve_dfs(maze, next_pos, end_pos):
                return True
                
            # This path didn't work, backtrack
            self.solution_path.pop()
            self.dead_ends.add(next_pos)
            self.reasoning_tokens.append(TFPToken.B_DEAD_END)
            self.reasoning_tokens.append(TFPToken.B_BACKTRACK)
        
        return False
    
    def _get_valid_moves(self, maze, pos):
        """
        Get valid moves from the current position.
        
        Args:
            maze (numpy.ndarray): Maze representation
            pos (tuple): Current position (y, x)
            
        Returns:
            list: Valid next positions
        """
        y, x = pos
        possible_moves = []
        
        # Check all four directions
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            
            # Check if the move is valid (within bounds, not a wall, not visited, not a dead end)
            if (0 <= ny < maze.shape[0] and 
                0 <= nx < maze.shape[1] and 
                maze[ny, nx] == 0 and
                (ny, nx) not in self.visited and
                (ny, nx) not in self.dead_ends):
                possible_moves.append((ny, nx))
        
        return possible_moves
    
    def _random_valid_move(self, maze, pos):
        """
        Make a random valid move for initial exploration.
        
        Args:
            maze (numpy.ndarray): Maze representation
            pos (tuple): Current position (y, x)
            
        Returns:
            tuple or None: Next position or None if no valid moves
        """
        valid_moves = self._get_valid_moves(maze, pos)
        if valid_moves:
            return random.choice(valid_moves)
        return None
