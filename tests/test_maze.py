"""
Tests for the maze generator and solver modules.
"""
import unittest
import numpy as np
import sys
import os

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze_generator import MazeGenerator
from maze_solver import MazeSolver, TFPToken


class TestMazeGenerator(unittest.TestCase):
    """Test cases for the maze generator."""
    
    def test_maze_dimensions(self):
        """Test that generated mazes have correct dimensions."""
        width, height = 15, 20
        generator = MazeGenerator(width=width, height=height)
        maze = generator.generate()
        
        self.assertEqual(maze.shape, (height, width))
    
    def test_maze_has_borders(self):
        """Test that maze has borders (walls on the edges)."""
        generator = MazeGenerator(width=10, height=10)
        maze = generator.generate()
        
        # Check top and bottom borders
        self.assertTrue(np.all(maze[0, :] == 1))
        self.assertTrue(np.all(maze[-1, :] == 1))
        
        # Check left and right borders
        self.assertTrue(np.all(maze[:, 0] == 1))
        self.assertTrue(np.all(maze[:, -1] == 1))
    
    def test_maze_has_start_and_end(self):
        """Test that maze has entry and exit points."""
        generator = MazeGenerator(width=10, height=10)
        maze = generator.generate()
        
        # Check start point (top-left inner cell)
        self.assertEqual(maze[1, 1], 0)
        
        # Check end point (bottom-right inner cell)
        self.assertEqual(maze[-2, -2], 0)


class TestMazeSolver(unittest.TestCase):
    """Test cases for the maze solver."""
    
    def test_solver_finds_path(self):
        """Test that solver finds a path in a simple maze."""
        # Create a simple maze with a direct path
        maze = np.ones((7, 7), dtype=np.int8)
        # Create a direct path from start to end
        maze[1, 1:6] = 0
        maze[1:6, 5] = 0
        
        solver = MazeSolver()
        path, tokens = solver.solve(maze)
        
        # Check that a path was found
        self.assertGreater(len(path), 0)
        
        # Check that the path leads to the end position
        self.assertEqual(path[-1], (5, 5))
    
    def test_tokens_structure(self):
        """Test that the solver uses the expected token structure."""
        # Create a simple maze
        maze = np.ones((5, 5), dtype=np.int8)
        maze[1:4, 1:4] = 0  # Create open space
        
        solver = MazeSolver()
        _, tokens = solver.solve(maze)
        
        # Check that tokens start and end with the appropriate markers
        self.assertEqual(tokens[0], TFPToken.M_START)
        self.assertEqual(tokens[-1], TFPToken.M_END)
        
        # Check that exploration tokens are used
        self.assertTrue(any(token == TFPToken.B_EXPLORE for token in tokens))


if __name__ == '__main__':
    unittest.main()
