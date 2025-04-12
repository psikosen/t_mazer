#!/usr/bin/env python3
"""
T_Mazer Enhanced - With model training and continuous maze solving.
"""
import sys
import os
import time
import random
import argparse
import pickle
import json
from datetime import datetime

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

# Directory for saving model weights
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ===== SIMPLE MODEL WITH TERNARY WEIGHTS =====

class TernaryNeuralSolver:
    """
    Simple neural network that learns to solve mazes using ternary weights.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.history = []  # Track performance over time
        self.training_steps = 0
        
    def _initialize_weights(self, input_size, hidden_size=16, output_size=4):
        """Initialize weights randomly within a small range."""
        np.random.seed(42)  # For reproducibility
        
        # Initialize with small random values
        self.weights = {
            'w1': np.random.randn(input_size, hidden_size) * 0.01,
            'w2': np.random.randn(hidden_size, output_size) * 0.01,
            'b1': np.zeros((1, hidden_size)),
            'b2': np.zeros((1, output_size))
        }
        
        # Ternarize the initial weights
        self._ternarize_weights()
    
    def _ternarize_weights(self):
        """Convert weights to ternary values {-1, 0, 1}."""
        for key in ['w1', 'w2']:
            # Thresholding approach for ternarization
            threshold = 0.3 * np.mean(np.abs(self.weights[key]))
            
            # Create ternary weights
            weights = np.zeros_like(self.weights[key])
            weights[self.weights[key] > threshold] = 1
            weights[self.weights[key] < -threshold] = -1
            
            # Store ternary weights
            self.weights[key] = weights
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
    
    def _relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
        
    def _prepare_input(self, maze, current_pos, end_pos):
        """
        Convert maze state around current position to a feature vector.
        
        Args:
            maze: 2D array representing the maze
            current_pos: Current position (y, x)
            end_pos: Target position (y, x)
            
        Returns:
            np.array: Feature vector
        """
        y, x = current_pos
        height, width = len(maze), len(maze[0]) if isinstance(maze, list) else maze.shape
        
        # Features:
        # 1-4. Is there a wall in each of the four directions?
        # 5-8. Direction to target (normalized y and x components)
        features = []
        
        # Wall features - can we move in each direction?
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # right, down, left, up
            ny, nx = y + dy, x + dx
            # Check if this is a valid move
            if 0 <= ny < height and 0 <= nx < width and maze[ny][nx] == 0:
                features.append(0)  # No wall
            else:
                features.append(1)  # Wall
                
        # Direction to target
        end_y, end_x = end_pos
        # Normalize by the maximum possible distance
        features.append((end_y - y) / height)
        features.append((end_x - x) / width)
        
        # Additional feature: have we visited this cell before?
        # This requires the solver to store visited cells
        # but we'll leave this as a placeholder for now
        features.append(0)
        
        # Return as numpy array
        return np.array(features).reshape(1, -1)
    
    def predict(self, maze, current_pos, end_pos):
        """
        Predict the best move given the current state.
        
        Returns:
            int: Direction index (0: right, 1: down, 2: left, 3: up)
        """
        if self.weights is None:
            # If weights haven't been initialized, create them based on input size
            input_features = self._prepare_input(maze, current_pos, end_pos)
            self._initialize_weights(input_features.shape[1])
        
        # Get input features
        X = self._prepare_input(maze, current_pos, end_pos)
        
        # Forward pass through the network
        z1 = np.dot(X, self.weights['w1']) + self.weights['b1']
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.weights['w2']) + self.weights['b2']
        outputs = z2  # Raw scores for each direction
        
        # Return the index of the highest score
        return np.argmax(outputs)
    
    def train(self, maze, current_pos, end_pos, best_action, reward):
        """
        Train the model on a single step.
        
        Args:
            maze: The maze
            current_pos: Current position
            end_pos: Target position
            best_action: The best action to take (0-3)
            reward: Reward value for this action
        """
        # Skip training if no numpy
        if not HAS_NUMPY:
            return
            
        self.training_steps += 1
        
        # Get input features
        X = self._prepare_input(maze, current_pos, end_pos)
        
        # If weights not initialized yet, create them
        if self.weights is None:
            self._initialize_weights(X.shape[1])
        
        # Current prediction
        z1 = np.dot(X, self.weights['w1']) + self.weights['b1']
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.weights['w2']) + self.weights['b2']
        current_outputs = z2
        
        # Create target outputs by modifying the current outputs
        target_outputs = current_outputs.copy()
        target_outputs[0, best_action] += reward  # Increase the value of the best action
        
        # Simple loss function (mean squared error)
        loss = np.mean((current_outputs - target_outputs)**2)
        
        # Backward pass (simplified)
        # In a real implementation, we would compute proper gradients
        dz2 = 2 * (current_outputs - target_outputs) / X.shape[0]
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.weights['w2'].T)
        dz1 = da1 * (z1 > 0)  # Derivative of ReLU
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.weights['w1'] -= self.learning_rate * dw1
        self.weights['w2'] -= self.learning_rate * dw2
        self.weights['b1'] -= self.learning_rate * db1
        self.weights['b2'] -= self.learning_rate * db2
        
        # Ternarize weights periodically (every 10 steps)
        if self.training_steps % 10 == 0:
            self._ternarize_weights()
            
        # Record performance
        self.history.append({
            'loss': float(loss),
            'reward': reward,
            'steps': self.training_steps
        })
    
    def save(self, filename=None):
        """Save the model weights to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(MODEL_DIR, f"ternary_solver_{timestamp}.pkl")
        
        data = {
            'weights': self.weights,
            'history': self.history,
            'training_steps': self.training_steps
        }
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save as pickle or JSON depending on numpy availability
        if HAS_NUMPY:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                'history': self.history,
                'training_steps': self.training_steps
            }
            with open(filename, 'w') as f:
                json.dump(json_data, f)
        
        print(f"Model saved to {filename}")
        return filename
    
    def load(self, filename):
        """Load model weights from a file."""
        try:
            # Try pickle first
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
            self.weights = data.get('weights')
            self.history = data.get('history', [])
            self.training_steps = data.get('training_steps', 0)
            
        except (pickle.UnpicklingError, ModuleNotFoundError):
            # Try JSON as fallback
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    
                self.history = data.get('history', [])
                self.training_steps = data.get('training_steps', 0)
                # Note: weights not loaded from JSON
                
            except json.JSONDecodeError:
                print(f"Error: Could not load model from {filename}")
                return False
        
        except FileNotFoundError:
            print(f"Error: Model file {filename} not found")
            return False
            
        print(f"Model loaded from {filename}")
        print(f"Model has been trained for {self.training_steps} steps")
        return True

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

class SimpleTrainableSolver:
    """Simple maze solver that can learn from experience."""
    
    def __init__(self):
        """Initialize the solver with a neural network model."""
        self.solution = []
        self.visited = set()
        self.model = TernaryNeuralSolver()
        self.latest_reward = 0
        self.latest_action = None
        self.previous_position = None
        
    def solve(self, maze, width, height, training_mode=True):
        """
        Solve the maze using the neural network for guidance.
        
        Args:
            maze: The maze to solve
            width: Maze width
            height: Maze height
            training_mode: Whether to train the model during solving
            
        Returns:
            list: Solution path
        """
        # Reset state
        self.solution = []
        self.visited = set()
        self.latest_reward = 0
        self.latest_action = None
        
        # Start and end positions
        start_pos = (1, 1)
        end_pos = (height-2, width-2)
        
        # Start solving
        current_pos = start_pos
        self.solution.append(current_pos)
        self.visited.add(current_pos)
        
        # Maximum steps to prevent infinite loops
        max_steps = width * height * 2
        step_count = 0
        
        # Direction vectors (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while current_pos != end_pos and step_count < max_steps:
            # Get neural network's prediction
            action = self.model.predict(maze, current_pos, end_pos)
            
            # Get the direction vector
            dy, dx = directions[action]
            next_y, next_x = current_pos[0] + dy, current_pos[1] + dx
            
            # Check if this is a valid move
            valid_move = (0 <= next_y < height and 
                         0 <= next_x < width and 
                         maze[next_y][next_x] == 0 and
                         (next_y, next_x) not in self.visited)
            
            # If it's not a valid move, try other directions
            if not valid_move:
                # Calculate rewards for each direction
                rewards = [-1] * 4  # Default to -1
                
                # Check all directions
                for i, (dy, dx) in enumerate(directions):
                    ny, nx = current_pos[0] + dy, current_pos[1] + dx
                    
                    # If this is a valid move, calculate reward
                    if (0 <= ny < height and 
                        0 <= nx < width and 
                        maze[ny][nx] == 0 and
                        (ny, nx) not in self.visited):
                        
                        # Calculate Manhattan distance to goal
                        distance = abs(ny - end_pos[0]) + abs(nx - end_pos[1])
                        rewards[i] = 1 / (distance + 1)  # Higher reward for closer to goal
                
                # Choose the best direction
                best_action = rewards.index(max(rewards))
                
                # Get new direction
                dy, dx = directions[best_action]
                next_y, next_x = current_pos[0] + dy, current_pos[1] + dx
                
                # Check if this is now a valid move
                valid_move = (0 <= next_y < height and 
                             0 <= next_x < width and 
                             maze[next_y][next_x] == 0 and
                             (next_y, next_x) not in self.visited)
                
                # If still not valid, try random exploration
                if not valid_move:
                    # Random exploration of unvisited neighbors
                    valid_neighbors = []
                    for i, (dy, dx) in enumerate(directions):
                        ny, nx = current_pos[0] + dy, current_pos[1] + dx
                        if (0 <= ny < height and 
                            0 <= nx < width and 
                            maze[ny][nx] == 0 and
                            (ny, nx) not in self.visited):
                            valid_neighbors.append((ny, nx, i))
                    
                    # If we have valid neighbors, pick one randomly
                    if valid_neighbors:
                        next_y, next_x, best_action = random.choice(valid_neighbors)
                        valid_move = True
                    else:
                        # Backtrack
                        if len(self.solution) > 1:
                            # Calculate which action led to the previous position
                            prev_pos = self.solution[-2]
                            for i, (dy, dx) in enumerate(directions):
                                if (prev_pos[0] == current_pos[0] + dy and 
                                    prev_pos[1] == current_pos[1] + dx):
                                    best_action = i
                                    break
                            
                            # Move back
                            next_y, next_x = prev_pos
                            valid_move = True
                
                # If training mode, train the model
                if training_mode and self.previous_position is not None:
                    # Calculate reward: -1 for bad move, +0.5 for getting closer, +1 for reaching the goal
                    if not valid_move:
                        reward = -1
                    elif (next_y, next_x) == end_pos:
                        reward = 1
                    else:
                        # Reward based on distance change
                        old_dist = abs(current_pos[0] - end_pos[0]) + abs(current_pos[1] - end_pos[1])
                        new_dist = abs(next_y - end_pos[0]) + abs(next_x - end_pos[1])
                        reward = 0.5 if new_dist < old_dist else -0.2
                    
                    # Train model on previous state
                    self.model.train(maze, self.previous_position, end_pos, 
                                    self.latest_action, reward)
            
            # Make the move
            if valid_move:
                self.previous_position = current_pos
                self.latest_action = action
                
                current_pos = (next_y, next_x)
                self.solution.append(current_pos)
                self.visited.add(current_pos)
            
            step_count += 1
        
        # Train one last time if we reached the goal
        if training_mode and current_pos == end_pos and self.previous_position is not None:
            self.model.train(maze, self.previous_position, end_pos, 
                          self.latest_action, 1.0)  # Big reward for reaching goal
            
            # Save the model after successfully solving
            self.model.save()
        
        # Return the solution
        return self.solution

def display_simple_maze(maze, path=None, current=None, stats=None):
    """Display the maze with optional stats."""
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
    
    # Show stats if provided
    if stats:
        print("\n=== T_Mazer Stats ===")
        print(f"Mazes Solved: {stats['solved']}")
        print(f"Total Steps: {stats['total_steps']}")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print(f"Training Steps: {stats['training_steps']}")
        print(f"Learning Rate: {stats['learning_rate']:.4f}")
        print("=====================\n")
    
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

def run_simple_training_version(args):
    """Run the simple training version with ternary weights."""
    print(f"T_Mazer - Training Version (Ternary Weights)")
    
    # Load an existing model if available
    solver = SimpleTrainableSolver()
    
    # Find the latest model file
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("ternary_solver_")]
    if model_files:
        latest_model = os.path.join(MODEL_DIR, sorted(model_files)[-1])
        solver.model.load(latest_model)
    
    # Stats tracking
    stats = {
        'solved': 0,
        'total_steps': 0,
        'avg_steps': 0,
        'training_steps': solver.model.training_steps,
        'learning_rate': solver.model.learning_rate
    }
    
    # Continuous maze solving loop
    try:
        while True:
            # Generate a new maze
            print(f"\nGenerating maze #{stats['solved'] + 1} with dimensions {args.width}x{args.height}...")
            maze_gen = SimpleMaze(args.width, args.height, args.complexity, args.density)
            maze = maze_gen.generate()
            
            # Show initial maze
            print("\nInitial Maze:")
            display_simple_maze(maze, stats=stats)
            
            # Solve maze
            print("\nSolving maze using neural network...")
            start_time = time.time()
            solution = solver.solve(maze, args.width, args.height, training_mode=True)
            end_time = time.time()
            
            if not solution or solution[-1] != (args.height-2, args.width-2):
                print("Failed to find a solution!")
                continue
            
            # Update stats
            stats['solved'] += 1
            stats['total_steps'] += len(solution)
            stats['avg_steps'] = stats['total_steps'] / stats['solved']
            stats['training_steps'] = solver.model.training_steps
            
            # Display solution step by step
            print(f"\nSolution found! Path length: {len(solution)}")
            print(f"Solving time: {end_time - start_time:.2f} seconds")
            print("Visualizing solution (press Ctrl+C to stop)...")
            
            # Step through the solution
            for i, pos in enumerate(solution):
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"Maze #{stats['solved']} - Step {i+1}/{len(solution)}")
                display_simple_maze(maze, solution[:i+1], pos, stats)
                time.sleep(args.delay)
            
            print("\nMaze solved! Training model...")
            
            # Save model periodically
            if stats['solved'] % 5 == 0:
                solver.model.save()
            
            # Ask user if they want to continue
            print("\nMaze solved successfully! Press Enter to generate a new maze...")
            print("Press Ctrl+C to exit.")
            input()  # Wait for user input
            
    except KeyboardInterrupt:
        print("\nExiting training mode...")
        # Save model before exiting
        solver.model.save()

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
    
    class NumpyTernaryModel:
        """Neural network model with ternary weights for maze solving."""
        
        def __init__(self, input_size=7, hidden_size=16, output_size=4, learning_rate=0.01):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.learning_rate = learning_rate
            
            # Initialize weights
            self.weights = {
                'w1': np.random.randn(input_size, hidden_size) * 0.01,
                'w2': np.random.randn(hidden_size, output_size) * 0.01,
                'b1': np.zeros((1, hidden_size)),
                'b2': np.zeros((1, output_size))
            }
            
            # Ternarize weights
            self._ternarize_weights()
            
            # Training history
            self.history = []
            self.training_steps = 0
        
        def _ternarize_weights(self):
            """Convert weights to ternary values {-1, 0, 1}."""
            for key in ['w1', 'w2']:
                # Thresholding approach
                threshold = 0.3 * np.mean(np.abs(self.weights[key]))
                
                # Create ternary weights
                weights = np.zeros_like(self.weights[key])
                weights[self.weights[key] > threshold] = 1
                weights[self.weights[key] < -threshold] = -1
                
                # Store ternary weights
                self.weights[key] = weights
        
        def forward(self, X):
            """Forward pass through the network."""
            # First layer
            z1 = np.dot(X, self.weights['w1']) + self.weights['b1']
            a1 = np.maximum(0, z1)  # ReLU
            
            # Second layer
            z2 = np.dot(a1, self.weights['w2']) + self.weights['b2']
            
            return z2, a1, z1
        
        def train(self, X, target_output, reward=1.0):
            """Train the model on a single example."""
            self.training_steps += 1
            
            # Forward pass
            current_output, a1, z1 = self.forward(X)
            
            # Adjust target based on reward
            target = np.zeros((1, self.output_size))
            target[0, target_output] = reward
            
            # Compute loss
            loss = np.mean((current_output - target)**2)
            
            # Backward pass (simplified)
            dz2 = 2 * (current_output - target) / X.shape[0]
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.weights['w2'].T)
            dz1 = da1 * (z1 > 0)  # ReLU derivative
            dw1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.weights['w1'] -= self.learning_rate * dw1
            self.weights['w2'] -= self.learning_rate * dw2
            self.weights['b1'] -= self.learning_rate * db1
            self.weights['b2'] -= self.learning_rate * db2
            
            # Ternarize weights periodically
            if self.training_steps % 10 == 0:
                self._ternarize_weights()
            
            # Record history
            self.history.append({
                'loss': float(loss),
                'reward': float(reward),
                'steps': self.training_steps
            })
            
            return loss
        
        def predict(self, X):
            """Predict the best action given the input."""
            output, _, _ = self.forward(X)
            return np.argmax(output[0])
        
        def save(self, filename=None):
            """Save the model to a file."""
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(MODEL_DIR, f"numpy_ternary_{timestamp}.pkl")
            
            data = {
                'weights': self.weights,
                'history': self.history,
                'training_steps': self.training_steps,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Model saved to {filename}")
            return filename
        
        def load(self, filename):
            """Load model from a file."""
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                self.weights = data['weights']
                self.history = data['history']
                self.training_steps = data['training_steps']
                self.input_size = data['input_size']
                self.hidden_size = data['hidden_size']
                self.output_size = data['output_size']
                self.learning_rate = data['learning_rate']
                
                print(f"Model loaded from {filename}")
                print(f"Model has been trained for {self.training_steps} steps")
                return True
                
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
    
    class NumpyTrainableSolver:
        """Numpy-based maze solver with neural network training."""
        
        def __init__(self):
            # Initialize neural network model
            self.model = None
            self.solution_path = []
            self.visited = set()
            self.previous_state = None
            self.previous_action = None
        
        def _get_features(self, maze, position, target):
            """Extract features from the current state."""
            y, x = position
            height, width = maze.shape
            
            # Features array
            features = np.zeros((1, 7))
            
            # Wall features (can we move in each direction?)
            for i, (dy, dx) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0:
                    features[0, i] = 1  # Can move in this direction
            
            # Direction to target
            ty, tx = target
            features[0, 4] = (ty - y) / height  # Normalized y direction
            features[0, 5] = (tx - x) / width   # Normalized x direction
            
            # Bias term
            features[0, 6] = 1
            
            return features
        
        def solve(self, maze, training_mode=True):
            """
            Solve the maze using neural network guidance.
            
            Args:
                maze: NumPy array representing the maze
                training_mode: Whether to train the model during solving
                
            Returns:
                list: Solution path
                list: Reasoning tokens
            """
            # Reset state
            self.solution_path = []
            self.visited = set()
            self.previous_state = None
            self.previous_action = None
            
            # Initialize model if needed
            if self.model is None:
                self.model = NumpyTernaryModel()
            
            # Start and end positions
            start = (1, 1)
            end = (maze.shape[0] - 2, maze.shape[1] - 2)
            
            # Direction vectors (right, down, left, up)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            # Add start position
            current = start
            self.solution_path.append(current)
            self.visited.add(current)
            
            # Maximum steps to prevent infinite loops
            max_steps = maze.shape[0] * maze.shape[1] * 2
            steps = 0
            
            while current != end and steps < max_steps:
                # Get state features
                features = self._get_features(maze, current, end)
                
                # Get model prediction
                action = self.model.predict(features)
                
                # Try to move in the predicted direction
                dy, dx = directions[action]
                ny, nx = current[0] + dy, current[1] + dx
                
                # Check if this is a valid move
                valid_move = (0 <= ny < maze.shape[0] and 
                              0 <= nx < maze.shape[1] and 
                              maze[ny, nx] == 0)
                
                # If the move is valid, take it
                if valid_move:
                    # If we've already visited this cell, it might be a loop
                    if (ny, nx) in self.visited:
                        # Try to find an unvisited neighbor
                        unvisited = []
                        for i, (dy, dx) in enumerate(directions):
                            py, px = current[0] + dy, current[1] + dx
                            if (0 <= py < maze.shape[0] and 
                                0 <= px < maze.shape[1] and 
                                maze[py, px] == 0 and
                                (py, px) not in self.visited):
                                unvisited.append((i, (py, px)))
                        
                        # If we have unvisited neighbors, choose one randomly
                        if unvisited:
                            action, (ny, nx) = random.choice(unvisited)
                        else:
                            # Backtrack
                            if len(self.solution_path) > 1:
                                # Remove current position from path
                                self.solution_path.pop()
                                # Get previous position
                                ny, nx = self.solution_path[-1]
                                # Try to find the action that led to the previous position
                                for i, (dy, dx) in enumerate(directions):
                                    if (ny, nx) == (current[0] + dy, current[1] + dx):
                                        action = i
                                        break
                    
                    # Train the model if in training mode
                    if training_mode and self.previous_state is not None:
                        # Calculate reward
                        if (ny, nx) == end:
                            reward = 1.0  # Reached the goal
                        else:
                            # Calculate Manhattan distance change
                            old_dist = abs(current[0] - end[0]) + abs(current[1] - end[1])
                            new_dist = abs(ny - end[0]) + abs(nx - end[1])
                            
                            if new_dist < old_dist:
                                reward = 0.5  # Getting closer
                            else:
                                reward = -0.2  # Getting farther
                        
                        # Train model
                        self.model.train(self.previous_state, self.previous_action, reward)
                    
                    # Store current state for training
                    self.previous_state = features
                    self.previous_action = action
                    
                    # Move to the new position
                    current = (ny, nx)
                    self.solution_path.append(current)
                    self.visited.add(current)
                
                else:
                    # Invalid move, try to find a valid one
                    valid_moves = []
                    for i, (dy, dx) in enumerate(directions):
                        ny, nx = current[0] + dy, current[1] + dx
                        if (0 <= ny < maze.shape[0] and 
                            0 <= nx < maze.shape[1] and 
                            maze[ny, nx] == 0 and
                            (ny, nx) not in self.visited):
                            valid_moves.append((i, (ny, nx)))
                    
                    # If we have valid moves, choose one
                    if valid_moves:
                        action, (ny, nx) = random.choice(valid_moves)
                        
                        # Train the model on the invalid move
                        if training_mode:
                            self.model.train(features, action, -0.5)  # Penalty for invalid move
                        
                        # Store current state for later training
                        self.previous_state = features
                        self.previous_action = action
                        
                        # Move to the new position
                        current = (ny, nx)
                        self.solution_path.append(current)
                        self.visited.add(current)
                    
                    else:
                        # No valid moves, backtrack
                        if len(self.solution_path) > 1:
                            # Remove current position from path
                            self.solution_path.pop()
                            # Get previous position
                            current = self.solution_path[-1]
                
                steps += 1
            
            # If we reached the goal and we're training, give a final reward
            if current == end and training_mode and self.previous_state is not None:
                self.model.train(self.previous_state, self.previous_action, 1.0)
                
                # Save the model after successfully solving
                if steps % 10 == 0:  # Save every 10 successful solves
                    self.model.save()
            
            # Return the solution path and empty reasoning tokens
            return self.solution_path, ["[Step]"] * len(self.solution_path)

    def run_numpy_training_version(args):
        """Run the NumPy version with ternary neural network training."""
        print(f"T_Mazer - NumPy Training Version")
        
        # Initialize solver with neural network
        solver = NumpyTrainableSolver()
        
        # Try to load an existing model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("numpy_ternary_")]
        if model_files:
            latest_model = os.path.join(MODEL_DIR, sorted(model_files)[-1])
            solver.model = NumpyTernaryModel()
            solver.model.load(latest_model)
        else:
            solver.model = NumpyTernaryModel()
        
        # Stats tracking
        stats = {
            'solved': 0,
            'total_steps': 0,
            'avg_steps': 0,
            'training_steps': solver.model.training_steps if solver.model else 0,
            'learning_rate': solver.model.learning_rate if solver.model else 0.01
        }
        
        # Continuous maze solving loop
        try:
            while True:
                # Generate a new maze
                print(f"\nGenerating maze #{stats['solved'] + 1} with dimensions {args.width}x{args.height}...")
                generator = NumpyMazeGenerator(
                    width=args.width,
                    height=args.height,
                    complexity=args.complexity,
                    density=args.density
                )
                maze = generator.generate()
                
                # Show initial maze
                print("\nInitial Maze:")
                display_numpy_maze(maze, stats=stats)
                
                # Solve maze with training
                print("\nSolving maze using neural network...")
                start_time = time.time()
                solution_path, _ = solver.solve(maze, training_mode=True)
                end_time = time.time()
                
                if not solution_path or solution_path[-1] != (maze.shape[0] - 2, maze.shape[1] - 2):
                    print("Failed to find a solution!")
                    continue
                
                # Update stats
                stats['solved'] += 1
                stats['total_steps'] += len(solution_path)
                stats['avg_steps'] = stats['total_steps'] / stats['solved']
                stats['training_steps'] = solver.model.training_steps
                stats['learning_rate'] = solver.model.learning_rate
                
                # Display solution step by step
                print(f"\nSolution found! Path length: {len(solution_path)}")
                print(f"Solving time: {end_time - start_time:.2f} seconds")
                print("Visualizing solution (press Ctrl+C to stop)...")
                
                try:
                    for i, pos in enumerate(solution_path):
                        # Clear screen
                        os.system('cls' if os.name == 'nt' else 'clear')
                        
                        print(f"Maze #{stats['solved']} - Step {i+1}/{len(solution_path)}")
                        display_numpy_maze(maze, solution_path[:i+1], pos, stats)
                        time.sleep(args.delay)
                    
                    print("\nMaze solved! Training model...")
                    
                    # Save model periodically
                    if stats['solved'] % 5 == 0:
                        solver.model.save()
                    
                    # Ask user if they want to continue
                    print("\nMaze solved successfully! Press Enter to generate a new maze...")
                    print("Press Ctrl+C to exit.")
                    input()  # Wait for user input
                
                except KeyboardInterrupt:
                    raise
                
        except KeyboardInterrupt:
            print("\nExiting training mode...")
            # Save model before exiting
            if solver.model:
                solver.model.save()

    def display_numpy_maze(maze, path=None, current_pos=None, stats=None):
        """Display a NumPy maze in text format with stats."""
        height, width = maze.shape
        
        # Convert path to a set for quick lookup
        path_set = set(path) if path else set()
        
        # Show stats if provided
        if stats:
            print("\n=== T_Mazer Stats ===")
            print(f"Mazes Solved: {stats['solved']}")
            print(f"Total Steps: {stats['total_steps']}")
            print(f"Average Steps: {stats['avg_steps']:.1f}")
            print(f"Training Steps: {stats['training_steps']}")
            print(f"Learning Rate: {stats['learning_rate']:.4f}")
            print("=====================\n")
        
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

# ===== PYGAME VERSION WITH TRAINING =====

if HAS_PYGAME and HAS_NUMPY:
    class PyGameTrainingGUI:
        """Interactive GUI for visualizing maze generation and solving with training."""
        
        # Color definitions
        COLORS = {
            'wall': (40, 40, 40),       # Dark gray
            'path': (255, 255, 255),    # White
            'start': (50, 200, 50),     # Green
            'end': (200, 50, 50),       # Red
            'current': (50, 150, 250),  # Blue
            'visited': (180, 180, 250), # Light blue
            'solution': (250, 250, 100), # Yellow
            'background': (20, 20, 20), # Very dark gray
            'text': (200, 200, 200),    # Light gray
            'highlight': (255, 165, 0)  # Orange
        }
        
        def __init__(self, cell_size=20, margin=1, window_width=None, window_height=None):
            self.cell_size = cell_size
            self.margin = margin
            self.screen = None
            self.clock = None
            self.font = None
            self.width = 0
            self.height = 0
            self.maze = None
            self.is_initialized = False
            self.custom_window_width = window_width
            self.custom_window_height = window_height
            self.stats = {
                'solved': 0,
                'total_steps': 0,
                'avg_steps': 0,
                'training_steps': 0,
                'learning_rate': 0.01
            }
            
        def initialize(self, maze, stats=None):
            """Initialize the pygame window based on maze dimensions."""
            self.maze = maze
            height, width = maze.shape
            self.width = width
            self.height = height
            
            if stats:
                self.stats = stats
            
            # Calculate window size with extra space for stats
            maze_width = width * (self.cell_size + self.margin) + self.margin
            maze_height = height * (self.cell_size + self.margin) + self.margin
            
            # Allow for stats panel on the right
            stats_width = 200
            
            # Use custom dimensions if provided
            if self.custom_window_width and self.custom_window_height:
                window_width = self.custom_window_width
                window_height = self.custom_window_height
            else:
                window_width = maze_width + stats_width
                window_height = max(maze_height, 300)  # Minimum height for stats
            
            # Initialize pygame
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("T_Mazer - Training Mode")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 14)
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
            
            # Draw stats panel
            self._draw_stats_panel()
            
            # Update display
            pygame.display.flip()
        
        def _draw_stats_panel(self):
            """Draw statistics panel on the right side."""
            # Calculate panel position
            maze_width = self.width * (self.cell_size + self.margin) + self.margin
            panel_x = maze_width + 10
            panel_y = 10
            line_height = 25
            
            # Draw stats
            stats_lines = [
                f"Mazes Solved: {self.stats['solved']}",
                f"Total Steps: {self.stats['total_steps']}",
                f"Avg Steps: {self.stats['avg_steps']:.1f}",
                f"Training Steps: {self.stats['training_steps']}",
                f"Learning Rate: {self.stats['learning_rate']:.4f}"
            ]
            
            # Draw title
            title_font = pygame.font.SysFont('Arial', 18, bold=True)
            title_surf = title_font.render("T_Mazer Stats", True, self.COLORS['highlight'])
            self.screen.blit(title_surf, (panel_x, panel_y))
            
            # Draw stats lines
            for i, line in enumerate(stats_lines):
                text_surf = self.font.render(line, True, self.COLORS['text'])
                self.screen.blit(text_surf, (panel_x, panel_y + 30 + i * line_height))
            
            # Draw instructions
            instruction_y = panel_y + 30 + len(stats_lines) * line_height + 20
            instructions = [
                "Press ENTER to restart",
                "Press SPACE to pause",
                "Press ESC to exit"
            ]
            
            instruction_title = title_font.render("Controls", True, self.COLORS['highlight'])
            self.screen.blit(instruction_title, (panel_x, instruction_y))
            
            for i, instr in enumerate(instructions):
                text_surf = self.font.render(instr, True, self.COLORS['text'])
                self.screen.blit(text_surf, (panel_x, instruction_y + 30 + i * line_height))
            
        def run_training_loop(self, generator, solver, args):
            """Run continuous training loop with maze generation and solving."""
            if not self.is_initialized:
                # Generate initial maze
                maze = generator.generate()
                self.initialize(maze)
            
            running = True
            solving = False
            solution_path = []
            current_step = 0
            visited = set()
            
            # Main game loop
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_RETURN:
                            # Generate new maze
                            maze = generator.generate()
                            self.maze = maze
                            solving = True
                            solution_path = []
                            current_step = 0
                            visited = set()
                            
                            # Solve the maze
                            start_time = time.time()
                            solution_path, _ = solver.solve(maze, training_mode=True)
                            end_time = time.time()
                            
                            # Update stats
                            if solution_path and solution_path[-1] == (maze.shape[0] - 2, maze.shape[1] - 2):
                                self.stats['solved'] += 1
                                self.stats['total_steps'] += len(solution_path)
                                self.stats['avg_steps'] = self.stats['total_steps'] / self.stats['solved']
                            
                            # Update training stats
                            self.stats['training_steps'] = solver.model.training_steps
                            self.stats['learning_rate'] = solver.model.learning_rate
                            
                            # Save model periodically
                            if self.stats['solved'] % 10 == 0:
                                solver.model.save()
                        
                        elif event.key == pygame.K_SPACE:
                            # Toggle solving
                            solving = not solving
                
                # Clear screen
                self.screen.fill(self.COLORS['background'])
                
                # If we're solving, advance to the next step
                if solving and solution_path and current_step < len(solution_path):
                    pos = solution_path[current_step]
                    visited.add(pos)
                    
                    # Draw the maze with the current state
                    self.draw_maze(visited, solution_path[:current_step+1], pos)
                    
                    # Advance to next step
                    current_step += 1
                    
                    # If we've reached the end, wait a bit
                    if current_step >= len(solution_path):
                        # Display completion message
                        font = pygame.font.SysFont('Arial', 24, bold=True)
                        text = font.render("Maze Solved! Press ENTER for new maze", True, self.COLORS['highlight'])
                        text_rect = text.get_rect(center=(self.width * (self.cell_size + self.margin) // 2, 
                                                          self.height * (self.cell_size + self.margin) + 30))
                        self.screen.blit(text, text_rect)
                        pygame.display.flip()
                        
                        # Wait a bit
                        pygame.time.wait(1000)
                        
                        # Generate a new maze automatically
                        maze = generator.generate()
                        self.maze = maze
                        solving = True
                        solution_path = []
                        current_step = 0
                        visited = set()
                        
                        # Solve the new maze
                        start_time = time.time()
                        solution_path, _ = solver.solve(maze, training_mode=True)
                        end_time = time.time()
                        
                        # Update stats
                        if solution_path and solution_path[-1] == (maze.shape[0] - 2, maze.shape[1] - 2):
                            self.stats['solved'] += 1
                            self.stats['total_steps'] += len(solution_path)
                            self.stats['avg_steps'] = self.stats['total_steps'] / self.stats['solved']
                        
                        # Update training stats
                        self.stats['training_steps'] = solver.model.training_steps
                        self.stats['learning_rate'] = solver.model.learning_rate
                        
                        # Save model periodically
                        if self.stats['solved'] % 10 == 0:
                            solver.model.save()
                else:
                    # Just draw the current state
                    self.draw_maze(visited, solution_path[:current_step] if current_step > 0 else [],
                                   solution_path[current_step-1] if current_step > 0 else None)
                
                # Cap the frame rate
                self.clock.tick(10)
                
                # Add a small delay for visualization
                pygame.time.wait(int(args.delay * 1000))
            
            # Clean up
            pygame.quit()
        
        def close(self):
            """Close the GUI and clean up resources."""
            if self.is_initialized:
                pygame.quit()
                self.is_initialized = False

    def run_pygame_training_version(args):
        """Run the pygame version with neural network training."""
        print("T_Mazer - PyGame Training Version")
        
        # Initialize maze generator
        generator = NumpyMazeGenerator(
            width=args.width,
            height=args.height,
            complexity=args.complexity,
            density=args.density
        )
        
        # Initialize solver with neural network
        solver = NumpyTrainableSolver()
        
        # Try to load an existing model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("numpy_ternary_")]
        if model_files:
            latest_model = os.path.join(MODEL_DIR, sorted(model_files)[-1])
            solver.model = NumpyTernaryModel()
            solver.model.load(latest_model)
        else:
            solver.model = NumpyTernaryModel()
        
        # Initialize GUI
        gui = PyGameTrainingGUI(cell_size=args.cell_size)
        
        # Generate initial maze
        maze = generator.generate()
        
        # Initialize stats
        stats = {
            'solved': 0,
            'total_steps': 0,
            'avg_steps': 0,
            'training_steps': solver.model.training_steps,
            'learning_rate': solver.model.learning_rate
        }
        
        # Initialize GUI with maze and stats
        gui.initialize(maze, stats)
        
        # Run the training loop
        try:
            gui.run_training_loop(generator, solver, args)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            # Save model before exiting
            if solver.model:
                solver.model.save()
            gui.close()

# ===== MAIN APPLICATION =====

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="T_Mazer - Enhanced Version with Training")
    
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
    
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Disable neural network training"
    )
    
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to a model file to load"
    )
    
    return parser.parse_args()

def main():
    """Main application entry point."""
    print("\nT_Mazer Enhanced - With Model Training and Continuous Solving\n")
    
    # Parse command line arguments
    args = parse_args()
    
    # Determine which version to run
    if args.force_simple:
        print("Using Simple version with training (forced by --force-simple)")
        run_simple_training_version(args)
    elif args.force_text:
        if HAS_NUMPY:
            print("Using Text version with training (forced by --force-text)")
            run_numpy_training_version(args)
        else:
            print("Cannot run Text version without NumPy. Falling back to Simple version.")
            run_simple_training_version(args)
    elif HAS_PYGAME and HAS_NUMPY:
        print("Using Graphical training version (pygame available)")
        run_pygame_training_version(args)
    elif HAS_NUMPY:
        print("Using Text version with training (pygame not available)")
        run_numpy_training_version(args)
    else:
        print("Using Simple version with training (no dependencies available)")
        run_simple_training_version(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting training mode...")
        sys.exit(0)
