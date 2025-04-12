"""
Neural Test Mode - Implementations for testing neural network model without algorithmic fallbacks.
"""
import os
import time
import numpy as np
import pygame

# Check if required dependencies are available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

class NeuralTestModeSolver:
    """
    A solver that uses only neural network predictions to solve mazes.
    This class uses the trained weights without any algorithmic fallbacks
    to test how well the model has learned.
    """
    
    def __init__(self, model):
        """
        Initialize with an existing model.
        
        Args:
            model: Trained neural network model
        """
        self.model = model
        self.solution_path = []
        self.visited = set()
        self.test_stats = {
            'total_moves': 0,
            'valid_moves': 0,
            'invalid_moves': 0,
            'backtracks': 0,
            'success_rate': 0.0
        }
    
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
    
    def solve(self, maze):
        """
        Solve the maze using ONLY the neural network's predictions.
        Unlike the regular solver, this one does not use any algorithmic
        fallbacks or heuristics - it purely tests what the model has learned.
        
        Args:
            maze: NumPy array representing the maze
            
        Returns:
            list: Solution path
            dict: Test statistics
        """
        # Reset state
        self.solution_path = []
        self.visited = set()
        self.test_stats = {
            'total_moves': 0,
            'valid_moves': 0,
            'invalid_moves': 0,
            'backtracks': 0,
            'success_rate': 0.0
        }
        
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
        
        # Main solving loop - using ONLY neural network
        while current != end and steps < max_steps:
            # Get state features
            features = self._get_features(maze, current, end)
            
            # Get model prediction - the model decides where to go
            action = self.model.predict(features)
            
            # Increment total moves
            self.test_stats['total_moves'] += 1
            
            # Try to move in the predicted direction
            dy, dx = directions[action]
            ny, nx = current[0] + dy, current[1] + dx
            
            # Check if this is a valid move
            valid_move = (0 <= ny < maze.shape[0] and 
                          0 <= nx < maze.shape[1] and 
                          maze[ny, nx] == 0 and
                          (ny, nx) not in self.visited)
            
            # If the move is valid, take it
            if valid_move:
                # Valid prediction by the neural network
                self.test_stats['valid_moves'] += 1
                
                # Move to the new position
                current = (ny, nx)
                self.solution_path.append(current)
                self.visited.add(current)
            else:
                # Invalid prediction - we have two options:
                # 1. Backtrack (if possible)
                # 2. Give up (if we can't backtrack)
                self.test_stats['invalid_moves'] += 1
                
                if len(self.solution_path) > 1:
                    # Backtrack to previous position
                    self.test_stats['backtracks'] += 1
                    self.solution_path.pop()
                    current = self.solution_path[-1]
                else:
                    # Can't backtrack further, we're stuck at the start
                    break
            
            steps += 1
        
        # Calculate success rate
        if self.test_stats['total_moves'] > 0:
            self.test_stats['success_rate'] = (self.test_stats['valid_moves'] / 
                                             self.test_stats['total_moves']) * 100
        
        # Check if we reached the goal
        solution_found = (current == end)
        
        return self.solution_path, self.test_stats, solution_found

class TestModeUI:
    """UI helper for test mode visualization."""
    
    @staticmethod
    def print_test_stats(stats, current_maze=None, total_mazes=None):
        """Print test statistics in a formatted way."""
        print("\n=== Neural Network Test Mode ===")
        if current_maze is not None and total_mazes is not None:
            print(f"Maze: {current_maze}/{total_mazes}")
        print(f"Total Moves: {stats['total_moves']}")
        print(f"Valid Predictions: {stats['valid_moves']} ({stats['success_rate']:.1f}%)")
        print(f"Invalid Predictions: {stats['invalid_moves']}")
        print(f"Backtracks: {stats['backtracks']}")
        print("===============================\n")
    
    @staticmethod
    def wait_for_key(key_prompt="Press Enter to continue, Q to quit..."):
        """Wait for a key press with a prompt."""
        print(key_prompt)
        try:
            response = input().strip().lower()
            return response != 'q'
        except KeyboardInterrupt:
            return False

class TestModeRunner:
    """
    Test mode runner to evaluate neural network performance
    on multiple mazes without algorithmic help.
    """
    
    def __init__(self, model, generator, num_mazes=10):
        """
        Initialize the test runner.
        
        Args:
            model: Trained neural network model
            generator: Maze generator
            num_mazes: Number of mazes to test
        """
        self.model = model
        self.generator = generator
        self.num_mazes = num_mazes
        self.solver = NeuralTestModeSolver(model)
        self.overall_stats = {
            'mazes_solved': 0,
            'mazes_attempted': 0,
            'total_moves': 0,
            'valid_moves': 0,
            'invalid_moves': 0,
            'backtracks': 0,
            'success_rate': 0.0,
            'maze_success_rate': 0.0
        }
    
    def run_text_mode(self, args):
        """Run test mode with text visualization."""
        print("\n=== Running Neural Network Test Mode ===")
        print(f"Testing on {self.num_mazes} mazes without algorithmic fallbacks")
        print("This will show how well the model has learned to solve mazes on its own")
        
        # Wait for user to start
        if not TestModeUI.wait_for_key("Press Enter to start the test, Q to quit..."):
            return
        
        # Run tests on multiple mazes
        for i in range(self.num_mazes):
            # Generate a new maze
            print(f"\nGenerating test maze #{i+1}/{self.num_mazes}...")
            maze = self.generator.generate()
            
            # Show initial maze
            from t_mazer_enhanced import display_numpy_maze
            print("\nTest Maze:")
            display_numpy_maze(maze)
            
            # Solve using only neural network
            print("\nSolving with neural network only (no algorithmic fallbacks)...")
            start_time = time.time()
            solution_path, test_stats, solved = self.solver.solve(maze)
            end_time = time.time()
            
            # Update overall stats
            self.overall_stats['mazes_attempted'] += 1
            if solved:
                self.overall_stats['mazes_solved'] += 1
            self.overall_stats['total_moves'] += test_stats['total_moves']
            self.overall_stats['valid_moves'] += test_stats['valid_moves']
            self.overall_stats['invalid_moves'] += test_stats['invalid_moves']
            self.overall_stats['backtracks'] += test_stats['backtracks']
            
            # Print results
            TestModeUI.print_test_stats(test_stats, i+1, self.num_mazes)
            
            if solved:
                print(f"Maze SOLVED using only neural network! Path length: {len(solution_path)}")
            else:
                print("Neural network FAILED to solve this maze without algorithmic help")
            
            print(f"Solving time: {end_time - start_time:.2f} seconds")
            
            # Visualize solution if found
            if solved:
                print("\nVisualizing solution (press Ctrl+C to stop)...")
                try:
                    for j, pos in enumerate(solution_path):
                        # Clear screen
                        os.system('cls' if os.name == 'nt' else 'clear')
                        
                        print(f"Neural Network Test - Maze {i+1}/{self.num_mazes} - Step {j+1}/{len(solution_path)}")
                        TestModeUI.print_test_stats(test_stats)
                        display_numpy_maze(maze, solution_path[:j+1], pos)
                        time.sleep(args.delay)
                except KeyboardInterrupt:
                    print("\nVisualization stopped by user")
            
            # Calculate overall success rates
            if self.overall_stats['total_moves'] > 0:
                self.overall_stats['success_rate'] = (self.overall_stats['valid_moves'] / 
                                                    self.overall_stats['total_moves']) * 100
            
            if self.overall_stats['mazes_attempted'] > 0:
                self.overall_stats['maze_success_rate'] = (self.overall_stats['mazes_solved'] / 
                                                         self.overall_stats['mazes_attempted']) * 100
            
            # Wait for user to continue
            if i < self.num_mazes - 1:  # Not the last maze
                if not TestModeUI.wait_for_key():
                    break
        
        # Show final statistics
        print("\n=== Neural Network Test Results ===")
        print(f"Mazes Solved: {self.overall_stats['mazes_solved']}/{self.overall_stats['mazes_attempted']} ({self.overall_stats['maze_success_rate']:.1f}%)")
        print(f"Total Moves: {self.overall_stats['total_moves']}")
        print(f"Valid Predictions: {self.overall_stats['valid_moves']} ({self.overall_stats['success_rate']:.1f}%)")
        print(f"Invalid Predictions: {self.overall_stats['invalid_moves']}")
        print(f"Backtracks: {self.overall_stats['backtracks']}")
        print("==================================")
    
    def run_pygame_mode(self, args):
        """Run test mode with pygame visualization."""
        if not HAS_PYGAME:
            print("Error: Pygame is required for graphical test mode")
            return
        
        print("\n=== Running Neural Network Test Mode (Pygame) ===")
        print(f"Testing on {self.num_mazes} mazes without algorithmic fallbacks")
        
        # Initialize pygame
        pygame.init()
        
        # Define colors
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
            'highlight': (255, 165, 0),  # Orange
            'test_mode': (255, 0, 255)   # Purple - specific for test mode
        }
        
        # Set up window and font
        cell_size = args.cell_size
        margin = 1
        width, height = args.width, args.height
        
        # Calculate window size with extra space for stats
        maze_width = width * (cell_size + margin) + margin
        maze_height = height * (cell_size + margin) + margin
        stats_width = 300
        
        window_width = maze_width + stats_width
        window_height = max(maze_height, 400)
        
        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("T_Mazer - Neural Network Test Mode")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 14)
        title_font = pygame.font.SysFont('Arial', 18, bold=True)
        
        # Helper function to draw maze and stats
        def draw_maze_and_stats(maze, path=None, current_pos=None, stats=None, maze_num=None, solved=None):
            # Fill background
            screen.fill(COLORS['background'])
            
            # Convert to sets/lists if None
            path_set = set(path) if path else set()
            
            # Draw each cell
            for y in range(height):
                for x in range(width):
                    # Calculate position
                    rect_x = x * (cell_size + margin) + margin
                    rect_y = y * (cell_size + margin) + margin
                    rect = pygame.Rect(rect_x, rect_y, cell_size, cell_size)
                    
                    # Determine cell color
                    if (y, x) == (1, 1):  # Start position
                        color = COLORS['start']
                    elif (y, x) == (height - 2, width - 2):  # End position
                        color = COLORS['end']
                    elif current_pos and (y, x) == current_pos:  # Current position
                        color = COLORS['current']
                    elif (y, x) in path_set:  # Solution path
                        color = COLORS['solution']
                    elif (y, x) in self.solver.visited:  # Visited position
                        color = COLORS['visited']
                    elif maze[y, x] == 1:  # Wall
                        color = COLORS['wall']
                    else:  # Path
                        color = COLORS['path']
                    
                    # Draw the cell
                    pygame.draw.rect(screen, color, rect)
            
            # Draw stats panel
            panel_x = maze_width + 10
            panel_y = 10
            line_height = 25
            
            # Draw title with test mode indicator
            title = "Neural Network Test Mode"
            title_surf = title_font.render(title, True, COLORS['test_mode'])
            screen.blit(title_surf, (panel_x, panel_y))
            
            # Draw maze number if available
            if maze_num is not None:
                maze_text = f"Maze: {maze_num}/{self.num_mazes}"
                maze_surf = font.render(maze_text, True, COLORS['highlight'])
                screen.blit(maze_surf, (panel_x, panel_y + 30))
            
            # Draw stats
            if stats:
                stats_y = panel_y + 60
                stats_lines = [
                    f"Total Moves: {stats['total_moves']}",
                    f"Valid Predictions: {stats['valid_moves']}",
                    f"Prediction Success: {stats['success_rate']:.1f}%",
                    f"Invalid Predictions: {stats['invalid_moves']}",
                    f"Backtracks: {stats['backtracks']}"
                ]
                
                for i, line in enumerate(stats_lines):
                    text_surf = font.render(line, True, COLORS['text'])
                    screen.blit(text_surf, (panel_x, stats_y + i * line_height))
            
            # Draw overall stats
            if self.overall_stats['mazes_attempted'] > 0:
                overall_y = panel_y + 200
                overall_title = "Overall Results"
                overall_surf = title_font.render(overall_title, True, COLORS['highlight'])
                screen.blit(overall_surf, (panel_x, overall_y))
                
                overall_lines = [
                    f"Mazes Solved: {self.overall_stats['mazes_solved']}/{self.overall_stats['mazes_attempted']}",
                    f"Maze Success: {self.overall_stats['maze_success_rate']:.1f}%",
                    f"Move Success: {self.overall_stats['success_rate']:.1f}%"
                ]
                
                for i, line in enumerate(overall_lines):
                    text_surf = font.render(line, True, COLORS['text'])
                    screen.blit(text_surf, (panel_x, overall_y + 30 + i * line_height))
            
            # Draw solution status if available
            if solved is not None:
                status_y = panel_y + 320
                status_text = "SOLVED!" if solved else "FAILED"
                status_color = COLORS['highlight'] if solved else COLORS['end']
                status_surf = title_font.render(status_text, True, status_color)
                screen.blit(status_surf, (panel_x, status_y))
            
            # Draw instructions
            instruction_y = window_height - 80
            instructions = [
                "Press ENTER for next maze",
                "Press SPACE to pause/resume",
                "Press ESC to exit"
            ]
            
            for i, instr in enumerate(instructions):
                text_surf = font.render(instr, True, COLORS['text'])
                screen.blit(text_surf, (panel_x, instruction_y + i * line_height))
            
            # Update display
            pygame.display.flip()
        
        # Run test mode
        maze_num = 0
        running = True
        paused = False
        solving = False
        current_maze = None
        solution_path = []
        current_step = 0
        solved = None
        
        # Generate first maze
        maze_num = 1
        current_maze = self.generator.generate()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RETURN and not solving:
                        # Go to next maze
                        if maze_num < self.num_mazes:
                            maze_num += 1
                            current_maze = self.generator.generate()
                            solving = True
                            solution_path = []
                            current_step = 0
                            solved = None
                        else:
                            # Show final results
                            solving = False
                    elif event.key == pygame.K_SPACE:
                        # Toggle pause
                        paused = not paused
            
            # Start solving if not started
            if current_maze is not None and not paused and not solution_path:
                solution_path, test_stats, solved = self.solver.solve(current_maze)
                
                # Update overall stats
                self.overall_stats['mazes_attempted'] += 1
                if solved:
                    self.overall_stats['mazes_solved'] += 1
                self.overall_stats['total_moves'] += test_stats['total_moves']
                self.overall_stats['valid_moves'] += test_stats['valid_moves']
                self.overall_stats['invalid_moves'] += test_stats['invalid_moves']
                self.overall_stats['backtracks'] += test_stats['backtracks']
                
                # Calculate overall success rates
                if self.overall_stats['total_moves'] > 0:
                    self.overall_stats['success_rate'] = (self.overall_stats['valid_moves'] / 
                                                        self.overall_stats['total_moves']) * 100
                
                if self.overall_stats['mazes_attempted'] > 0:
                    self.overall_stats['maze_success_rate'] = (self.overall_stats['mazes_solved'] / 
                                                            self.overall_stats['mazes_attempted']) * 100
                
                solving = True
                current_step = 0
            
            # Advance visualization if solving
            if solving and not paused and solution_path and current_step < len(solution_path):
                current_pos = solution_path[current_step]
                
                # Draw current state
                draw_maze_and_stats(
                    current_maze, 
                    solution_path[:current_step+1], 
                    current_pos,
                    self.solver.test_stats,
                    maze_num,
                    solved
                )
                
                # Advance to next step
                current_step += 1
                
                # If we've finished, wait for user
                if current_step >= len(solution_path):
                    solving = False
            else:
                # Just draw current state
                current_pos = solution_path[current_step-1] if current_step > 0 and solution_path else None
                draw_maze_and_stats(
                    current_maze,
                    solution_path[:current_step] if current_step > 0 else [],
                    current_pos,
                    self.solver.test_stats,
                    maze_num,
                    solved
                )
            
            # Cap the frame rate
            clock.tick(10)
            
            # Add delay for visualization
            pygame.time.wait(int(args.delay * 1000))
        
        # Clean up pygame
        pygame.quit()
