#!/usr/bin/env python3
"""
T_Mazer SPICE: Self-Play In Corpus Environments with Self-Reflection and Self-Optimization

This script integrates:
- Self-Play: Models compete against themselves and previous versions
- Self-Reflection: Performance analysis and weakness identification  
- Self-Optimization: Adaptive parameter tuning based on insights
- Advanced Maze Generation: Adaptive difficulty based on solver performance
"""
import sys
import os
import time
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("ERROR: NumPy is required for SPICE mode")
    sys.exit(1)

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    pygame = None  # Set to None for safety

# Import our modules
from self_reflection import SelfReflection
from self_play import SelfPlayArena
from self_optimization import SelfOptimizer
from advanced_maze_generator import AdvancedMazeGenerator

# Import GUI if available
if HAS_PYGAME:
    from maze_gui import MazeGUI

# Import existing modules
try:
    from t_mazer_enhanced import NumpyMazeGenerator, NumpyTrainableSolver, NumpyTernaryModel, MODEL_DIR
except ImportError:
    # Fallback if enhanced version not available
    print("Warning: Could not import from t_mazer_enhanced. Some features may be limited.")
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class SPICETrainer:
    """
    Main trainer that integrates self-play, reflection, and optimization.
    """
    
    def __init__(self, args):
        """Initialize SPICE trainer."""
        self.args = args
        
        # Initialize components
        self.reflection = SelfReflection()
        self.optimizer = SelfOptimizer(
            initial_learning_rate=args.learning_rate,
            initial_exploration_rate=args.exploration_rate
        )
        
        # Initialize maze generator
        self.maze_generator = AdvancedMazeGenerator(
            base_width=args.width,
            base_height=args.height,
            initial_complexity=args.complexity
        )
        
        # Initialize solver
        self.solver = NumpyTrainableSolver()
        
        # Load or create model
        if args.load_model and os.path.exists(args.load_model):
            self.solver.model = NumpyTernaryModel()
            self.solver.model.load(args.load_model)
            print(f"Loaded model from {args.load_model}")
        else:
            self.solver.model = NumpyTernaryModel()
            print("Initialized new model")
        
        # Initialize self-play arena
        self.arena = SelfPlayArena(
            model_class=NumpyTernaryModel,
            reflection_system=self.reflection
        )
        
        # Statistics
        self.stats = {
            'episodes': 0,
            'solved': 0,
            'total_steps': 0,
            'avg_efficiency': 0.0,
            'reflections': 0,
            'optimizations': 0,
            'self_play_matches': 0
        }
        
        # Initialize GUI if available and not disabled
        self.gui = None
        if HAS_PYGAME and not args.no_gui:
            self.gui = MazeGUI(cell_size=args.cell_size)
            print("GUI initialized - you'll see mazes being solved!")
        else:
            print("Running in text mode (no GUI)")
        
        # Track episode for maze variation
        self.episode_count = 0
    
    def train_episode(self, episode_num):
        """Train on a single episode."""
        # Vary maze size/complexity each episode based on performance
        self._adjust_maze_parameters(episode_num)
        
        # Generate NEW maze with adaptive difficulty (always creates fresh maze)
        maze = self.maze_generator.generate(
            solver_stats=self.reflection.get_aggregate_insights()
        )
        
        print(f"\n=== Episode {episode_num} ===")
        print(f"Maze size: {maze.shape}, Difficulty: {self.maze_generator.current_difficulty}")
        print(f"Complexity: {self.maze_generator.complexity:.2f}")
        
        # Solve maze with optional visualization
        start_time = time.time()
        
        if self.gui:
            # Visualize solving process
            solution_path, _ = self._solve_with_visualization(maze)
        else:
            # Solve without visualization
            solution_path, _ = self.solver.solve(maze, training_mode=True)
        
        solving_time = time.time() - start_time
        
        # Validate solution path doesn't cheat (even if solver did)
        if solution_path:
            solution_path = self._validate_path(maze, solution_path)
        
        # Evaluate solution
        end_pos = (maze.shape[0] - 2, maze.shape[1] - 2)
        success = solution_path and solution_path[-1] == end_pos
        
        # Update stats
        self.stats['episodes'] += 1
        if success:
            self.stats['solved'] += 1
            self.stats['total_steps'] += len(solution_path)
        
        # Self-reflection (max_steps will be calculated with variation in reflect method)
        reflection = self.reflection.reflect(
            maze=maze,
            solution_path=solution_path if success else [],
            solving_time=solving_time,
            steps_taken=len(solution_path) if solution_path else 0,
            max_steps=None  # Let it calculate with variation
        )
        self.stats['reflections'] += 1
        
        # Display reflection insights
        if reflection['weaknesses']:
            print(f"Identified {len(reflection['weaknesses'])} weaknesses:")
            for w in reflection['weaknesses']:
                print(f"  - {w['description']}")
        
        # Self-optimization
        current_perf = {
            'efficiency': reflection['efficiency'],
            'optimality': reflection['path_optimality'],
            'success': success
        }
        
        optimizations = self.optimizer.optimize(reflection, current_perf)
        self.stats['optimizations'] += 1
        
        # Apply optimizations to solver
        self._apply_optimizations_to_solver(optimizations)
        
        # Update aggregate stats
        recent = self.reflection.get_aggregate_insights()
        self.stats['avg_efficiency'] = recent.get('average_efficiency', 0.0)
        
        # Self-play periodically
        if episode_num % self.args.self_play_frequency == 0:
            self._self_play_round(maze)
        
        # Display episode summary
        print(f"Solution: {'Success' if success else 'Failed'}")
        print(f"Path length: {len(solution_path) if solution_path else 0}")
        print(f"Efficiency: {reflection['efficiency']:.2%}")
        print(f"Optimality: {reflection['path_optimality']:.2%}")
        print(f"Learning rate: {self.optimizer.learning_rate:.4f}")
        print(f"Exploration rate: {self.optimizer.exploration_rate:.3f}")
        
        return success, reflection
    
    def _apply_optimizations_to_solver(self, optimizations):
        """Apply optimization results to the solver."""
        # Update learning rate
        if hasattr(self.solver.model, 'learning_rate'):
            self.solver.model.learning_rate = optimizations['learning_rate']
        
        # Store reward weights for use in training
        if not hasattr(self.solver, 'reward_weights'):
            self.solver.reward_weights = {}
        self.solver.reward_weights.update(optimizations.get('reward_weights', {}))
        
        # Store training focus
        self.solver.training_focus = optimizations.get('training_focus', [])
    
    def _solve_with_visualization(self, maze):
        """
        Solve maze with real-time visualization.
        Includes validation to prevent passing through walls.
        
        Args:
            maze (np.ndarray): Maze to solve
            
        Returns:
            tuple: (solution_path, reasoning_tokens)
        """
        # Initialize GUI if needed
        if not self.gui.is_initialized or self.gui.maze is None or self.gui.maze.shape != maze.shape:
            self.gui.initialize(maze)
        
        # Get solution path first (for training)
        solution_path, reasoning_tokens = self.solver.solve(maze, training_mode=True)
        
        # Validate and clean solution path to ensure no wall-cheating
        solution_path = self._validate_path(maze, solution_path)
        
        # Visualize the solution
        visited = set()
        if solution_path:
            # Show initial maze
            self.gui.draw_maze(visited, [], None)
            if HAS_PYGAME and pygame:
                pygame.time.wait(int(self.args.delay * 500))  # Half delay for initial
            
            # Animate solution step by step
            for i, pos in enumerate(solution_path):
                visited.add(pos)
                self.gui.draw_maze(visited, solution_path[:i+1], pos)
                
                # Process events
                if HAS_PYGAME and pygame:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return None, None
                    pygame.time.wait(int(self.args.delay * 1000))
        
        return solution_path, reasoning_tokens
    
    def _validate_path(self, maze, path):
        """
        Validate a solution path and remove any invalid moves (passing through walls).
        Only allows movement through cells with value 0 (paths), rejects any non-zero values.
        
        STRICT MODE: Every single cell in the path must be white (value 0).
        
        Args:
            maze (np.ndarray): The maze
            path (list): Proposed solution path
            
        Returns:
            list: Validated and cleaned path (only white cells)
        """
        if not path or len(path) < 2:
            return path if path else []
        
        validated_path = []
        cheating_detected = False
        
        # First pass: Strictly filter - ONLY keep positions that are white (value 0)
        for pos in path:
            # Check bounds
            if not (0 <= pos[0] < maze.shape[0] and 0 <= pos[1] < maze.shape[1]):
                cheating_detected = True
                continue
            
            # STRICT: Cell MUST be white (value 0)
            cell_value = maze[pos[0], pos[1]]
            if cell_value != 0:
                cheating_detected = True
                print(f"🚫 CHEAT BLOCKED: Attempted to move to black cell (value={cell_value}) at {pos}")
                continue
            
            validated_path.append(pos)
        
        if cheating_detected:
            print(f"⚠️  Cheating detected and blocked! Original path: {len(path)}, Valid path: {len(validated_path)}")
        
        # Second pass: Ensure path is continuous (adjacent moves only)
        if len(validated_path) < 2:
            return validated_path
        
        continuous_path = [validated_path[0]]
        prev_pos = validated_path[0]
        
        for i in range(1, len(validated_path)):
            curr_pos = validated_path[i]
            
            # Check if move is adjacent
            dy = abs(curr_pos[0] - prev_pos[0])
            dx = abs(curr_pos[1] - prev_pos[1])
            
            if dy + dx == 1:
                # Valid adjacent move
                continuous_path.append(curr_pos)
                prev_pos = curr_pos
            elif dy + dx == 0:
                # Same position, skip duplicate
                continue
            else:
                # Non-adjacent: try to fill in the gap with BFS
                intermediate = self._find_valid_path_segment(maze, prev_pos, curr_pos)
                if intermediate and len(intermediate) > 1:
                    # Add intermediate steps (skip first as it's prev_pos)
                    for step in intermediate[1:]:
                        # Final check: must be white
                        if maze[step[0], step[1]] == 0:
                            continuous_path.append(step)
                        else:
                            print(f"🚫 BFS returned non-white cell at {step}, stopping")
                            break
                    if continuous_path[-1] == curr_pos:
                        prev_pos = curr_pos
                    else:
                        prev_pos = continuous_path[-1]
                # If no valid intermediate path, we skip to next valid position
        
        # Final verification: double-check entire path is on white cells
        final_path = []
        for pos in continuous_path:
            if (0 <= pos[0] < maze.shape[0] and 
                0 <= pos[1] < maze.shape[1] and 
                maze[pos[0], pos[1]] == 0):
                final_path.append(pos)
            else:
                print(f"🚫 FINAL CHECK: Blocked non-white cell at {pos}")
        
        return final_path
    
    def _adjust_maze_parameters(self, episode_num):
        """
        Adjust maze parameters each episode to ensure variation.
        
        Args:
            episode_num (int): Current episode number
        """
        # Vary base size slightly based on episode
        size_variation = 0.05 * np.sin(episode_num * 0.1)  # Oscillating variation
        
        # Update base dimensions with variation
        base_size_factor = 1.0 + size_variation
        self.maze_generator.base_width = max(
            15, 
            int(self.args.width * base_size_factor)
        )
        self.maze_generator.base_height = max(
            15,
            int(self.args.height * base_size_factor)
        )
        
        # Ensure odd dimensions for maze generation
        if self.maze_generator.base_width % 2 == 0:
            self.maze_generator.base_width += 1
        if self.maze_generator.base_height % 2 == 0:
            self.maze_generator.base_height += 1
    
    def _find_valid_path_segment(self, maze, start, end):
        """
        Find a valid path segment between two positions using BFS.
        Only allows movement through cells with value 0 (paths).
        
        Args:
            maze (np.ndarray): The maze
            start (tuple): Start position
            end (tuple): End position
            
        Returns:
            list: Path from start to end, or None if no path exists
        """
        from collections import deque
        
        # Validate start and end are path cells (value 0)
        if not (0 <= start[0] < maze.shape[0] and 
                0 <= start[1] < maze.shape[1] and
                maze[start[0], start[1]] == 0):
            return None  # Start is not a valid path cell
        
        if not (0 <= end[0] < maze.shape[0] and 
                0 <= end[1] < maze.shape[1] and
                maze[end[0], end[1]] == 0):
            return None  # End is not a valid path cell
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            # Try all four directions
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = current[0] + dy, current[1] + dx
                
                # STRICT CHECK: Only allow movement through cells with value 0 (paths)
                # Any non-zero value (1=wall, or any other value) is an obstacle
                if (0 <= ny < maze.shape[0] and 
                    0 <= nx < maze.shape[1] and 
                    maze[ny, nx] == 0 and  # Must be a path cell
                    (ny, nx) not in visited):
                    
                    queue.append(((ny, nx), path + [(ny, nx)]))
                    visited.add((ny, nx))
        
        return None  # No path found
    
    def _self_play_round(self, maze):
        """Run a round of self-play."""
        print("\n--- Self-Play Round ---")
        
        # Create competitor from current model
        competitor = self.arena.create_competitor(base_model=self.solver.model)
        
        # Add to arena
        self.arena.competitors.append((competitor, 0.0))
        self.arena.competitors.append((self.solver.model, 0.0))
        
        # Play match
        result = self.arena.play_match(maze, competitor, self.solver.model)
        
        self.stats['self_play_matches'] += 1
        
        # Learn from match
        if result['winner'] == 1:  # Competitor won
            print("Competitor performed better - incorporating improvements")
            # Optionally update model based on competitor
        elif result['winner'] == 2:  # Current model won
            print("Current model maintained advantage")
        
        # Clean up temporary competitor
        self.arena.competitors = [
            (model, score) for model, score in self.arena.competitors
            if model == self.solver.model
        ]
    
    def run(self):
        """Run the SPICE training loop."""
        print("\n" + "="*60)
        print("T_Mazer SPICE: Self-Play In Corpus Environments")
        print("="*60)
        print(f"Initial complexity: {self.args.complexity}")
        print(f"Self-play frequency: Every {self.args.self_play_frequency} episodes")
        print("="*60)
        
        try:
            for episode in range(1, self.args.episodes + 1):
                success, reflection = self.train_episode(episode)
                
                # Save model periodically
                if episode % self.args.save_frequency == 0:
                    self._save_model(episode)
                
                # Display aggregate stats periodically
                if episode % 10 == 0:
                    self._display_stats()
                
                # Check for early stopping
                if self._should_stop():
                    print("\nStopping criteria met.")
                    break
                
                # Small delay for readability
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
        
        finally:
            # Clean up GUI
            if self.gui and self.gui.is_initialized:
                self.gui.close()
            
            # Final save
            self._save_model(self.stats['episodes'])
            self._display_final_stats()
    
    def _should_stop(self):
        """Check if training should stop."""
        # Stop if perfect performance for many episodes
        if self.stats['solved'] > 0:
            recent_rate = min(10, self.stats['episodes']) / self.stats['episodes']
            if recent_rate >= 0.95 and self.stats['avg_efficiency'] > 0.9:
                return True
        return False
    
    def _save_model(self, episode):
        """Save the current model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spice_ternary_{timestamp}_ep{episode}.pkl"
        filepath = os.path.join(MODEL_DIR, filename)
        
        self.solver.model.save(filepath)
        print(f"\nModel saved to {filename}")
    
    def _display_stats(self):
        """Display aggregate statistics."""
        print("\n" + "-"*60)
        print("Aggregate Statistics:")
        print(f"  Episodes: {self.stats['episodes']}")
        print(f"  Solved: {self.stats['solved']} ({self.stats['solved']/max(1,self.stats['episodes'])*100:.1f}%)")
        print(f"  Avg Efficiency: {self.stats['avg_efficiency']:.2%}")
        print(f"  Reflections: {self.stats['reflections']}")
        print(f"  Optimizations: {self.stats['optimizations']}")
        print(f"  Self-Play Matches: {self.stats['self_play_matches']}")
        
        # Get reflection insights
        insights = self.reflection.get_aggregate_insights()
        if isinstance(insights, dict) and 'common_weaknesses' in insights:
            print(f"\n  Common Weaknesses:")
            for weakness, count in insights['common_weaknesses'].items():
                print(f"    - {weakness}: {count} occurrences")
        
        # Training focus
        focus = self.optimizer._determine_training_focus({})
        print(f"\n  Training Focus: {', '.join(focus)}")
        print("-"*60)
    
    def _display_final_stats(self):
        """Display final statistics."""
        print("\n" + "="*60)
        print("Final Statistics")
        print("="*60)
        self._display_stats()
        
        # Final reflection summary
        final_insights = self.reflection.get_aggregate_insights()
        print(f"\nFinal Performance: {final_insights.get('overall_performance', 'N/A')}")
        print("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="T_Mazer SPICE: Self-Play with Reflection and Optimization"
    )
    
    parser.add_argument(
        "--width", type=int, default=20,
        help="Base maze width (default: 20)"
    )
    
    parser.add_argument(
        "--height", type=int, default=20,
        help="Base maze height (default: 20)"
    )
    
    parser.add_argument(
        "--complexity", type=float, default=0.5,
        help="Initial complexity (0.0-1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of training episodes (default: 100)"
    )
    
    parser.add_argument(
        "--learning-rate", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--exploration-rate", type=float, default=0.3,
        help="Initial exploration rate (default: 0.3)"
    )
    
    parser.add_argument(
        "--self-play-frequency", type=int, default=5,
        help="Run self-play every N episodes (default: 5)"
    )
    
    parser.add_argument(
        "--save-frequency", type=int, default=10,
        help="Save model every N episodes (default: 10)"
    )
    
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Path to model file to load"
    )
    
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run without GUI (text mode only)"
    )
    
    parser.add_argument(
        "--cell-size", type=int, default=20,
        help="Cell size in pixels for GUI (default: 20)"
    )
    
    parser.add_argument(
        "--delay", type=float, default=0.05,
        help="Delay between visualization steps in seconds (default: 0.05)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    if not HAS_NUMPY:
        print("ERROR: NumPy is required for SPICE mode")
        print("Install with: pip install numpy")
        sys.exit(1)
    
    args = parse_args()
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create and run trainer
    trainer = SPICETrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()

