"""
Self-Play module for training through competitive play.
Inspired by SPICE: Self-Play In Corpus Environments Improves Reasoning
"""
import numpy as np
import random
import copy
from datetime import datetime


class SelfPlayArena:
    """Manages self-play between different model versions."""
    
    def __init__(self, model_class, reflection_system=None):
        """
        Initialize the self-play arena.
        
        Args:
            model_class: Class for creating model instances
            reflection_system: SelfReflection instance for analysis
        """
        self.model_class = model_class
        self.reflection = reflection_system
        self.competitors = []  # List of (model, performance_score)
        self.match_history = []
        
    def create_competitor(self, base_model=None):
        """
        Create a new competitor model (either clone or variation).
        
        Args:
            base_model: Model to clone/vary from, or None for new model
            
        Returns:
            Model instance
        """
        if base_model is None:
            # Create new model
            competitor = self.model_class()
        else:
            # Create variation of base model
            competitor = copy.deepcopy(base_model)
            # Add small random variations to weights
            self._mutate_model(competitor)
        
        return competitor
    
    def _mutate_model(self, model, mutation_rate=0.05):
        """Add small mutations to model weights."""
        if hasattr(model, 'weights') and model.weights:
            for key in ['w1', 'w2']:
                if key in model.weights:
                    mask = np.random.random(model.weights[key].shape) < mutation_rate
                    mutation = np.random.randn(*model.weights[key].shape) * 0.1
                    model.weights[key][mask] += mutation[mask]
                    # Re-ternarize if needed
                    if hasattr(model, '_ternarize_weights'):
                        model._ternarize_weights()
    
    def play_match(self, maze, competitor1, competitor2):
        """
        Play a match between two competitors on the same maze.
        
        Args:
            maze (np.ndarray): The maze to solve
            competitor1: First competitor model
            competitor2: Second competitor model
            
        Returns:
            dict: Match results
        """
        import time
        
        # Both competitors solve the same maze
        start_time1 = time.time()
        solution1 = self._solve_with_model(competitor1, maze)
        time1 = time.time() - start_time1
        
        start_time2 = time.time()
        solution2 = self._solve_with_model(competitor2, maze)
        time2 = time.time() - start_time2
        
        # Evaluate solutions
        result1 = self._evaluate_solution(maze, solution1)
        result2 = self._evaluate_solution(maze, solution2)
        
        # Determine winner
        if result1['success'] and not result2['success']:
            winner = 1
        elif result2['success'] and not result1['success']:
            winner = 2
        elif result1['success'] and result2['success']:
            # Both succeeded - compare efficiency
            if result1['score'] > result2['score']:
                winner = 1
            elif result2['score'] > result1['score']:
                winner = 2
            else:
                winner = 0  # Tie
        else:
            # Both failed - compare attempts
            if result1['score'] > result2['score']:
                winner = 1
            elif result2['score'] > result1['score']:
                winner = 2
            else:
                winner = 0
        
        match_result = {
            'timestamp': datetime.now().isoformat(),
            'competitor1_result': result1,
            'competitor2_result': result2,
            'winner': winner,
            'maze_size': maze.shape
        }
        
        self.match_history.append(match_result)
        return match_result
    
    def _solve_with_model(self, model, maze):
        """Use a model to solve a maze."""
        # This assumes the model has a solve method
        if hasattr(model, 'solve'):
            solution, _ = model.solve(maze, training_mode=False)
            return solution
        elif hasattr(model, 'predict'):
            # Use model's prediction to guide solving
            return self._guided_solve(model, maze)
        else:
            # Fallback to random walk
            return self._random_walk(maze)
    
    def _guided_solve(self, model, maze):
        """Use model predictions to guide solving."""
        start = (1, 1)
        end = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        path = [start]
        visited = {start}
        current = start
        max_steps = maze.size * 2  # Prevent infinite loops
        
        for _ in range(max_steps):
            if current == end:
                break
            
            # Get model's preferred direction
            try:
                direction = model.predict(maze, current, end)
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                dy, dx = directions[direction]
            except:
                # Fallback to random
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                dy, dx = random.choice(directions)
            
            next_pos = (current[0] + dy, current[1] + dx)
            
            # Check if valid
            if (0 <= next_pos[0] < maze.shape[0] and 
                0 <= next_pos[1] < maze.shape[1] and 
                maze[next_pos[0], next_pos[1]] == 0):
                
                if next_pos not in visited:
                    path.append(next_pos)
                    visited.add(next_pos)
                    current = next_pos
                else:
                    # Already visited, try another direction
                    valid_moves = self._get_valid_moves(maze, current, visited)
                    if valid_moves:
                        current = random.choice(valid_moves)
                        if current not in visited:
                            path.append(current)
                            visited.add(current)
                    else:
                        break
            else:
                # Invalid move, try another direction
                valid_moves = self._get_valid_moves(maze, current, visited)
                if valid_moves:
                    current = random.choice(valid_moves)
                    if current not in visited:
                        path.append(current)
                        visited.add(current)
                else:
                    break
        
        return path
    
    def _random_walk(self, maze):
        """Random walk fallback."""
        start = (1, 1)
        end = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        path = [start]
        visited = {start}
        current = start
        max_steps = maze.size
        
        for _ in range(max_steps):
            if current == end:
                break
            
            valid_moves = self._get_valid_moves(maze, current, visited)
            if not valid_moves:
                break
            
            current = random.choice(valid_moves)
            path.append(current)
            visited.add(current)
        
        return path
    
    def _get_valid_moves(self, maze, pos, visited):
        """Get valid unvisited moves from position."""
        y, x = pos
        moves = []
        
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < maze.shape[0] and 
                0 <= nx < maze.shape[1] and 
                maze[ny, nx] == 0 and
                (ny, nx) not in visited):
                moves.append((ny, nx))
        
        return moves
    
    def _evaluate_solution(self, maze, solution_path):
        """Evaluate a solution path."""
        if not solution_path:
            return {
                'success': False,
                'score': 0.0,
                'path_length': 0,
                'reached_end': False
            }
        
        start = (1, 1)
        end = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        reached_end = solution_path[-1] == end
        path_length = len(solution_path)
        
        # Calculate score
        if reached_end:
            # Score based on efficiency
            optimal_distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
            efficiency = optimal_distance / path_length if path_length > 0 else 0
            score = 100.0 * efficiency
        else:
            # Partial credit for how close we got
            last_pos = solution_path[-1]
            distance_to_end = abs(end[0] - last_pos[0]) + abs(end[1] - last_pos[1])
            max_distance = maze.shape[0] + maze.shape[1]
            score = 50.0 * (1 - distance_to_end / max_distance)
        
        return {
            'success': reached_end,
            'score': score,
            'path_length': path_length,
            'reached_end': reached_end
        }
    
    def tournament(self, mazes, num_rounds=10):
        """
        Run a tournament between current competitors.
        
        Args:
            mazes (list): List of mazes to use for matches
            num_rounds (int): Number of rounds to play
            
        Returns:
            dict: Tournament results
        """
        if len(self.competitors) < 2:
            return {'error': 'Need at least 2 competitors for tournament'}
        
        scores = {i: 0 for i in range(len(self.competitors))}
        wins = {i: 0 for i in range(len(self.competitors))}
        
        for round_num in range(num_rounds):
            maze = random.choice(mazes)
            
            # Match each competitor against others
            for i in range(len(self.competitors)):
                for j in range(i + 1, len(self.competitors)):
                    result = self.play_match(
                        maze,
                        self.competitors[i][0],
                        self.competitors[j][0]
                    )
                    
                    if result['winner'] == 1:
                        scores[i] += result['competitor1_result']['score']
                        wins[i] += 1
                    elif result['winner'] == 2:
                        scores[j] += result['competitor2_result']['score']
                        wins[j] += 1
                    else:
                        # Tie - split points
                        scores[i] += result['competitor1_result']['score'] / 2
                        scores[j] += result['competitor2_result']['score'] / 2
        
        # Update competitor scores
        for i, (model, _) in enumerate(self.competitors):
            self.competitors[i] = (model, scores[i])
        
        # Sort by score
        self.competitors.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'rounds': num_rounds,
            'scores': scores,
            'wins': wins,
            'champion': 0,  # Top competitor after sorting
            'ranking': list(range(len(self.competitors)))
        }
    
    def evolve_population(self, keep_top_n=3, add_mutations=2):
        """
        Evolve the population by keeping top performers and adding mutations.
        
        Args:
            keep_top_n (int): Number of top performers to keep
            add_mutations (int): Number of new mutations to add
            
        Returns:
            list: Updated competitors
        """
        if len(self.competitors) == 0:
            return []
        
        # Sort by performance
        self.competitors.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top performers
        new_competitors = self.competitors[:keep_top_n].copy()
        
        # Add mutations of top performers
        for _ in range(add_mutations):
            base = random.choice(new_competitors[:keep_top_n])[0]
            mutant = self.create_competitor(base_model=base)
            new_competitors.append((mutant, 0.0))  # Start with zero score
        
        self.competitors = new_competitors
        return self.competitors

