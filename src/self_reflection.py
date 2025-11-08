"""
Self-Reflection module for analyzing solver performance and identifying weaknesses.
Inspired by SPICE: Self-Play In Corpus Environments Improves Reasoning
"""
import numpy as np
from collections import defaultdict
from datetime import datetime


class SelfReflection:
    """Analyzes solver performance to identify patterns and weaknesses."""
    
    def __init__(self):
        """Initialize the reflection system."""
        self.performance_history = []
        self.failure_patterns = defaultdict(int)
        self.success_patterns = defaultdict(int)
        self.reflection_insights = []
        
    def reflect(self, maze, solution_path, solving_time, steps_taken, max_steps=None):
        """
        Perform self-reflection on a solving attempt.
        
        Args:
            maze (np.ndarray): The maze that was solved
            solution_path (list): Path taken to solve
            solving_time (float): Time taken to solve
            steps_taken (int): Number of steps in solution
            max_steps (int): Maximum steps allowed (calculated if None)
            
        Returns:
            dict: Reflection insights including weaknesses and recommendations
        """
        # Calculate max_steps if not provided (varies based on maze size)
        if max_steps is None:
            import random
            base_max = maze.shape[0] * maze.shape[1] * 2
            variation = random.uniform(0.8, 1.5)  # Vary each reflection
            max_steps = int(base_max * variation)
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'maze_size': maze.shape,
            'solution_length': len(solution_path) if solution_path else 0,
            'solving_time': solving_time,
            'efficiency': self._calculate_efficiency(maze, solution_path),
            'dead_ends': self._count_dead_ends(solution_path),
            'path_optimality': self._assess_path_optimality(maze, solution_path),
            'weaknesses': [],
            'strengths': [],
            'recommendations': []
        }
        
        # Analyze patterns
        self._analyze_path_patterns(maze, solution_path, insights)
        self._identify_weaknesses(insights)
        self._generate_recommendations(insights)
        
        # Store for historical analysis
        self.performance_history.append(insights)
        self.reflection_insights.append(insights)
        
        return insights
    
    def _calculate_efficiency(self, maze, solution_path):
        """Calculate how efficient the solution path is."""
        if not solution_path:
            return 0.0
        
        # Manhattan distance from start to end
        start = solution_path[0]
        end = solution_path[-1]
        optimal_distance = abs(end[0] - start[0]) + abs(end[1] - start[1])
        
        if optimal_distance == 0:
            return 1.0
        
        # Efficiency = optimal / actual
        actual_distance = len(solution_path) - 1
        efficiency = optimal_distance / actual_distance if actual_distance > 0 else 0.0
        
        return min(efficiency, 1.0)  # Cap at 1.0
    
    def _count_dead_ends(self, solution_path):
        """Count how many times the path backtracked (indicating dead ends)."""
        if not solution_path or len(solution_path) < 2:
            return 0
        
        dead_ends = 0
        visited = set()
        
        for pos in solution_path:
            if pos in visited:
                dead_ends += 1
            visited.add(pos)
        
        return dead_ends
    
    def _assess_path_optimality(self, maze, solution_path):
        """Assess if the path is close to optimal."""
        if not solution_path:
            return 0.0
        
        start = solution_path[0]
        end = solution_path[-1]
        
        # Calculate shortest possible path using BFS
        shortest_path = self._bfs_shortest_path(maze, start, end)
        
        if shortest_path is None:
            return 0.0
        
        optimal_length = len(shortest_path) - 1
        actual_length = len(solution_path) - 1
        
        if optimal_length == 0:
            return 1.0
        
        return optimal_length / actual_length if actual_length > 0 else 0.0
    
    def _bfs_shortest_path(self, maze, start, end):
        """Find shortest path using BFS for comparison."""
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            (y, x), path = queue.pop(0)
            
            if (y, x) == end:
                return path
            
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < maze.shape[0] and 
                    0 <= nx < maze.shape[1] and 
                    maze[ny, nx] == 0 and
                    (ny, nx) not in visited):
                    queue.append(((ny, nx), path + [(ny, nx)]))
                    visited.add((ny, nx))
        
        return None
    
    def _analyze_path_patterns(self, maze, solution_path, insights):
        """Analyze patterns in the solution path."""
        if not solution_path or len(solution_path) < 2:
            return
        
        # Analyze direction changes
        direction_changes = 0
        last_direction = None
        
        for i in range(1, len(solution_path)):
            prev = solution_path[i-1]
            curr = solution_path[i]
            
            dy = curr[0] - prev[0]
            dx = curr[1] - prev[1]
            
            current_direction = (dy, dx)
            if last_direction and current_direction != last_direction:
                direction_changes += 1
            last_direction = current_direction
        
        insights['direction_changes'] = direction_changes
        insights['zigzag_factor'] = direction_changes / len(solution_path) if solution_path else 0
        
        # Record patterns
        pattern_key = f"size_{maze.shape[0]}x{maze.shape[1]}_steps_{len(solution_path)}"
        if insights['path_optimality'] > 0.8:
            self.success_patterns[pattern_key] += 1
        else:
            self.failure_patterns[pattern_key] += 1
    
    def _identify_weaknesses(self, insights):
        """Identify specific weaknesses from the analysis."""
        weaknesses = []
        
        if insights['efficiency'] < 0.5:
            weaknesses.append({
                'type': 'low_efficiency',
                'severity': 'high',
                'description': f"Path efficiency is only {insights['efficiency']:.2%}"
            })
        
        if insights['path_optimality'] < 0.7:
            weaknesses.append({
                'type': 'suboptimal_paths',
                'severity': 'medium',
                'description': f"Path is only {insights['path_optimality']:.2%} of optimal length"
            })
        
        if insights.get('dead_ends', 0) > 5:
            weaknesses.append({
                'type': 'excessive_backtracking',
                'severity': 'high',
                'description': f"Encountered {insights['dead_ends']} dead ends"
            })
        
        if insights.get('zigzag_factor', 0) > 0.5:
            weaknesses.append({
                'type': 'zigzagging',
                'severity': 'medium',
                'description': "Path shows excessive zigzagging pattern"
            })
        
        insights['weaknesses'] = weaknesses
    
    def _generate_recommendations(self, insights):
        """Generate recommendations based on identified weaknesses."""
        recommendations = []
        
        for weakness in insights['weaknesses']:
            if weakness['type'] == 'low_efficiency':
                recommendations.append({
                    'action': 'improve_heuristic',
                    'priority': 'high',
                    'description': 'Improve distance estimation heuristic'
                })
            elif weakness['type'] == 'suboptimal_paths':
                recommendations.append({
                    'action': 'enhance_search',
                    'priority': 'medium',
                    'description': 'Enhance search algorithm to find better paths'
                })
            elif weakness['type'] == 'excessive_backtracking':
                recommendations.append({
                    'action': 'better_exploration',
                    'priority': 'high',
                    'description': 'Improve exploration strategy to avoid dead ends'
                })
            elif weakness['type'] == 'zigzagging':
                recommendations.append({
                    'action': 'smoother_paths',
                    'priority': 'medium',
                    'description': 'Reduce direction changes for smoother paths'
                })
        
        insights['recommendations'] = recommendations
    
    def get_aggregate_insights(self, window_size=10):
        """
        Get aggregated insights from recent performance.
        
        Args:
            window_size (int): Number of recent attempts to analyze
            
        Returns:
            dict: Aggregate insights
        """
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        recent = self.performance_history[-window_size:]
        
        avg_efficiency = np.mean([h['efficiency'] for h in recent])
        avg_optimality = np.mean([h['path_optimality'] for h in recent])
        total_dead_ends = sum([h.get('dead_ends', 0) for h in recent])
        
        common_weaknesses = defaultdict(int)
        for h in recent:
            for w in h.get('weaknesses', []):
                common_weaknesses[w['type']] += 1
        
        return {
            'window_size': len(recent),
            'average_efficiency': avg_efficiency,
            'average_optimality': avg_optimality,
            'total_dead_ends': total_dead_ends,
            'common_weaknesses': dict(common_weaknesses),
            'overall_performance': 'good' if avg_efficiency > 0.7 else 'needs_improvement'
        }
    
    def get_training_focus(self):
        """
        Determine what aspects to focus on in training.
        
        Returns:
            dict: Training focus recommendations
        """
        aggregate = self.get_aggregate_insights()
        
        focus_areas = []
        priorities = []
        
        if aggregate.get('average_efficiency', 0) < 0.6:
            focus_areas.append('path_efficiency')
            priorities.append('high')
        
        if aggregate.get('average_optimality', 0) < 0.7:
            focus_areas.append('optimal_pathfinding')
            priorities.append('high')
        
        dead_ends = aggregate.get('total_dead_ends', 0)
        if dead_ends > 10:
            focus_areas.append('dead_end_avoidance')
            priorities.append('high')
        
        if not focus_areas:
            focus_areas.append('general_improvement')
            priorities.append('low')
        
        return {
            'focus_areas': focus_areas,
            'priorities': priorities,
            'recommended_adjustments': self._get_training_adjustments(focus_areas)
        }
    
    def _get_training_adjustments(self, focus_areas):
        """Get specific training adjustments based on focus areas."""
        adjustments = {}
        
        if 'path_efficiency' in focus_areas:
            adjustments['learning_rate'] = 0.015  # Increase learning rate
            adjustments['reward_bonus_efficiency'] = 1.5  # Reward efficient paths more
        
        if 'optimal_pathfinding' in focus_areas:
            adjustments['exploration_rate'] = 0.25  # Reduce exploration slightly
            adjustments['reward_bonus_optimal'] = 1.3
        
        if 'dead_end_avoidance' in focus_areas:
            adjustments['dead_end_penalty'] = -0.5  # Penalize dead ends
            adjustments['exploration_rate'] = 0.2  # Reduce random exploration
        
        return adjustments

