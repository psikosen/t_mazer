"""
Self-Optimization module that adjusts training based on reflection insights.
Inspired by SPICE: Self-Play In Corpus Environments Improves Reasoning
"""
import numpy as np
from collections import deque


class SelfOptimizer:
    """
    Automatically optimizes training parameters and strategies based on performance analysis.
    """
    
    def __init__(self, initial_learning_rate=0.01, initial_exploration_rate=0.3):
        """
        Initialize self-optimizer.
        
        Args:
            initial_learning_rate (float): Starting learning rate
            initial_exploration_rate (float): Starting exploration rate
        """
        self.learning_rate = initial_learning_rate
        self.exploration_rate = initial_exploration_rate
        self.optimization_history = []
        self.performance_window = deque(maxlen=20)  # Track last 20 performances
        
    def optimize(self, reflection_insights, current_performance):
        """
        Optimize training parameters based on reflection insights.
        
        Args:
            reflection_insights (dict): Insights from self-reflection
            current_performance (dict): Current performance metrics
            
        Returns:
            dict: Optimized parameters and strategy adjustments
        """
        # Store performance
        self.performance_window.append(current_performance)
        
        # Analyze trends
        trend = self._analyze_trend()
        
        # Get optimization recommendations
        optimizations = {
            'learning_rate': self._optimize_learning_rate(reflection_insights, trend),
            'exploration_rate': self._optimize_exploration_rate(reflection_insights, trend),
            'reward_weights': self._optimize_reward_weights(reflection_insights),
            'training_focus': self._determine_training_focus(reflection_insights),
            'adjustments': {}
        }
        
        # Apply adjustments
        self._apply_optimizations(optimizations)
        
        # Store history
        self.optimization_history.append({
            'parameters': optimizations.copy(),
            'reflection': reflection_insights,
            'performance': current_performance
        })
        
        return optimizations
    
    def _analyze_trend(self):
        """Analyze performance trends over recent history."""
        if len(self.performance_window) < 3:
            return 'insufficient_data'
        
        # Calculate average performance over time windows
        recent = list(self.performance_window)[-5:]
        older = list(self.performance_window)[:-5] if len(self.performance_window) > 5 else []
        
        if not older:
            return 'improving'  # Optimistic if we don't have enough data
        
        recent_avg = np.mean([p.get('efficiency', 0.5) for p in recent])
        older_avg = np.mean([p.get('efficiency', 0.5) for p in older])
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _optimize_learning_rate(self, reflection_insights, trend):
        """
        Optimize learning rate based on performance.
        
        Args:
            reflection_insights (dict): Reflection insights
            trend (str): Performance trend
            
        Returns:
            float: Optimized learning rate
        """
        base_lr = self.learning_rate
        adjustments = []
        
        # Adjust based on weaknesses
        weaknesses = reflection_insights.get('weaknesses', [])
        for weakness in weaknesses:
            if weakness['type'] == 'low_efficiency':
                adjustments.append(0.005)  # Increase LR
            elif weakness['type'] == 'excessive_backtracking':
                adjustments.append(-0.002)  # Decrease LR for stability
        
        # Adjust based on trend
        if trend == 'improving':
            # Learning is working, can maintain or slightly increase
            adjustments.append(0.001)
        elif trend == 'declining':
            # Learning might be too aggressive, decrease
            adjustments.append(-0.003)
        
        # Average adjustments
        if adjustments:
            adjustment = np.mean(adjustments)
        else:
            adjustment = 0
        
        # Clamp learning rate
        new_lr = base_lr + adjustment
        new_lr = max(0.001, min(0.1, new_lr))
        
        return new_lr
    
    def _optimize_exploration_rate(self, reflection_insights, trend):
        """
        Optimize exploration rate.
        
        Args:
            reflection_insights (dict): Reflection insights
            trend (str): Performance trend
            
        Returns:
            float: Optimized exploration rate
        """
        base_er = self.exploration_rate
        adjustments = []
        
        # Adjust based on weaknesses
        weaknesses = reflection_insights.get('weaknesses', [])
        for weakness in weaknesses:
            if weakness['type'] == 'excessive_backtracking':
                adjustments.append(-0.05)  # Reduce exploration
            elif weakness['type'] == 'suboptimal_paths':
                adjustments.append(0.02)  # Slight increase for better exploration
        
        # Adjust based on efficiency
        efficiency = reflection_insights.get('efficiency', 0.5)
        if efficiency > 0.8:
            # Doing well, reduce exploration slightly
            adjustments.append(-0.02)
        elif efficiency < 0.4:
            # Struggling, might need more exploration
            adjustments.append(0.03)
        
        # Average adjustments
        if adjustments:
            adjustment = np.mean(adjustments)
        else:
            adjustment = 0
        
        # Clamp exploration rate
        new_er = base_er + adjustment
        new_er = max(0.1, min(0.7, new_er))
        
        return new_er
    
    def _optimize_reward_weights(self, reflection_insights):
        """
        Optimize reward function weights.
        
        Args:
            reflection_insights (dict): Reflection insights
            
        Returns:
            dict: Reward weight adjustments
        """
        weights = {
            'success_reward': 1.0,
            'efficiency_bonus': 0.5,
            'optimality_bonus': 0.3,
            'dead_end_penalty': -0.2,
            'step_penalty': -0.01
        }
        
        # Adjust based on weaknesses
        weaknesses = reflection_insights.get('weaknesses', [])
        for weakness in weaknesses:
            if weakness['type'] == 'low_efficiency':
                weights['efficiency_bonus'] = 0.8  # Increase efficiency reward
            elif weakness['type'] == 'suboptimal_paths':
                weights['optimality_bonus'] = 0.5  # Increase optimality reward
            elif weakness['type'] == 'excessive_backtracking':
                weights['dead_end_penalty'] = -0.5  # Heavier penalty
        
        return weights
    
    def _determine_training_focus(self, reflection_insights):
        """
        Determine what aspects of training to focus on.
        
        Args:
            reflection_insights (dict): Reflection insights
            
        Returns:
            list: Training focus areas
        """
        focus = []
        weaknesses = reflection_insights.get('weaknesses', [])
        
        # Map weaknesses to training focuses
        weakness_to_focus = {
            'low_efficiency': 'path_efficiency',
            'suboptimal_paths': 'optimal_search',
            'excessive_backtracking': 'exploration_strategy',
            'zigzagging': 'smooth_navigation'
        }
        
        for weakness in weaknesses:
            focus_type = weakness_to_focus.get(weakness['type'])
            if focus_type and focus_type not in focus:
                focus.append(focus_type)
        
        # Add general focus if none specified
        if not focus:
            focus.append('general_improvement')
        
        return focus
    
    def _apply_optimizations(self, optimizations):
        """Apply optimizations to internal parameters."""
        self.learning_rate = optimizations['learning_rate']
        self.exploration_rate = optimizations['exploration_rate']
    
    def get_current_parameters(self):
        """Get current optimized parameters."""
        return {
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'history_size': len(self.optimization_history)
        }
    
    def reset(self):
        """Reset optimizer to initial state."""
        self.learning_rate = 0.01
        self.exploration_rate = 0.3
        self.optimization_history.clear()
        self.performance_window.clear()

