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
        
        # Track episode count for time-based decay
        self.episode_count = 0
        self.success_streak = 0  # Track consecutive successes
        
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
        self.episode_count += 1
        
        # Track success streak for exploration decay
        if current_performance.get('success', False):
            self.success_streak += 1
        else:
            self.success_streak = 0
        
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
        Optimize learning rate with adaptive scheduling.
        
        Key insight: Start with moderate LR, increase when struggling,
        decrease when stable. Don't let it get too high or learning destabilizes.
        
        Args:
            reflection_insights (dict): Reflection insights
            trend (str): Performance trend
            
        Returns:
            float: Optimized learning rate
        """
        base_lr = self.learning_rate
        adjustments = []
        
        # Time-based decay component (gradual decrease over time)
        if self.episode_count > 30:
            time_decay = -0.0005  # Slight decay after initial learning
            adjustments.append(time_decay)
        
        # Adjust based on weaknesses
        weaknesses = reflection_insights.get('weaknesses', [])
        for weakness in weaknesses:
            if weakness['type'] == 'low_efficiency':
                # Only increase if LR is relatively low
                if base_lr < 0.03:
                    adjustments.append(0.003)
                else:
                    adjustments.append(0.001)  # Smaller increase if already moderate
            elif weakness['type'] == 'excessive_backtracking':
                adjustments.append(-0.002)  # Decrease LR for stability
        
        # Adjust based on trend
        if trend == 'improving':
            # Learning is working, slight decrease to stabilize
            adjustments.append(-0.0005)
        elif trend == 'declining':
            # Learning might be too aggressive or too slow
            if base_lr > 0.03:
                adjustments.append(-0.002)  # Too aggressive, slow down
            elif base_lr < 0.015:
                adjustments.append(0.002)  # Too slow, speed up
        elif trend == 'stable':
            # Stable performance, slight decrease for fine-tuning
            adjustments.append(-0.0002)
        
        # Success streak adjustment
        if self.success_streak >= 3:
            # Doing well, can decrease LR for fine-tuning
            adjustments.append(-0.001)
        
        # Efficiency-based adjustment
        efficiency = reflection_insights.get('efficiency', 0.5)
        if efficiency > 0.8:
            adjustments.append(-0.001)  # Very good, fine-tune
        elif efficiency < 0.1 and base_lr < 0.02:
            adjustments.append(0.002)  # Very bad, need to learn faster
        
        # Average adjustments
        if adjustments:
            adjustment = np.mean(adjustments)
        else:
            adjustment = 0
        
        # Apply adjustment with BETTER bounds
        new_lr = base_lr + adjustment
        
        # CRITICAL: Lower max to 0.05 (was 0.1!)
        # LR above 0.05 can cause unstable learning with ternary weights
        new_lr = max(0.002, min(0.05, new_lr))
        
        return new_lr
    
    def _optimize_exploration_rate(self, reflection_insights, trend):
        """
        Optimize exploration rate with time-based decay and success-based adjustment.
        
        Key insight: High exploration early, but MUST decay over time.
        The model needs to exploit learned knowledge as training progresses.
        
        Args:
            reflection_insights (dict): Reflection insights
            trend (str): Performance trend
            
        Returns:
            float: Optimized exploration rate
        """
        base_er = self.exploration_rate
        adjustments = []
        
        # TIME-BASED DECAY: Exploration should decrease over episodes
        # Formula: decay_factor = 0.995^episode_count (exponential decay)
        time_decay = 0.995 ** self.episode_count
        target_from_decay = 0.5 * time_decay  # Start at 0.5, decay toward 0
        
        # If current rate is higher than time-decayed target, pull it down
        if base_er > target_from_decay + 0.1:
            adjustments.append(-0.02)  # Gradually decrease
        
        # SUCCESS STREAK BONUS: If doing well, reduce exploration faster
        if self.success_streak >= 3:
            adjustments.append(-0.03)  # Doing well, exploit more
        elif self.success_streak >= 5:
            adjustments.append(-0.05)  # Very strong, heavily reduce exploration
        
        # Adjust based on weaknesses (but smaller increases now)
        weaknesses = reflection_insights.get('weaknesses', [])
        for weakness in weaknesses:
            if weakness['type'] == 'excessive_backtracking':
                adjustments.append(-0.03)  # Reduce exploration
            elif weakness['type'] == 'suboptimal_paths':
                # Only slightly increase if we're not already high
                if base_er < 0.25:
                    adjustments.append(0.01)
        
        # Adjust based on efficiency
        efficiency = reflection_insights.get('efficiency', 0.5)
        if efficiency > 0.7:
            # Doing well, reduce exploration
            adjustments.append(-0.02)
        elif efficiency > 0.5:
            adjustments.append(-0.01)
        elif efficiency < 0.2 and base_er < 0.25:
            # Struggling AND exploration is low, might need slight bump
            adjustments.append(0.01)
        
        # Trend-based adjustment
        if trend == 'improving':
            adjustments.append(-0.01)  # Keep exploiting what works
        elif trend == 'declining' and base_er < 0.2:
            adjustments.append(0.005)  # Tiny exploration bump only if very low
        
        # Average adjustments
        if adjustments:
            adjustment = np.mean(adjustments)
        else:
            adjustment = 0
        
        # Apply adjustment with LOWER bounds
        new_er = base_er + adjustment
        
        # CRITICAL: Lower max to 0.35 (was 0.7!)
        # Exploration above 0.35 means too much randomness
        new_er = max(0.05, min(0.35, new_er))
        
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

