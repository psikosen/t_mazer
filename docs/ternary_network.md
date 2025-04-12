# Ternary Neural Network Architecture

The T_Mazer project implements a simple neural network with ternary weights for maze solving. This document explains the architecture and training approach.

## Ternary Weights Concept

In standard neural networks, weights are typically stored as floating-point values (e.g., 32-bit or 16-bit). Ternary quantization reduces each weight to just three possible values:

- `-1`: Negative weight
- `0`: No connection
- `+1`: Positive weight

This approach reduces memory requirements significantly and simplifies computations, making the model more efficient. Ternary quantization is inspired by the "TerEffic" approach mentioned in the project's conceptual basis.

## Network Architecture

```
                    Ternary Weights
                          │
                          ▼
 ┌─────────┐      ┌─────────────┐      ┌─────────┐
 │ Input   │      │ Hidden      │      │ Output  │
 │ Layer   │ ──── │ Layer       │ ──── │ Layer   │
 │ (7-10   │ {-1, │ (16 neurons)│ {-1, │ (4      │
 │ features)│  0,  │ ReLU        │  0,  │ actions)│
 └─────────┘  +1} └─────────────┘  +1} └─────────┘
```

### Layer Details:

1. **Input Layer**: 
   - Features that represent the maze state (7-10 neurons)
   - Wall detection in each direction (4 features)
   - Direction to target (2 features)
   - Distance metrics and bias term

2. **Hidden Layer**:
   - 16 neurons with ReLU activation
   - Ternary weights connecting to input layer

3. **Output Layer**:
   - 4 neurons (one for each direction: right, down, left, up)
   - Ternary weights connecting to hidden layer
   - Raw output scores (highest score determines chosen direction)

## Training Method

The model uses a reinforcement learning approach:

1. **Action Selection**:
   - Forward pass through the network to predict best move
   - Exploration vs. exploitation balance (occasionally random moves)

2. **Reward System**:
   - Positive reward (+1.0) for reaching the goal
   - Medium reward (+0.5) for moving closer to the goal
   - Negative reward (-0.2) for moving away from the goal
   - Penalty (-1.0) for invalid moves (hitting walls)

3. **Weight Update**:
   - Simplified gradient descent
   - Periodic ternarization (every 10 training steps)
   - Model saved after each successful maze solution

## Ternarization Process

The model converts standard float weights to ternary values using a thresholding approach:

```python
def ternarize_weights(weights):
    # Calculate threshold based on weight magnitudes
    threshold = 0.3 * np.mean(np.abs(weights))
    
    # Ternarize weights
    ternary_weights = np.zeros_like(weights)
    ternary_weights[weights > threshold] = 1    # Strong positive -> +1
    ternary_weights[weights < -threshold] = -1  # Strong negative -> -1
    # Values between -threshold and threshold -> 0
    
    return ternary_weights
```

## Benefits

1. **Efficiency**: Reduced memory footprint and simplified computation
2. **Generalization**: Potentially better generalization due to weight constraints
3. **Transfer Learning**: Can be applied to various maze types after training
4. **Continuous Improvement**: Performance improves as more mazes are solved

The model is saved periodically to the `models/` directory during training, allowing users to reload trained models for future use.
