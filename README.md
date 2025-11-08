# T_Mazer: Ternary Fair Play Maze Solver

## Project Overview
T_Mazer is an interactive maze-solving application that demonstrates Ternary Fair Play (TFP) concepts. It combines:

1. **Procedural Maze Generation**: Creates random, solvable mazes of varying complexity
2. **Interactive GUI**: Visualizes maze-solving in real-time
3. **TFP-inspired AI Solver**: A maze-solving AI based on Ternary Fair Play model principles

## Key Features
- Procedural maze generation with adjustable complexity
- Real-time visualization of AI solving process
- Self-learning AI that improves over time
- Implementation of Fair Play tokens for structured reasoning
- Randomized initial moves to encourage exploration

## Project Structure
- `src/`: Source code for maze generation, solver, and GUI
- `models/`: Trained model files and serialized AI states
- `data/`: Generated maze configurations and training data
- `docs/`: Documentation and development notes
- `tests/`: Unit and integration tests

## TFP Model Inspiration
This project draws inspiration from the Ternary Fair Play (TFP) model architecture, which combines:
- **Multi-Head Latent Attention (MLA)**: Efficient transformer attention mechanism
- **Ternary Quantization**: Ultra-efficient weight representation {-1, 0, 1}
- **Fair Play Tokens**: Structured tokens that make reasoning explicit and interpretable

## Development Phases
1. **Initialization**: Project setup and environment configuration
2. **Maze Generation**: Procedural maze generator implementation
3. **Solver Core**: Basic maze-solving algorithms
4. **GUI Development**: Real-time visualization interface
5. **AI Integration**: Implementing TFP-inspired learning mechanisms
6. **Evaluation & Refinement**: Fine-tuning and optimization

## Getting Started

### Quick Start
The easiest way to run T_Mazer is with the automatic quick start script:

```bash
python3 start.py
```

This script will automatically detect available dependencies and run the appropriate version of the application.

### Prerequisites
- Python 3.7 or higher
- For the full version: pygame and numpy
- For the text version: numpy only
- For the simple version: no external dependencies

### Installation
Install the required dependencies (optional):

```bash
# For full graphical version
pip install numpy pygame

# For text-only version
pip install numpy
```

### Running the Application
Choose the version that works best for your environment:

1. Navigate to the project directory:
   ```bash
   cd /path/to/t_mazer
   ```

2. Run the application (one of the following):
   ```bash
   # All-in-one version (most recommended)
   python3 t_mazer.py
   
   # Using automatic detection
   python3 start.py
   
   # Full graphical version
   python3 run.py
   
   # Text-only version
   python3 text_run.py
   
   # Simple version (no dependencies)
   python3 simple_run.py
   ```

**Unified Script Options**

The `t_mazer.py` script includes all versions in one file and has additional options:

```bash
# Force text mode even if pygame is available
python3 t_mazer.py --force-text

# Force simple mode with no dependencies
python3 t_mazer.py --force-simple
```

## Enhanced Version with Training

For the full experience with neural network training and continuous maze solving:

```bash
# Run enhanced version with training
python3 t_mazer_enhanced.py
```

The enhanced version features:

- **Neural Network with Ternary Weights**: Model is quantized to {-1, 0, 1} values
- **Continuous Learning**: The model improves as it solves more mazes
- **Automatic Restart**: New mazes are generated after each solution
- **Training Statistics**: Track performance improvement over time
- **Saved Models**: Trained models are saved to the `models/` directory
- **Test Mode**: Test what the model has learned without algorithmic fallbacks

Additional options:

```bash
# Customize maze size
python3 t_mazer_enhanced.py --width 30 --height 30

# Adjust visualization speed
python3 t_mazer_enhanced.py --delay 0.05

# Force text mode
python3 t_mazer_enhanced.py --force-text

# Load a specific model
python3 t_mazer_enhanced.py --load-model models/ternary_solver_YYYYMMDD_HHMMSS.pkl

# Run in test mode (neural network only, no algorithmic fallbacks)
python3 t_mazer_enhanced.py --test-model
```

### Neural Network Test Mode

The test mode allows you to evaluate what the neural network has learned without any algorithmic fallbacks:

```bash
# Run dedicated test mode
python3 t_mazer_enhanced.py --test-model
```

While running in normal mode, you can also:
- Press the 'T' key to toggle test mode during execution
- Watch the neural network solve mazes using only what it has learned
- See statistics about prediction accuracy and success rates

Test mode shows the "pure" learning progress of the neural network, highlighting how well it has internalized the maze-solving task rather than relying on the algorithm.

## SPICE Mode: Self-Play with Reflection and Optimization

Inspired by the SPICE (Self-Play In Corpus Environments) paper, T_Mazer now includes advanced self-improvement capabilities:

### Features

- **Self-Reflection**: The model analyzes its own performance, identifying weaknesses and inefficiencies
- **Self-Optimization**: Automatic parameter tuning based on reflection insights
- **Self-Play**: Models compete against themselves and previous versions to improve
- **Advanced Maze Generation**: Adaptive difficulty that adjusts based on solver performance

### Running SPICE Mode

```bash
# Basic SPICE training
python3 t_mazer_spice.py

# Customize training
python3 t_mazer_spice.py --episodes 200 --complexity 0.6

# Adjust self-play frequency
python3 t_mazer_spice.py --self-play-frequency 10

# Load existing model
python3 t_mazer_spice.py --load-model models/spice_ternary_*.pkl
```

### SPICE Command-Line Options

```bash
--width WIDTH              Base maze width (default: 20)
--height HEIGHT            Base maze height (default: 20)
--complexity FLOAT         Initial complexity 0.0-1.0 (default: 0.5)
--episodes N               Number of training episodes (default: 100)
--learning-rate FLOAT      Initial learning rate (default: 0.01)
--exploration-rate FLOAT   Initial exploration rate (default: 0.3)
--self-play-frequency N    Run self-play every N episodes (default: 5)
--save-frequency N         Save model every N episodes (default: 10)
--load-model PATH          Path to model file to load
```

### How SPICE Works

1. **Self-Reflection**: After each maze-solving attempt, the system analyzes:
   - Path efficiency (how close to optimal)
   - Dead-end encounters
   - Direction changes and zigzagging
   - Overall performance metrics

2. **Self-Optimization**: Based on reflection insights, the system automatically adjusts:
   - Learning rate (increase if improving, decrease if declining)
   - Exploration rate (reduce if excessive backtracking)
   - Reward weights (emphasize weaknesses)
   - Training focus areas

3. **Self-Play**: Periodically, the model creates variants of itself and competes:
   - Creates mutated versions with small weight variations
   - Competes on the same maze
   - Learns from better-performing variants

4. **Adaptive Maze Generation**: Maze difficulty automatically adjusts:
   - Increases if solver performs well (>80% efficiency)
   - Decreases if solver struggles (<40% efficiency)
   - Adds challenging features (dead ends, bottlenecks) based on complexity

3. Command-line options:
   ```
   # Generate a larger maze
   python3 run.py --width 30 --height 30
   
   # Adjust visualization speed
   python3 run.py --delay 0.02
   
   # Increase maze complexity
   python3 run.py --complexity 0.9
   
   # Run without GUI (text-only mode)
   python3 run.py --no-gui
   ```

4. Close the application by closing the window or pressing Ctrl+C in the terminal.

### Alternative Versions

#### Text-Only Version
If you don't have pygame installed or prefer a simpler version, you can use the text-only version:

```
python3 text_run.py
```

The text-only version supports the same command-line options:
```
python3 text_run.py --width 30 --height 30 --delay 0.1
```

#### Simple Version (No Dependencies)
For environments with limited Python packages, use the simple version that requires no external dependencies:

```
python3 simple_run.py
```

This version works with only the Python standard library:
```
python3 simple_run.py --width 15 --height 10 --delay 0.2
```

### Troubleshooting
- If you see "ModuleNotFoundError: No module named 'pygame'", install pygame:
  ```
  pip install pygame
  ```
- For "ModuleNotFoundError: No module named 'numpy'", install numpy:
  ```
  pip install numpy
  ```
- If you can't install pygame, use the text-only version with `text_run.py`
