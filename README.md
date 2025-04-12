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
```

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
