# Installation Guide for T_Mazer

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation Steps

### 1. Clone or download the repository

```bash
# If using git
git clone <repository_url>
cd t_mazer

# Or navigate to the downloaded location
cd path/to/t_mazer
```

### 2. Create a virtual environment (optional but recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Run the application

```bash
# Run with default settings
python run.py

# Run with custom settings
python run.py --width 30 --height 30 --complexity 0.8 --delay 0.05
```

## Command Line Options

The following options can be used when running the application:

- `--width N`: Set the maze width (default: 20)
- `--height N`: Set the maze height (default: 20)
- `--complexity F`: Set the maze complexity factor 0.0-1.0 (default: 0.7)
- `--density F`: Set the maze density factor 0.0-1.0 (default: 0.7)
- `--cell-size N`: Set the cell size in pixels (default: 30)
- `--delay F`: Set the delay between visualization steps in seconds (default: 0.1)
- `--no-gui`: Run without GUI visualization

## Example Commands

Generate a large, complex maze:
```bash
python run.py --width 40 --height 40 --complexity 0.9 --density 0.8
```

Generate a simple maze with fast visualization:
```bash
python run.py --width 15 --height 15 --complexity 0.5 --delay 0.05
```

Run without GUI (command-line only):
```bash
python run.py --no-gui
```

## Troubleshooting

If you encounter any issues with PyGame or other dependencies, ensure you have the necessary system libraries installed:

### On Ubuntu/Debian:
```bash
sudo apt-get install python3-dev python3-numpy python3-pygame
```

### On macOS:
```bash
brew install sdl sdl_image sdl_mixer sdl_ttf portmidi
```

### On Windows:
The required libraries should be installed automatically with the PyGame package.
