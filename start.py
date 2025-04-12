#!/usr/bin/env python3
"""
Quick start script for T_Mazer that detects available dependencies
and runs the appropriate version of the program.
"""
import sys
import subprocess
import os

def check_module(module_name):
    """Check if a Python module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_args():
    """Get command line arguments to pass to the selected runner."""
    # Skip the script name
    args = sys.argv[1:]
    return args

def main():
    """Determine which version to run based on available dependencies."""
    print("T_Mazer Quick Start")
    print("Checking available dependencies...")
    
    # Check for pygame (full version)
    has_pygame = check_module("pygame")
    # Check for numpy (text version)
    has_numpy = check_module("numpy")
    
    # Determine which script to run
    if has_pygame:
        print("✓ Pygame is available - using full graphical version")
        script = "run.py"
    elif has_numpy:
        print("✓ NumPy is available - using text-based version")
        print("✗ Pygame is not available - install with 'pip install pygame' for graphical version")
        script = "text_run.py"
    else:
        print("✗ Pygame is not available - install with 'pip install pygame' for graphical version")
        print("✗ NumPy is not available - install with 'pip install numpy' for improved version")
        print("Using simple version that requires no dependencies")
        script = "simple_run.py"
    
    # Get command line arguments
    args = get_args()
    
    # Build the command
    cmd = [sys.executable, script] + args
    
    print(f"\nStarting T_Mazer with: {' '.join(cmd)}\n")
    
    # Run the selected script
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExiting T_Mazer...")
        sys.exit(0)

if __name__ == "__main__":
    main()
