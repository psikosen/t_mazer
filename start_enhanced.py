#!/usr/bin/env python3
"""
Quick start script for T_Mazer Enhanced that detects available dependencies
and runs the enhanced version with the best options.
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
    """Get command line arguments to pass to the runner."""
    # Skip the script name
    args = sys.argv[1:]
    return args

def main():
    """Determine which version to run based on available dependencies."""
    print("T_Mazer Enhanced Quick Start")
    print("Checking available dependencies...")
    
    # Check for pygame (full version)
    has_pygame = check_module("pygame")
    # Check for numpy (text version)
    has_numpy = check_module("numpy")
    
    # Determine which settings to use
    if has_pygame and has_numpy:
        print("✓ Pygame and NumPy are available - using full graphical version with training")
        extra_args = []
    elif has_numpy:
        print("✓ NumPy is available - using text-based version with training")
        print("✗ Pygame is not available - install with 'pip install pygame' for graphical version")
        extra_args = ["--force-text"]
    else:
        print("✗ Pygame is not available - install with 'pip install pygame' for graphical version")
        print("✗ NumPy is not available - install with 'pip install numpy' for improved version")
        print("Using simple version with basic training")
        extra_args = ["--force-simple"]
    
    # Get command line arguments
    args = get_args()
    
    # Find latest model if it exists
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_args = []
    
    if os.path.exists(model_dir):
        # Look for models based on availability
        if has_numpy:
            model_files = [f for f in os.listdir(model_dir) if f.startswith("numpy_ternary_")]
        else:
            model_files = [f for f in os.listdir(model_dir) if f.startswith("ternary_solver_")]
        
        if model_files:
            latest_model = os.path.join(model_dir, sorted(model_files)[-1])
            print(f"✓ Found existing model: {os.path.basename(latest_model)}")
            model_args = ["--load-model", latest_model]
    
    # Build the command
    cmd = [sys.executable, "t_mazer_enhanced.py"] + extra_args + model_args + args
    
    print(f"\nStarting T_Mazer Enhanced with: {' '.join(cmd)}\n")
    
    # Run the enhanced version
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExiting T_Mazer Enhanced...")
        sys.exit(0)

if __name__ == "__main__":
    main()
