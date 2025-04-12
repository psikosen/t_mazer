#!/usr/bin/env python3
"""
Test Model script - Run the T_Mazer in test mode to evaluate neural network without
algorithmic fallbacks.
"""
import sys
import subprocess
import os

def main():
    """Run the model in test mode."""
    print("T_Mazer Neural Network Test Mode")
    print("Testing what the model has learned without algorithmic fallbacks")
    
    # Get command line arguments (skip script name)
    args = sys.argv[1:]
    
    # Add the --test-model flag
    cmd = [sys.executable, "t_mazer_enhanced.py", "--test-model"] + args
    
    print(f"\nStarting T_Mazer test mode with: {' '.join(cmd)}\n")
    
    # Run the enhanced version in test mode
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nExiting T_Mazer test mode...")
        sys.exit(0)

if __name__ == "__main__":
    main()
