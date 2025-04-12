#!/usr/bin/env python3
"""
Simple wrapper script to run the T_Mazer application.
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main function
from src.main import main

if __name__ == "__main__":
    main()
