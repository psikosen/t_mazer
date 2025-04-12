"""
Main module for the T_Mazer application.
"""
import argparse
import sys
import time
import pygame

from maze_generator import MazeGenerator
from maze_solver import MazeSolver
from maze_gui import MazeGUI


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="T_Mazer - Ternary Fair Play Maze Solver")
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=20,
        help="Width of the maze (default: 20)"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=20,
        help="Height of the maze (default: 20)"
    )
    
    parser.add_argument(
        "--complexity", 
        type=float, 
        default=0.7,
        help="Maze complexity factor 0.0-1.0 (default: 0.7)"
    )
    
    parser.add_argument(
        "--density", 
        type=float, 
        default=0.7,
        help="Maze density factor 0.0-1.0 (default: 0.7)"
    )
    
    parser.add_argument(
        "--cell-size", 
        type=int, 
        default=30,
        help="Cell size in pixels (default: 30)"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.1,
        help="Delay between visualization steps in seconds (default: 0.1)"
    )
    
    parser.add_argument(
        "--no-gui", 
        action="store_true",
        help="Run without GUI visualization"
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_args()
    
    # Initialize maze generator
    generator = MazeGenerator(
        width=args.width,
        height=args.height,
        complexity=args.complexity,
        density=args.density
    )
    
    # Generate a maze
    print("Generating maze...")
    maze = generator.generate()
    print(f"Maze generated with dimensions {args.width}x{args.height}")
    
    # Initialize solver
    solver = MazeSolver()
    
    if args.no_gui:
        # Solve without visualization
        print("Solving maze...")
        start_time = time.time()
        solution_path, reasoning_tokens = solver.solve(maze)
        end_time = time.time()
        
        print(f"Maze solved in {end_time - start_time:.2f} seconds")
        print(f"Solution path length: {len(solution_path)}")
        print(f"Number of reasoning tokens: {len(reasoning_tokens)}")
        
        # Print a sample of the reasoning process
        print("\nSample reasoning process:")
        for token in reasoning_tokens[:10]:
            print(f"  {token}")
        if len(reasoning_tokens) > 10:
            print(f"  ... and {len(reasoning_tokens) - 10} more tokens")
    else:
        # Initialize GUI
        gui = MazeGUI(cell_size=args.cell_size)
        gui.initialize(maze)
        
        # Visualize solving process
        print("Visualizing maze solving...")
        solution_path, reasoning_tokens = gui.visualize_solving(
            maze, solver, delay=args.delay
        )
        
        if solution_path is None:
            print("Visualization was terminated by the user")
            return
        
        print(f"Maze solved! Solution path length: {len(solution_path)}")
        
        # Keep the window open until the user closes it
        print("Close the window to exit")
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                time.sleep(0.1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            gui.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
