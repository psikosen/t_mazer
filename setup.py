"""
Setup script for the T_Mazer package.
"""
from setuptools import setup, find_packages

setup(
    name="t_mazer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pygame>=2.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "t_mazer=src.main:main",
        ],
    },
    author="Raymond Gonzalez",
    author_email="example@example.com",
    description="Ternary Fair Play Maze Solver",
    keywords="maze, ai, solver, ternary, fair play",
    python_requires=">=3.7",
)
