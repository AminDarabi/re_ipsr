# This file is developed by Amin Darabi as part of the project of
# the course "Geometric Modeling and shape analysis" (IFT6113) at the
# Université de Montréal (UdeM) in Fall 2024.
"""
"""

from .utility import initial_normal, calculate_triangle_centroids
from .ipsr import iterative_poisson_reconstruction

__version__ = "0.1.0"
__author__ = "Amin Darabi"

__all__ = [
    "iterative_poisson_reconstruction",
    "initial_normal",
    "calculate_triangle_centroids",
]