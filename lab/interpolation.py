from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

class Interpolation(ABC):

    @abstractmethod
    def computeInterpolation(self):
        passs

class Monomial(Interpolation):

    def __init__(self):
        pass
    
    def _generateBasis(self, x, i):
        return x ** i
    
    def computeInterpolation(self, func, x, X):
        """
            Compute the interpolation of the function
            func: function to interpolate
            x: points where we want to compute the interpolation
            X: interpolation points
        """
        n = len(X) - 1

        # Build the Vandermonde matrix
        V = np.array([[self._generateBasis(X[i], j) for j in range(n + 1)] for i in range(n + 1)])

        # Build the vector of the interpolation points computed on the function
        u = func(X)

        # Build the matrix of all the points where we want to compute the interpolation
        B = np.array([[self._generateBasis(x[i], j) for j in range(n + 1)] for i in range(len(x))])

        # Solve the linear system
        p = np.linalg.solve(V, u)
        
        return B, V, u, p