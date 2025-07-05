# -*- coding: utf-8 -*-

from scipy.integrate import odeint
import numpy as np
from .model import Model

class Lorenz96(Model):
    """Implementation of the Lorenz 96 model"""

    def __init__(self, n=40, F=8):
        """
        Initialize the Lorenz96 model.

        Parameters
        ----------
        n : int, optional
            Number of variables (default is 40).
        F : int, optional
            Forcing constant (default is 8).
        """
        self.n = n
        self.F = F
        self._L = None  # Decorrelation matrix

    def lorenz96(self, x, t):
        """
        Computes the Lorenz96 dynamical system.

        Parameters
        ----------
        x : array-like
            State of the system.
        t : float
            Timestamp.

        Returns
        -------
        array-like
            Dynamical model.
        """
        n = self.n
        F = self.F
        return [(x[np.mod(i+1, n)] - x[i-2]) * x[i-1] - x[i] + F for i in range(n)]

    def get_number_of_variables(self):
        """Returns the number of variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self.n

    def get_initial_condition(self, seed=10, T=np.arange(0, 10, 0.1)):
        """Computes the initial values to propagate the model.

        Parameters
        ----------
        seed : int, optional
            Seed used to generate the initial conditions (default is 10).
        T : array-like, optional
            Timestamp vector used for propagation (default is np.arange(0, 10, 0.1)).

        Returns
        -------
        array-like
            Propagation of the model.
        """
        np.random.seed(seed=seed)
        x0 = np.random.randn(self.n)
        return self.propagate(x0, T)

    def propagate(self, x0, T, just_final_state=True):
        """Solves a system of ordinary differential equations using x0 as initial conditions.

        Parameters
        ----------
        x0 : array-like
            Initial conditions.
        T : array-like
            Timestamp vector used for propagation.
        just_final_state : bool, optional
            Determines whether to return just the final state or all states (default is True).

        Returns
        -------
        array-like
            Final state or all states.
        """
        x1 = odeint(self.lorenz96, x0, T)
        if just_final_state:
            return x1[-1, :]
        else:
            return x1

    def create_decorrelation_matrix(self, r):
        """Create L matrix by removing correlations.

        Parameters
        ----------
        r : int
            Value used in the process of removing correlations.

        Returns
        -------
        array-like
            Matrix with correlations removed.
        """
        n = self.n
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                dij = np.min([np.abs(i-j), np.abs((n-1)-j+i)])
                L[i, j] = (dij**2) / (2 * r**2)
                L[j, i] = L[i, j]
        self._L = np.exp(-L)

    def get_decorrelation_matrix(self):
        """Get the decorrelation matrix.

        Returns
        -------
        array-like
            Decorrelation matrix.
        """
        return self._L
    
    def get_ngb(self, i, r):
        return np.arange(i-r,i+r+1)%(self.n)
    
    def get_pre(self, i, r):
        ngb = self.get_ngb(i, r)
        return ngb[ngb<i]
