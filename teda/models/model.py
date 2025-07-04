from abc import ABC, abstractmethod
from scipy.integrate import odeint
import numpy as np

class Model(ABC):
    """Abstract class for models."""

    @abstractmethod
    def get_number_of_variables(self):
        """Returns the number of variables."""
        pass

    @abstractmethod
    def get_initial_condition(self):
        """Computes the initial values to propagate the model."""
        pass

    @abstractmethod
    def propagate(self, x0, T, just_final_state=True):
        """Solves a system of ordinary differential equations using x0 as initial conditions."""
        pass

    @abstractmethod
    def create_decorrelation_matrix(self, r):
        """Create L matrix by removing correlations."""
        pass

    @abstractmethod
    def get_decorrelation_matrix(self):
        """Get the decorrelation matrix."""
        pass