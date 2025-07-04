from abc import ABC, abstractmethod

class Analysis(ABC):
    """Abstract class for analysis methods."""

    @abstractmethod
    def perform_assimilation(self, background, observation):
        """Perform the assimilation step given the background and observations."""
        pass

    @abstractmethod
    def get_analysis_state(self):
        """Return the computed column mean of the ensemble Xa."""
        pass

    @abstractmethod
    def get_ensemble(self):
        """Return the ensemble Xa."""
        pass

    @abstractmethod
    def get_error_covariance(self):
        """Return the computed covariance matrix of the ensemble Xa."""
        pass

    @abstractmethod
    def inflate_ensemble(self, inflation_factor):
        """Compute the new ensemble Xa given the inflation factor."""
        pass