# -*- coding: utf-8 -*-

import numpy as np

from .analysis_core import Analysis
from .registry import register_analysis

@register_analysis("enkf")
class AnalysisEnKF(Analysis):
    """
    Analysis EnKF full covariance matrix
    
    Methods
    -------
    perform_assimilation(background, observation)
        Perform the assimilation step given the background and observations
    get_analysis_state()
        Return the computed column mean of the ensemble Xa
    get_ensemble()
        Return the ensemble Xa
    get_error_covariance()
        Return the computed covariance matrix of the ensemble Xa
    inflate_ensemble(inflation_factor)
        Compute the new ensemble Xa given the inflation factor
    """

    def __init__(self, **kwargs):
        """
        Initialize the AnalysisEnKF object.
        
        Parameters
        ----------
        None
        """
        pass
  
    def perform_assimilation(self, background, observation):
        """
        Perform the assimilation step of the ensemble Xa given the background and observations.
        
        Parameters
        ----------
        background : Background Object
            The background object defined in the Background class
        observation : Observation Object
            The observation object defined in the Observation class
        
        Returns
        -------
        Xa : Matrix
            Assimilated ensemble Xa
        """
        Xb = background.get_ensemble()
        Pb = background.get_covariance_matrix()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()
        Ys = np.random.multivariate_normal(y, R, size=background.ensemble_size).T
        D = Ys - H @ Xb
        IN = R + H @ (Pb @ H.T)
        Z = np.linalg.solve(IN, D)
        self.Xa = Xb + Pb @ (H.T @ Z)
        return self.Xa
  
    def get_analysis_state(self):
        """
        Compute the column-wise mean vector of the ensemble Xa.
        
        Returns
        -------
        mean vector : array
            Column-wise mean vector of Xa
        """
        return np.mean(self.Xa, axis=1)
  
    def get_ensemble(self):
        """
        Return the ensemble Xa.
        
        Returns
        -------
        Xa : matrix
            Ensemble matrix Xa
        """
        return self.Xa
  
    def get_error_covariance(self):
        """
        Return the computed covariance matrix of the ensemble Xa.
        
        Returns
        -------
        covariance matrix : matrix
            Covariance matrix of Xa
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """
        Compute the ensemble Xa given the inflation factor.
        
        Parameters
        ----------
        inflation_factor : int or float
            Double number indicating the inflation factor
        
        Returns
        -------
        None
        """
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
