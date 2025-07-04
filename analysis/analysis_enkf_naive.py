# -*- coding: utf-8 -*-

import numpy as np

from analysis.analysis import Analysis


class AnalysisEnKFNaive(Analysis):
    def __init__(self, **kwargs):
        pass

    def perform_assimilation(self, background, observation):
        Xb = background.get_ensemble()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()
        n, ensemble_size = Xb.shape
        Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T
        D = Ys - H @ Xb
        xb = np.mean(Xb, axis=1)
        DX = Xb - np.outer(xb, np.ones(ensemble_size))
        DXG = 1 / (np.sqrt(ensemble_size - 1)) * DX
        Q = H @ DXG
        IN = R + Q @ Q.T
        Z = np.linalg.solve(IN, D)
        self.Xa = Xb + DXG @ (Q.T @ Z)
        return self.Xa

    def get_analysis_state(self):
        return np.mean(self.Xa, axis=1)

    def inflate_ensemble(self, inflation_factor):
        """Computes ensemble Xa given the inflation factor

        Parameters
        ----------
        inflation_factor : int
            A double number indicating the inflation factor

        Returns
        -------
        None
        """
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa

    def get_ensemble(self):
        """Returns ensemble Xa

        Parameters:
            None

        Returns:
            ensemble_matrix: Ensemble matrix
        """
        return self.Xa
  
    def get_error_covariance(self):
        """Returns the computed covariance matrix of the ensemble Xa

        Parameters:
            None

        Returns:
            covariance_matrix: None - we avoid computing the full covariance in this implementation
        """
        return None

