# -*- coding: utf-8 -*-

import numpy as np

from analysis.analysis import Analysis

class AnalysisETKF(Analysis):
    """
    Analysis Ensemble Transform Kalman Filter (ETKF)

    Methods
    -------
    perform_assimilation(background, observation)
        Performs the assimilation step given the background and observations.
    get_analysis_state()
        Returns the computed column mean of the ensemble Xa.
    get_ensemble()
        Returns the ensemble Xa.
    get_error_covariance()
        Returns the computed covariance matrix of the ensemble Xa.
    inflate_ensemble(inflation_factor)
        Computes the new ensemble Xa given the inflation factor.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AnalysisETKF object.
        """
        pass

    def perform_assimilation(self, background, observation):
        """
        Performs the assimilation step of the ensemble Xa given the background and observations.

        Parameters
        ----------
        background : Background Object
            The background object defined in the Background class.
        observation : Observation Object
            The observation object defined in the Observation class.

        Returns
        -------
        Xa : Matrix
            Assimilated ensemble Xa.
        """
        xb = background.get_background_state()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()

        N = background.get_ensemble_size()

        DX = background.get_member_deviations()

        d = y - H @ xb
        Q = H @ DX

        projection_onto_ensemble_space = Q.T @ np.linalg.solve(R, Q)
        U, S, _ = np.linalg.svd(projection_onto_ensemble_space)
        Pa_ens_inv = U @ np.diag(1 / (S + 1)) @ U.T

        rhs = Q.T @ np.linalg.solve(R, d)
        dxa = DX @ Pa_ens_inv @ rhs
        xa = xb + dxa

        Pat_sqrt = U @ np.diag(np.sqrt(1 / (S + 1))) @ U.T

        """
        Pat = I - V.T @ np.linalg.solve(IN, V)
        U, S, V = np.linalg.svd(Pat)
        Pat_sqrt = U @ np.diag(np.sqrt(S)) @ U.T
        """

        DXA_inc = DX @ Pat_sqrt

        self.xa = xa
        self.Xa = xa.reshape(-1, 1) + DXA_inc

        return self.Xa

    def get_analysis_state(self):
        """
        Computes the column-wise mean vector of the ensemble Xa.

        Returns
        -------
        mean vector : array
            Column-wise mean vector of Xa.
        """
        return self.xa

    def get_ensemble(self):
        """
        Returns the ensemble Xa.

        Returns
        -------
        Xa : matrix
            Ensemble matrix Xa.
        """
        return self.Xa

    def get_error_covariance(self):
        """
        Returns the computed covariance matrix of the ensemble Xa.

        Returns
        -------
        covariance matrix : matrix
            Covariance matrix of Xa.
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """
        Computes the ensemble Xa given the inflation factor.

        Parameters
        ----------
        inflation_factor : int or float
            Double number indicating the inflation factor.

        Returns
        -------
        None
        """
        _, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
