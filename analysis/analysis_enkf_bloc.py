import numpy as np
from analysis.analysis import Analysis

class AnalysisEnKFBLoc(Analysis):
    """Analysis EnKF B-Loc"""

    def __init__(self, model, r=1, **kwargs):
        """
        Initialize the AnalysisEnKFBLoc object.

        Parameters
        ----------
        model : Model object
            An object that has all the methods and attributes of the given model.
        r : int, optional
            Value used in the process of removing correlations (default is 1).
        """
        self.model = model
        self.model.create_decorrelation_matrix(r)

    def perform_assimilation(self, background, observation):
        """
        Perform the assimilation step of the ensemble Xa given the background and observations.

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
        Xb = background.get_ensemble()
        Pb = background.get_covariance_matrix()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()
        n, matrix_size = Xb.shape
        Ys = np.random.multivariate_normal(y, R, size=matrix_size).T
        L = self.model.get_decorrelation_matrix()
        Pb = L * np.cov(Xb)
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
            Column-wise mean vector of Xa.
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """
        Return the ensemble Xa.

        Returns
        -------
        Xa : matrix
            Ensemble matrix Xa.
        """
        return self.Xa

    def get_error_covariance(self):
        """
        Return the computed covariance matrix of the ensemble Xa.

        Returns
        -------
        covariance matrix : matrix
            Covariance matrix of Xa.
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """
        Compute the ensemble Xa given the inflation factor.

        Parameters
        ----------
        inflation_factor : int or float
            Double number indicating the inflation factor.

        Returns
        -------
        None
        """
        n, matrix_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(matrix_size))
        self.Xa = np.outer(xa, np.ones(matrix_size)) + inflation_factor * DXa
