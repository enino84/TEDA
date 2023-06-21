# -*- coding: utf-8 -*-

import numpy as np

class Observation:
    """Class that generates and handles observations.

    Attributes
    ----------
    m : int
        Number of observations.
    std_obs : float, optional
        Standard deviation of the observations.
    obs_operator_fixed : bool, optional
        Value that defines if there exists an observational operator.
    H : ndarray, optional
        Observation operator.

    Methods
    -------
    set_observation_operator(n)
        Create the observational operator H.
    generate_observation(x)
        Generates observations based on the obs_operator_fixed value.
    get_observation()
        Returns the observation vector y.
    get_observation_operator()
        Returns the observational operator H.
    get_data_error_covariance()
        Returns the data error covariance matrix.
    get_precision_error_covariance()
        Returns the reciprocal diagonal of the matrix R.
    """

    def __init__(self, m, std_obs=0.01, obs_operator_fixed=False, H=None):
        """
        Initialize the Observation class.

        Parameters
        ----------
        m : int
            Number of observations.
        std_obs : float, optional
            Standard deviation of the observations (default is 0.01).
        obs_operator_fixed : bool, optional
            Value that defines if there exists an observational operator.
        H : ndarray, optional
            Observation operator.
        """
        self.m = m
        self.H = H
        self.R = (std_obs**2) * np.eye(self.m, self.m)
        self.obs_operator_fixed = obs_operator_fixed

    def set_observation_operator(self, n):
        """Create the observational operator H.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        ndarray
            Observational operator H.
        """
        I = np.eye(n, n)
        H = np.random.choice(np.arange(0, n), self.m, replace=False)
        H.sort()
        self.H_index = H # for local filters
        H = I[H, :]
        self.H = H
        

    def generate_observation(self, x):
        """Generates observations based on the obs_operator_fixed value.
        If obs_operator_fixed is False, then the method set_observation_operator is called.
        If obs_operator_fixed is True, the observations are calculated.

        Parameters
        ----------
        x : ndarray
            Samples.

        Returns
        -------
        None
        """
        if not self.obs_operator_fixed:
            self.set_observation_operator(x.size)
        self.y = self.H @ x + np.random.multivariate_normal(np.zeros(self.m), self.R)

    def get_observation(self):
        """Returns the observation vector.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Observation vector y.
        """
        return self.y

    def get_observation_operator(self):
        """Returns the observational operator.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Observational operator H.
        """
        return self.H
    
    def get_observation_operator_index(self):
        return self.H_index

    def get_data_error_covariance(self):
        """Returns the data error covariance matrix.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Data error covariance matrix R.
        """
        return self.R

    def get_precision_error_covariance(self):
        """Returns the reciprocal diagonal of the matrix R.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Reciprocal diagonal of the matrix R.
        """
        return np.diag(np.reciprocal(np.diag(self.R)))
