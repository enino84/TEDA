# -*- coding: utf-8 -*-

import numpy as np

class Background:
    """
    Background matrix Xb of the simulation.
  
    Attributes
    ----------
    model : Model object
        An object that has all the methods and attributes of the given model
    ensemble_size : int, optional
        Value used to indicate the size of the ensemble for the simulation
      
    Methods
    -------
    get_initial_ensemble(initial_perturbation, time)
        Constructs the initial background matrix Xb.
    forecast_step(Xb, time)
        Propagates Xb for a given time.
    get_ensemble_size()
        Returns the ensemble size.
    get_ensemble()
        Returns the current Xb matrix.
    get_covariance_matrix()
        Returns the computed covariance matrix of the ensemble Xb.
    get_background_state()
        Computes the column-wise mean vector of the background matrix Xb.
    get_member_deviations(scale)
        Computes the ensemble matrix Xb deviations.
    """

    def __init__(self, model, ensemble_size=200):
        """
        Initialize the Background object.

        Parameters
        ----------
        model : Model object
            An object that has all the methods and attributes of the given model.
        ensemble_size : int, optional
            Value used to indicate the size of the ensemble for the simulation.
        """
        self.model = model
        self.ensemble_size = ensemble_size

    def get_initial_ensemble(self, initial_perturbation=0.05, time=np.arange(0,10,0.01)):
        """
        Construct the initial background matrix Xb.

        Parameters
        ----------
        initial_perturbation : int, optional
            Value used when calculating the initial background matrix.
        time : np.array, optional
            Value used when calculating the initial background matrix.

        Returns
        -------
        Xb : matrix
            Background Xb matrix.
        """
        n = self.model.get_number_of_variables()
        Xb = initial_perturbation * np.random.randn(n, self.ensemble_size)
        M = len(time)
        for e in range(0, self.ensemble_size):
            Xb[:, e] = self.model.propagate(Xb[:, e], time)
        self.Xb = Xb
        self.Xb0 = Xb
        return Xb
  
    def forecast_step(self, Xb, time=np.arange(0,1,0.01)):
        """
        Propagate the background ensemble matrix Xb for a given timestamp.

        Parameters
        ----------
        Xb : matrix
            Background matrix at a given state.
        time : np.array, optional
            Value used when calculating the propagation of the background matrix.

        Returns
        -------
        Xb : matrix
            Background Xb matrix.
        """
        ensemble_size = self.ensemble_size
        for e in range(0, ensemble_size):
            Xb[:, e] = self.model.propagate(Xb[:, e], time)
        self.Xb = Xb
        return Xb

    def get_ensemble_size(self):
        """
        Return the ensemble size.

        Returns
        -------
        ensemble_size : int
            Ensemble size value.
        """
        return self.ensemble_size
  
    def get_ensemble(self):
        """
        Return the ensemble matrix Xb.

        Returns
        -------
        Xb : matrix
            Background Xb matrix.
        """
        return self.Xb
  
    def get_covariance_matrix(self):
        """
        Return the computed covariance matrix of the matrix Xb.

        Returns
        -------
        covariance_matrix : matrix
            Covariance matrix of matrix Xb.
        """
        return np.cov(self.Xb)
  
    def get_background_state(self):
        """
        Compute the column-wise mean vector of the background matrix Xb.

        Returns
        -------
        mean_vector : array
            Column-wise mean vector.
        """
        return np.mean(self.Xb, axis=1)
    
    def get_member_deviations(self, scale=1):
        """
        Compute the ensemble matrix Xb deviations.

        Parameters
        ----------
        scale : int, optional
            Value used in the formula to get the deviation of the members.

        Returns
        -------
        deviations : matrix
            Ensemble matrix Xb deviations.
        """
        return scale * (self.Xb - np.outer(self.get_background_state(), np.ones(self.get_ensemble_size())))
