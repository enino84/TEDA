# -*- coding: utf-8 -*-

import numpy as np

class Background:
  """Background matrix Xb of the simulation.
  
    Attributes
    ----------
    model : Model object
      An object that has all the methods and attributes of the model given
    ensemble_size : int, optional
      value used to indicate the size of the ensemble for the simulation
      
    Methods
    -------
    getinitialensemble(initial_perturbation , time)
      Constructs the initial background matrix Xb.
    forecaststep(Xb, time)
      Propagates Xb a time given.
    getensemblesize()
      Returns ensemble size.
    getensemble()
      Returns current Xb matrix.
    getcovariancematrix()
      Returns the computed covariance matrix of the ensemble Xb
    getbackgroundstate()
      Compute column-wise mean vector of the background Matrix Xb
    getmemberdeviations(scale)
      Compute ensemble matrix Xb deviatio
  """
  def __init__(self, model, ensemble_size=200):
    """
        Parameters
        ----------
        model : Model object
          An object that has all the methods and attributes of the model given
        ensemble_size : int, optional
          value used to indicate the size of the ensemble for the simulation
    """
    self.model = model;
    self.ensemble_size = ensemble_size;

  def getinitialensemble(self, initial_perturbation = 0.05, time = np.arange(0,10,0.01)):
    """Constructs the initial background matrix Xb.

        Parameters
        ----------
        initial_perturbation : int, optional
          value to be used when calculating the initial background matrix.
        time : np.array, optional
          value used when calculating the initial background matrix.
        Returns
        -------
        background Xb matrix
    """
    n = self.model.getnumberofvariables();
    Xb = initial_perturbation*np.random.randn(n, self.ensemble_size);
    M = len(time);
    for e in range(0,self.ensemble_size):
      Xb[:,e] = self.model.propagate(Xb[:,e], time);
    self.Xb = Xb;
    return Xb;
  
  def forecaststep(self, Xb, time = np.arange(0,1,0.01)):
    """ Propagates background ensemble matrix Xb given a timestamp.

        Parameters
        ----------
        Xb : Matrix
          background matrix at a given state.
        time : np.array, optional
          value used when calculating the propagation of the background matrix.
        Returns
        -------
        background Xb matrix
    """
    ensemble_size = self.ensemble_size;
    for e in range(0,ensemble_size):
      Xb[:,e] = self.model.propagate(Xb[:,e], time);
    self.Xb = Xb;
    return Xb;

  def getensemblesize(self):
    """Returns ensemble size.

      Parameters
      ----------
        None
      Returns
      -------
        ensemble size value.
    """
    return self.ensemble_size;
  
  def getensemble(self):
    """Returns ensemble matrix Xb.

      Parameters
      ----------
        None
      Returns
      -------
        background Xb matrix
    """
    return self.Xb;
  
  def getcovariancematrix(self):
    """Returns the computed covariance matrix of the matrix Xb

        Parameters
        ----------
        None

        Returns
        -------
        covariance matrix of matrix Xb
    """
    return np.cov(self.Xb);
  
  def getbackgroundstate(self):
    """Compute column-wise mean vector of the background Matrix Xb

      Parameters
      ----------
        None
      Returns
      -------
        mean vector
    """
    return np.mean(self.Xb,axis=1);
    
  def getmemberdeviations(self,scale=1):
    """Compute ensemble matrix Xb deviatio

      Parameters
      ----------
        scale : int, optional
          value used in the formula to get the deviation of the members
      Returns
      -------
        ensemble matrix Xb deviatio
    """
    return scale*(self.Xb-np.outer(self.getbackgroundstate(),np.ones(self.getensemblesize())));
