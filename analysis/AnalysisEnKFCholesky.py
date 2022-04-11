# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis


class AnalysisEnKFCholesky(Analysis):
  """EnKF implementation Cholesky (ensemble space)
  
    Attributes
    ----------
    None

    Methods
    -------
    performassimilation(background, observation)
      Perform assimilation step given background and observations
    getanalysisstate()
      Returns the computed column mean of ensemble Xa
    getensemble()
      Returns ensemble Xa
    geterrorcovariance()
      Returns the computed covariance matrix of the ensemble Xa
    inflateensemble(inflation_factor)
      Computes new ensemble Xa given the inflation factor
  """
  def __init__(self):
    """
        Parameters
        ----------
        None
    """
    pass;
  
  def performassimilation(self, background, observation):
    """Perform assimilation step of ensemble Xa given the background and the 
    observations

        Parameters
        ----------
        background : Background Object
            The background object defined in the class background
        observation : Observation Object
            The observation object defined in the class observation
        
        Returns
        -------
        Xa : Matrix of ensemble
    """
    Xb = background.getensemble();
    y = observation.getobservation();
    H = observation.getobservationoperator();
    R = observation.getdataerrorcovariance();
    n, ensemble_size = Xb.shape; 
    Rinv = np.diag(np.reciprocal(np.diag(R)));
    Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T;
    D = Ys-H@Xb;
    xb = np.mean(Xb,axis=1);
    DX = Xb-np.outer(xb,np.ones(ensemble_size));
    Q = H@DX;
    IN = (ensemble_size-1)*np.eye(ensemble_size,ensemble_size) + Q.T@(Rinv@Q);
    L = np.linalg.cholesky(IN); 
    DG = Q.T@(Rinv@D); 
    ZG = sci.linalg.solve_triangular(L,DG,lower=True); 
    Z = sci.linalg.solve_triangular(L,ZG,trans='T',lower=True); 
    self.Xa = Xb + DX@Z;
    return self.Xa;
  
  def getanalysisstate(self):
    """Compute column-wise mean vector of Matrix of ensemble Xa

        Parameters
        ----------
        None

        Returns
        -------
        mean vector
    """
    return np.mean(self.Xa,axis=1);

  def getensemble(self):
    """Returns ensemble Xa

        Parameters
        ----------
        None

        Returns
        -------
        Ensemble matrix
    """
    return self.Xa;
  
  def geterrorcovariance(self):
    """Returns the computed covariance matrix of the ensemble Xa

        Parameters
        ----------
        None

        Returns
        -------
        covariance matrix of the ensemble Xa
    """
    return np.cov(self.Xa);

  def inflateensemble(self,inflation_factor):
    """Computes ensemble Xa given the inflation factor

        Parameters
        ----------
        inflation_factor : int
          double number indicating the inflation factor

        Returns
        -------
        None
    """
    n,ensemble_size = self.Xa.shape;
    xa = self.getanalysisstate();
    DXa = self.Xa-np.outer(xa,np.ones(ensemble_size));
    self.Xa = np.outer(xa,np.ones(ensemble_size))+inflation_factor*DXa;
