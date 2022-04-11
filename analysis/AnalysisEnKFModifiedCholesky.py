# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis

class AnalysisEnKFModifiedCholesky(Analysis):
  """Analysis EnKF Modified Cholesky decomposition
  
    Attributes
    ----------
    model : Model object
      An object that has all the methods and attributes of the model given
    r : int
      value used in the process of removing correlations

    Methods
    -------
    getprecisionmatrix(DX,r,regularization_factor=0.01)
      Returns the computed precision matrix
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

  def __init__(self,model,r=1):
    """
        Parameters
        ----------
        model : Model object
        An object that has all the methods and attributes of the model given
        r : int, optional
          value used in the process of removing correlations
    """
    self.model = model;
    self.r = r;

  def getprecisionmatrix(self,DX,r,regularization_factor=0.01):
    """
        Perform calculations to get the precision matrix given the deviation
        matrix.
        Parameters
        ----------
        DX : numpy Matrix
          Deviation matrix
        r : int
          value used in the process of removing correlations
        regularization_factor : double, optional
          value used as alpha in the ridge model

        Returns
        -------
        Precision matrix
    """
    n,ensemble_size = DX.shape;
    lr = Ridge(fit_intercept=False,alpha=regularization_factor);
    L = np.eye(n);
    D = np.zeros((n,n));
    D[0,0] = 1/np.var(DX[0,:]); #We are estimating D^{-1}
    for i in range(1,n):
      ind_prede = self.model.getpre(i);
      y = DX[i,:];
      X = DX[ind_prede,:].T;
      lr_fit = lr.fit(X,y);
      err_i = y-lr_fit.predict(X);
      D[i,i] = 1/np.var(err_i);
      L[i,ind_prede] = -lr_fit.coef_;
    
    return L.T@(D@L);

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
    Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T;
    xb = np.mean(Xb,axis=1);
    DX = Xb-np.outer(xb,np.ones(ensemble_size))
    Binv = self.model.getprecisionmatrix(DX,self.r);
    D = Ys-H@Xb;
    Rinv = np.diag(np.reciprocal(np.diag(R)));
    IN = Binv + H.T@(Rinv@H);
    Z = np.linalg.solve(IN,H.T@(Rinv@D));
    self.Xa = Xb + Z;
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
