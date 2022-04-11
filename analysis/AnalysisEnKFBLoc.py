# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis

class AnalysisEnKFBLoc(Analysis):
  """Analysis EnKF B-Loc
  
    Attributes
    ----------
    model : Model object
      An object that has all the methods and attributes of the model given
    r : int
      value used in the process of removing correlations

    Methods
    -------
    performassimilation(background, observation)
      Perform assimilation step given background and observations
    getanalysisstate()
      Returns the computed column mean of matrix Xa
    getanalysisstate()
      Returns Xa matrix
    geterrorcovariance()
      Returns the computed covariance matrix of the Xa
    inflateensemble(inflation_factor)
      Computes new matrix Xa given the inflation factor
  """
  def __init__(self,model,r=1):
    """
        Parameters
        ----------
        model : Model object
        An object that has all the methods and attributes of the model given
        r : int
          value used in the process of removing correlations
        model.createdecorrelatonmatrix(r) : matrix
          matrix with correlations removed
    """
    self.model = model;
    self.model.createdecorrelatonmatrix(r);

  def performassimilation(self, background, observation):
    """Perform assimilation step of matrix Xa given the background and the 
    observations

        Parameters
        ----------
        background : Background Object
            The background object defined in the class background
        observation : Observation Object
            The observation object defined in the class observation
        
        Returns
        -------
        Xa : Matrix of matrix
    """
    
    Xb = background.getgetensemble();
    Pb = background.getcovariancematrix();
    y = observation.getobservation();
    H = observation.getobservationoperator();
    R = observation.getdataerrorcovariance();
    n,  matrix_size = Xb.shape; 
    Ys = np.random.multivariate_normal(y, R, size= matrix_size).T;
    L = self.model.getdecorrelationmatrix();
    Pb = L*np.cov(Xb);
    D = Ys-H@Xb;
    IN = R + H@(Pb@H.T);
    Z = np.linalg.solve(IN,D);
    self.Xa = Xb + Pb@(H.T@Z);
    return self.Xa;

  def getanalysisstate(self):
    """Compute column-wise mean vector of the analysis Matrix Xa

        Parameters
        ----------
        None

        Returns
        -------
        mean vector
    """
    return np.mean(self.Xa,axis=1);

  def getanalysisstate(self):
    """Returns analysis matrix Xa

        Parameters
        ----------
        None

        Returns
        -------
        Xa matrix
    """
    return self.Xa;
  
  def geterrorcovariance(self):
    """Returns the computed covariance matrix of the matrix Xa

        Parameters
        ----------
        None

        Returns
        -------
        covariance matrix of the  matrix Xa
    """
    return np.cov(self.Xa);

  def inflateensemble(self,inflation_factor):
    """Computes matrix Xa given the inflation factor

        Parameters
        ----------
        inflation_factor : int
          double number indicating the inflation factor

        Returns
        -------
        None
    """
    n, matrix_size = self.Xa.shape;
    xa = self.getanalysisstate();
    DXa = self.Xa-np.outer(xa,np.ones( matrix_size));
    self.Xa = np.outer(xa,np.ones( matrix_size))+inflation_factor*DXa;
