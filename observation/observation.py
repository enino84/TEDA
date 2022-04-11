# -*- coding: utf-8 -*-

import numpy as np

class Observation:
  """Class that generates and handles the observations.
  
    Attributes
    ----------
    m : int
      number of observations
    std_obs : int, optional
      standard deviation of the observations
    obs_operator_fixed : boolean, optional
      value that defines if there exists an observational operator. 
    H : Matrix, optional
      observation operator
      
    Methods
    -------
    setobservationoperator(n)
      Create observational operator H.
    generateobservation(x)
      Generates observations based on the obs_operator_fixed value.
    getobservation()
      Returns the observation vector y.
    getobservationoperator()
      Returns observational operator H.
    getdataerrorcovariance()
      Returns the data error covariance matrix.
    getprecisionerrorcovariance()
      Returns the reciprocal diagonal of the matrix R.
  """
  def __init__(self,m,std_obs=0.01,obs_operator_fixed=False,H=None):
    """
        Parameters
        ----------
           m : int
            number of observations
          std_obs : int, optional
            standard deviation of the observations
          obs_operator_fixed : boolean, optional
            value that defines if there exists an observational operator. 
          H : Matrix, optional
            observation operator
    """
    self.m = m;
    self.H = H;
    self.R = (std_obs**2)*np.eye(self.m,self.m);
    self.obs_operator_fixed = obs_operator_fixed;

  def setobservationoperator(self,n):
    """Create observational operator H.

        Parameters
        ----------
        n : int
          number of samples.
        Returns
        -------
        Observational operator H.
    """
    I = np.eye(n,n);
    H = np.random.choice(np.arange(0,n),self.m,replace=False);
    H = I[H,:];
    self.H = H;

  def generateobservation(self,x):
    """Generates observations based on the obs_operator_fixed value.
        if there is not obs_operator_fixed then the method
        setobservationoperator is call, if there is then the observations
        are calculated.

        Parameters
        ----------
        x : vector
          samples.
        Returns
        -------
        Observations
    """
    if not self.obs_operator_fixed:
      self.setobservationoperator(x.size);
    self.y = self.H@x + np.random.multivariate_normal(np.zeros(self.m),self.R);
  
  def getobservation(self):
    """Returns the observation vector.

      Parameters
      ----------
        None
      Returns
      -------
       Observation vector y.
    """
    return self.y;

  def getobservationoperator(self):
    """Returns the observational operator.

      Parameters
      ----------
        None
      Returns
      -------
       Observational operator H.
    """
    return self.H;

  def getdataerrorcovariance(self):
    """Returns the data error covariance matrix.

      Parameters
      ----------
        None
      Returns
      -------
       Data error covariance matrix R.
    """
    return self.R;
  
  def getprecisionerrorcovariance(self):
    """Returns the reciprocal diagonal of the matrix R.

      Parameters
      ----------
        None
      Returns
      -------
       Reciprocal diagonal of the matrix R.
    """
    return np.diag(np.reciprocal(np.diag(R)));
