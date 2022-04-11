# -*- coding: utf-8 -*-

from scipy.integrate import odeint
import numpy as np
from models.model import Model

class Lorenz96(Model):
  """Implementation of Lorenz 96 model
  
    Attributes
    ----------
    n : int
      number of variables
    F : int
      the forcing constant

    Methods
    -------
    lorenz96(x, t)
      Computes the dynamical system.
    getnumberofvariables()
      Returns the number of variables.
    getinitialcondition(seed, T)
      Computes the initial values that are going to be used to propagate the
      model.
    propagate(x0, T, just_final_state)
      Solves a system of orninary differential equations using x0 has initial
      conditions with the sequence of time points T.
    createdecorrelatonmatrix(r)
      Create L matrix removing correlation.
    getdecorrelationmatrix()
      Get L matrix.
    getngb()

    getpre()

  """

  n = 40;
  F = 8;

  def __init__(self,n = 40,F = 8):
    """
    Parameters
    ----------
      n : int
        number of variables
      F : int
        the forcing constant
    """
    self.n = n;
    self.F = F;

  def lorenz96(self, x, t):
    """
        Computes the dynamical system.
        Parameters
        ----------
        x : int
          State of the system
        t : int
          timestamp
          value used as alpha in the ridge model

        Returns
        -------
        Dynamical model
    """
    n = self.n;
    F = self.F;
    return [(x[np.mod(i+1,n)]-x[i-2])*x[i-1]-x[i]+F for i in range(0,n)];

  def getnumberofvariables(self):
    """Returns the number of variables.
        Parameters
        ----------
        None

        Returns
        -------
        number of variables
    """
    return self.n;
  
  def getinitialcondition(self, seed = 10, T = np.arange(0,10,0.1)):
    """Computes the initial values that are going to be used to 
        propagate the model.
        Parameters
        ----------
        seed : int 
          Seed to be used to get the initial conditions.
        T : timestamp vector
          Timestamp that is going to be used to propagate the model.

        Returns
        -------
        Propagation of the model.
    """
    n = self.n;
    np.random.seed(seed=10);
    x0 = np.random.randn(n);
    return self.propagate(x0,T);

  def propagate(self, x0, T, just_final_state=True):
    """Solves a system of orninary differential equations using x0 has 
       initial conditions with the sequence of time points T.
        Parameters
        ----------
        x0 : int 
          Seed to be used to get the initial conditions.
        T : timestamp vector
          Timestamp that is going to be used to propagate the model.
        just_final_state : boolean
          variable that decides what will be returned. if just the final
          state or all the states.

        Returns
        -------
        final or all states.
    """
    x1 = odeint(self.lorenz96,x0,T);
    if just_final_state:
      return x1[-1,:];
    else:
      return x1;
    
  def createdecorrelatonmatrix(self,r):
    """Create L matrix removing correlation.
        Parameters
        ----------
        r : int
          value used in the process of removing correlations

        Returns
        -------
        matrix with correlations removed
    """
    n = self.n;
    L = np.zeros((n,n));
    for i in range(0,n):
      for j in range(i,n):
        dij = np.min([np.abs(i-j),np.abs((n-1)-j+i)]);
        L[i,j] = (dij**2)/(2*r**2);
        L[j,i] = L[i,j];
    self.L = np.exp(-L);
    
  def getdecorrelatonmatrix(self):
    """Get L matrix.
        Parameters
        ----------
        None

        Returns
        -------
        L matrix.
    """
    return self.L;
  
  def getngb(self,i):
    return np.arange(i-self.r,i+self.r+1)%(self.n);
  
  def getpre(self,i):
    ngb = self.getngb(i);
    return ngb[ngb<i];
    