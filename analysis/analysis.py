# -*- coding: utf-8 -*-

import abc

class Analysis(metaclass=abc.ABCMeta):
  """Analysis
  An Abstract class used to set the minimum methods that need to be defined
  when creating a new analysis method

    Attributes
    ----------
    None

    Methods
    -------
    performassimilation()
      None
    getanalysisstate()
      None
    performcovarianceinflation()
      None
  """
  def __init__(self):
    """
        Parameters
        ----------
        None
    """
    pass;
  def performassimilation():
    """
        Parameters
        ----------
        None
    """
    pass;
  def getanalysisstate():
    """
        Parameters
        ----------
        None
    """
    pass;
  def performcovarianceinflation():
    """
        Parameters
        ----------
        None
    """
    pass;