# -*- coding: utf-8 -*-

import abc
class Model(metaclass=abc.ABCMeta):
  """Model
  An Abstract class used to set the minimum methods that need to be defined
  when creating a new model

    Attributes
    ----------
    None

    Methods
    -------
    getinitialcondition()
      None
    propagate()
      None
    propagate()
      None
    getnumberofvariables()
      None
    createlocalizationmatrix()
      None
    getlocalizationmatrix()
      None
    getprecisionmatrix()
      None
  """
  def __init__(self):
    """
        Parameters
        ----------
        None
    """
    pass;
  def getinitialcondition():
    """
        Parameters
        ----------
        None
    """
    pass;
  def propagate():
    """
        Parameters
        ----------
        None
    """
    pass;
  def getnumberofvariables():
    """
        Parameters
        ----------
        None
    """
    pass;
  def createlocalizationmatrix():
    """
        Parameters
        ----------
        None
    """
    pass;
  def getlocalizationmatrix():
    """
        Parameters
        ----------
        None
    """
    pass;
  def getprecisionmatrix():
    """
        Parameters
        ----------
        None
    """
    pass;