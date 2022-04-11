# -*- coding: utf-8 -*-

import numpy as np

class Simulation:
  """Class that runs the simulation and gets the errors.
  
    Attributes
    ----------
    model : Model object
      An object that has all the methods and attributes of the model given
    background : Background object
      An object that has all the methods and attributes of the background given
    analysis : Analysis object
      An object that has all the methods and attributes of the analysis given
    observatio : Observation object
      An object that has all the methods and attributes of the observation given
    params : dict, optional
      parameters needed for the simulation
      
    Methods
    -------
    relative_error(model, background, analysis, observation, params)
      Calculares the Root-Mean-Square Error (RMSE).
    run()
      Runs the simulation with the given parameters.
    geterrors()
      Returns the errors of the background and analysis states.
  """
  def __init__(self,model,background,analysis,observation,
               params={'obs_freq':0.1,'obs_times':15,'inf_fact':1.04}):
    """
        Parameters
        ----------
        model : Model object
          An object that has all the methods and attributes of the model given
        background : Background object
          An object that has all the methods and attributes of the background given
        analysis : Analysis object
          An object that has all the methods and attributes of the analysis given
        observatio : Observation object
          An object that has all the methods and attributes of the observation given
        params : dict, optional
          parameters needed for the simulation
    """
    self.model = model;
    self.background = background;
    self.analysis = analysis;
    self.observation = observation;
    self.obs_freq = params['obs_freq'];
    self.obs_times = params['obs_times'];
    self.inf_fact = params['inf_fact'];
  
  def relative_error(self,xr,xs):
    """Calculates the Root-Mean-Square Error (RMSE).

        Parameters
        ----------
        xr : vector
          reference vector of values.
        xs : vector
          calculated vector given assimilation step.

        Returns
        -------
        Root-Mean-Square Error (RMSE) of xr and xs.
    """
    return np.linalg.norm(xs-xr)/np.linalg.norm(xr);

  def run(self):
    """Runs simulation given the background, observation, analysis method
    and model.

        Parameters
        ----------
          None

        Returns
        -------
        None
    """
    self.error_a = np.zeros(self.obs_times);
    self.error_b = np.zeros(self.obs_times);
    background = self.background;
    observation = self.observation;
    model = self.model;
    analysis = self.analysis;

    xtk = model.getinitialcondition(); #For reference
    ensemble_size = background.getensemblesize();
    Xbk = background.getinitialensemble();

    T = np.linspace(0,self.obs_freq,num=2);

    for k in range(0,self.obs_times):
      print(k)

      #Observations
      observation.generateobservation(xtk);

      #Analysis step
      Xak = analysis.performassimilation(background,observation);
  
      #Covariance inflation
      if self.inf_fact>0:
        analysis.inflateensemble(self.inf_fact);

      #Error computation
      xak = analysis.getanalysisstate();
      self.error_a[k] = self.relative_error(xtk,xak);
      xbk = background.getbackgroundstate();
      self.error_b[k] = self.relative_error(xtk,xbk);

      #Forecast step
      Xbk = background.forecaststep(Xak,T);
      xtk = model.propagate(xtk,T);


  
  def geterrors(self):
    """Returns the background and analysis error vectors

        Parameters
        ----------
          None

        Returns
        -------
        background and analysis error of the simulation
    """
    return self.error_b, self.error_a;
