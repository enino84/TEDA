# -*- coding: utf-8 -*-

import numpy as np

class Simulation:
    """Class that runs the simulation and calculates the errors.

    Attributes
    ----------
    model : Model object
        An object that has all the methods and attributes of the given model.
    background : Background object
        An object that has all the methods and attributes of the given background.
    analysis : Analysis object
        An object that has all the methods and attributes of the given analysis.
    observation : Observation object
        An object that has all the methods and attributes of the given observation.
    params : dict, optional
        Parameters needed for the simulation.

    Methods
    -------
    relative_error(xr, xs)
        Calculates the Root-Mean-Square Error (RMSE).
    run()
        Runs the simulation with the given parameters.
    get_errors()
        Returns the errors of the background and analysis states.
    """

    def __init__(self, model, background, analysis, observation,
                 params={'obs_freq': 0.1, 'obs_times': 15, 'inf_fact': 1.04}):
        """
        Parameters
        ----------
        model : Model object
            An object that has all the methods and attributes of the given model.
        background : Background object
            An object that has all the methods and attributes of the given background.
        analysis : Analysis object
            An object that has all the methods and attributes of the given analysis.
        observation : Observation object
            An object that has all the methods and attributes of the given observation.
        params : dict, optional
            Parameters needed for the simulation.
        """
        self.model = model
        self.background = background
        self.analysis = analysis
        self.observation = observation
        self.obs_freq = params['obs_freq']
        self.obs_times = params['obs_times']
        self.inf_fact = params['inf_fact']

    def relative_error(self, xr, xs):
        """Calculates the Root-Mean-Square Error (RMSE).

        Parameters
        ----------
        xr : vector
            Reference vector of values.
        xs : vector
            Calculated vector given the assimilation step.

        Returns
        -------
        Root-Mean-Square Error (RMSE) of xr and xs.
        """
        return np.linalg.norm(xs - xr) / np.linalg.norm(xr)

    def run(self):
        """Runs the simulation given the background, observation, analysis method, and model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.error_a = np.zeros(self.obs_times)
        self.error_b = np.zeros(self.obs_times)
        background = self.background
        observation = self.observation
        model = self.model
        analysis = self.analysis

        xtk = model.get_initial_condition()  # For reference
        ensemble_size = background.get_ensemble_size()
        Xbk = background.get_initial_ensemble()

        T = np.linspace(0, self.obs_freq, num=2)

        for k in range(0, self.obs_times):
            print(k)

            # Observations
            observation.generate_observation(xtk)

            # Analysis step
            Xak = analysis.perform_assimilation(background, observation)

            # Covariance inflation
            if self.inf_fact > 0:
                analysis.inflate_ensemble(self.inf_fact)

            # Error computation
            xak = analysis.get_analysis_state()
            self.error_a[k] = self.relative_error(xtk, xak)
            xbk = background.get_background_state()
            self.error_b[k] = self.relative_error(xtk, xbk)

            # Forecast step
            Xbk = background.forecast_step(Xak, T)
            xtk = model.propagate(xtk, T)

    def get_errors(self):
        """Returns the background and analysis error vectors.

        Parameters
        ----------
        None

        Returns
        -------
        Background and analysis errors of the simulation.
        """
        return self.error_b, self.error_a
