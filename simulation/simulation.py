# -*- coding: utf-8 -*-
import numpy as np
import logging

class Simulation:
    """Class that runs the simulation and calculates the errors."""

    def __init__(self, model, background, analysis, observation,
                 params={'obs_freq': 0.1, 'obs_times': 15, 'inf_fact': 1.04},
                 log_level=logging.INFO):
        """
        Parameters
        ----------
        model : Model object
        background : Background object
        analysis : Analysis object
        observation : Observation object
        params : dict
        log_level : logging level or None to disable logging
        """
        self.model = model
        self.background = background
        self.analysis = analysis
        self.observation = observation
        self.obs_freq = params['obs_freq']
        self.obs_times = params['obs_times']
        self.inf_fact = params['inf_fact']

        # Setup logging
        self.logger = logging.getLogger("Simulation")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.handlers = []  # Reset handlers
        self.logger.addHandler(handler)
        if log_level is not None:
            self.logger.setLevel(log_level)
        else:
            self.logger.disabled = True

    def relative_error(self, xr, xs):
        """Calculates RMSE."""
        return np.linalg.norm(xs - xr) / np.linalg.norm(xr)

    def run(self):
        """Runs the simulation."""
        self.error_a = np.zeros(self.obs_times)
        self.error_b = np.zeros(self.obs_times)

        xtk = self.model.get_initial_condition()
        Xbk = self.background.get_initial_ensemble()
        T = np.linspace(0, self.obs_freq, num=2)

        for k in range(self.obs_times):
            self.logger.info(f"Time step {k+1}/{self.obs_times}")

            self.observation.generate_observation(xtk)
            Xak = self.analysis.perform_assimilation(self.background, self.observation)

            if self.inf_fact > 0:
                self.analysis.inflate_ensemble(self.inf_fact)
                self.logger.debug(f"Inflated ensemble with factor {self.inf_fact}")

            xak = self.analysis.get_analysis_state()
            self.error_a[k] = self.relative_error(xtk, xak)
            xbk = self.background.get_background_state()
            self.error_b[k] = self.relative_error(xtk, xbk)

            self.logger.debug(f"Background error: {self.error_b[k]:.4f}, Analysis error: {self.error_a[k]:.4f}")

            Xbk = self.background.forecast_step(Xak, T)
            xtk = self.model.propagate(xtk, T)

        self.logger.info("Simulation completed.")

    def get_errors(self):
        """Returns the background and analysis error vectors."""
        return self.error_b, self.error_a
