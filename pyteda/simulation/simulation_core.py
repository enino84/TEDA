# -*- coding: utf-8 -*-
import numpy as np
import logging

class Simulation:
    """Class that runs the simulation and calculates the errors."""

    def __init__(self, model, background, analysis, observation,
                 params={'obs_freq': 0.1, 'end_time': 15, 'inf_fact': 1.04},
                 log_level=logging.INFO):
        self.model = model
        self.background = background
        self.analysis = analysis
        self.observation = observation
        self.obs_freq = params['obs_freq']
        self.end_time = params['end_time']
        self.inf_fact = params['inf_fact']
        self.store_back_state = params.get('store_back_state', False)
        self.store_post_state = params.get('store_post_state', False)
        self.store_ref_state = params.get('store_ref_state', False)
        self.store_state_at = params.get('store_state_at', [])  # new: which time steps to store

        # Optional storage for state snapshots
        self.background_states = [] if self.store_back_state else None
        self.analysis_states = [] if self.store_post_state else None
        self.truth_states = [] if self.store_ref_state else None
        self._stored_indices = []  # new: actual time indices saved

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
        self.error_a = []
        self.error_b = []

        xtk = self.model.get_initial_condition()
        Xbk = self.background.get_initial_ensemble()
        T = np.linspace(0, self.obs_freq, num=2)
        t = 0

        while t<=self.end_time:
            self.logger.info(f"Time step {t} - {self.end_time}")

            self.observation.generate_observation(xtk)
            Xak = self.analysis.perform_assimilation(self.background, self.observation)

            if self.inf_fact > 0:
                self.analysis.inflate_ensemble(self.inf_fact)
                self.logger.debug(f"Inflated ensemble with factor {self.inf_fact}")

            xak = self.analysis.get_analysis_state()
            xbk = self.background.get_background_state()

            self.error_a.append(self.relative_error(xtk, xak))
            self.error_b.append(self.relative_error(xtk, xbk))

            self.logger.debug(f"Background error: {self.error_b[-1]:.4f}, Analysis error: {self.error_a[-1]:.4f}")

            # ✅ Only store if current step is in store_state_at
            if t in self.store_state_at:
                if self.store_back_state:
                    self.background_states.append(xbk.copy())
                if self.store_post_state:
                    self.analysis_states.append(xak.copy())
                if self.store_ref_state:
                    self.truth_states.append(xtk.copy())
                self._stored_indices.append(t)

            # Forecast step
            Xbk = self.background.forecast_step(Xak, T)
            xtk = self.model.propagate(xtk, T)

            t+= self.obs_freq

        self.error_a = np.array(self.error_a)
        self.error_b = np.array(self.error_b)

        self.logger.info("Simulation completed.")

    def get_errors(self):
        """Returns the background and analysis error vectors."""
        return self.error_b, self.error_a

    def get_saved_states(self):
        """Returns stored states as a dictionary."""
        return {
            "background": self.background_states,
            "analysis": self.analysis_states,
            "truth": self.truth_states,
            "steps": self._stored_indices
        }
