# -*- coding: utf-8 -*-
import numpy as np
import logging
import matplotlib.pyplot as plt

from teda import Simulation
from teda.models import Lorenz96
from teda.background import Background
from teda.analysis import (
    AnalysisEnKF,
    AnalysisEnKFBLoc,
    AnalysisEnKFModifiedCholesky,
    AnalysisEnKFCholesky,
    AnalysisEnKFNaive,
    AnalysisLEnKF,
    AnalysisEnSRF,
    AnalysisETKF,
    AnalysisLETKF,
    AnalysisEnKFShrinkagePrecision
)
from teda.observation import Observation

if __name__ == '__main__':
    model = Lorenz96()
    background = Background(model, ensemble_size=20)

    # Choose one analysis method
    # analysis = AnalysisEnKF()
    # analysis = AnalysisEnKFBLoc(model)
    # analysis = AnalysisEnKFModifiedCholesky(model, r=2)
    # analysis = AnalysisEnKFCholesky()
    # analysis = AnalysisEnKFNaive()
    # analysis = AnalysisLEnKF(model)
    # analysis = AnalysisEnSRF()
    # analysis = AnalysisETKF()
    # analysis = AnalysisLETKF(model)
    analysis = AnalysisEnKFShrinkagePrecision(model, r=1)

    observation = Observation(m=32, std_obs=0.01)
    params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}

    sim = Simulation(model, background, analysis, observation, params=params, log_level=None)
    sim.run()

    errb, erra = sim.get_errors()

    # Plotting the relative errors
    plt.figure(figsize=(12, 10))
    plt.plot(np.log10(errb), '-ob', label='Background Error')
    plt.plot(np.log10(erra), '-or', label='Analysis Error')
    plt.title('Log10 of Relative Errors')
    plt.xlabel('Time Step')
    plt.ylabel('Log10 of Relative Error')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, len(errb), 1))
    plt.xlim(0, len(errb) - 1)
    plt.ylim(-5, 0)
    plt.tight_layout()
    plt.show()
