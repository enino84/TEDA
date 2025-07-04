# -*- coding: utf-8 -*-
import numpy as np
from models.lorenz96 import Lorenz96
from background.background import Background
from analysis.analysis_enkf import AnalysisEnKF
from analysis.analysis_enkf_bloc import AnalysisEnKFBLoc
from analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from analysis.analysis_enkf_cholesky import AnalysisEnKFCholesky
from analysis.analysis_enkf_naive import AnalysisEnKFNaive
from analysis.analysis_lenkf import AnalysisLEnKF
from analysis.analysis_ensrf import AnalysisEnSRF
from analysis.analysis_etkf import AnalysisETKF
from analysis.analysis_letkf import AnalysisLETKF
from analysis.analysis_enkf_shrinkage_precision import AnalysisEnKFShrinkagePrecision
from observation.observation import Observation
from simulation.simulation import Simulation
import logging

import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = Lorenz96()
    background = Background(model, ensemble_size=20)
    #analysis = AnalysisEnKF()
    #analysis = AnalysisEnKFBLoc(model)
    #analysis = AnalysisEnKFModifiedCholesky(model, r=2)
    #analysis = AnalysisEnKFCholesky()
    #analysis = AnalysisEnKFNaive()
    #analysis = AnalysisLEnKF(model)
    #analysis = AnalysisEnSRF()
    #analysis = AnalysisETKF()
    #analysis = AnalysisLETKF(model)
    analysis = AnalysisEnKFShrinkagePrecision(model, r=1)
    observation = Observation(m=32, std_obs=0.01)
    params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}
    sim = Simulation(model, background, analysis, observation, log_level=None)
    sim.run()

    sim.run()



    errb,erra = sim.get_errors();


    # You then can plot the errors using matplotlib
    plt.figure(figsize=(12, 10))
    plt.plot(np.log10(errb),'-ob')
    plt.plot(np.log10(erra),'-or')
    plt.title('Log10 of Relative Errors')
    plt.xlabel('Time Step')
    plt.ylabel('Log10 of Relative Error')
    plt.legend(['Background Error', 'Analysis Error'])
    plt.grid(True)
    plt.xticks(np.arange(0, len(errb), 1))
    plt.xlim(0, len(errb) - 1)
    plt.ylim(-5, 0)
    plt.tight_layout()
    plt.show()

