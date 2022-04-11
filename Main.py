# -*- coding: utf-8 -*-

from models.lorenz96 import Lorenz96
from background.background import Background
from analysis.AnalysisEnKF import AnalysisEnKF
from observation.observation import Observation
from simulation.simulation import Simulation

import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = Lorenz96();
    background = Background(model,ensemble_size=20);
    analysis = AnalysisEnKF();
    observation = Observation(m=32,std_obs=0.01);
    params={'obs_freq':0.1,'obs_times':15,'inf_fact':2};
    simulation = Simulation(model,background,analysis,observation,params=params);
    simulation.run();


    errb,erra = simulation.geterrors();

    plt.figure(figsize=(12, 10))
    plt.plot(errb,'-ob')
    plt.plot(erra,'-or')

