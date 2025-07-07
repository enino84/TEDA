# 🧪 Official method: 'enkf-mc' with r=2 using AnalysisFactory and QGModel
import numpy as np
import matplotlib.pyplot as plt

from pyteda.simulation import Simulation
from pyteda.models import QGModel  
from pyteda.background import Background
from pyteda.observation import Observation
from pyteda.analysis.analysis_factory import AnalysisFactory

# Create QG model instance
model = QGModel(N=32, dt=0.001, F=1600.0)  # Adjust N for feasibility if needed

# Background ensemble
background = Background(model, ensemble_size=40)

# Observation: observe m points out of N*N total
observation = Observation(m=400, std_obs=0.01)  # 👈 m must be < N*N (e.g. 400 out of 1024)

# Simulation parameters
params = {
    'obs_freq': 0.1,     # Frequency of observations
    'obs_times': 20,     # Number of observation cycles
    'inf_fact': 1.04     # Inflation factor
}

analysis = AnalysisFactory("enkf-modified-cholesky", model=model).create_analysis()

# Run simulation
sim = Simulation(model, background, analysis, observation, params=params)
sim.run()

# Plot errors
errb, erra = sim.get_errors()
plt.figure(figsize=(10, 6))
plt.plot(np.log10(errb), '-ob', label='Background Error')
plt.plot(np.log10(erra), '-or', label='Analysis Error')
plt.title("EnKF via a Modified Cholesky decomposition with QGModel – Log10 Errors")
plt.legend()
plt.grid(True)
plt.show()
